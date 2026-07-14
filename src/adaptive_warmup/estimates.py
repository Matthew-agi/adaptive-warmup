"""Model-agnostic batch-size and learning-rate probes."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer

from .directions import OptimizerDirection, optimizer_step_direction
from .types import CriticalLREstimate, GradientNoiseEstimate


def _scalar_loss(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() != 1:
        raise TypeError(f"{name} must return one scalar torch.Tensor.")
    return value.reshape(())


def _paired_gradient_vectors(
    parameters: list[torch.nn.Parameter],
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
    *,
    max_elements: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    remaining = int(max_elements)
    first_chunks: list[torch.Tensor] = []
    second_chunks: list[torch.Tensor] = []
    for parameter, grad_a, grad_b in zip(parameters, first, second):
        take = min(remaining, int(parameter.numel()))
        if take <= 0:
            break
        if grad_a is None:
            a = torch.zeros(take, dtype=torch.float32)
        else:
            a = grad_a.detach().reshape(-1)[:take].float().cpu()
        if grad_b is None:
            b = torch.zeros(take, dtype=torch.float32)
        else:
            b = grad_b.detach().reshape(-1)[:take].float().cpu()
        first_chunks.append(a)
        second_chunks.append(b)
        remaining -= take
    if not first_chunks:
        raise ValueError("No differentiable parameters were available for the gradient probe.")
    return torch.cat(first_chunks), torch.cat(second_chunks)


def estimate_gradient_noise(
    parameters: Iterable[torch.nn.Parameter],
    loss_closure: Callable[[Any], torch.Tensor],
    batch_a: Any,
    batch_b: Any,
    *,
    batch_size: int,
    max_elements: int = 200_000,
    epsilon: float = 1e-12,
) -> GradientNoiseEstimate:
    """Estimate critical batch size from two independent minibatch gradients.

    ``loss_closure`` receives one batch and must return a scalar loss. The
    function uses ``torch.autograd.grad`` and therefore does not overwrite the
    gradients already stored on parameters for the real optimizer step.
    """

    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("parameters must contain at least one trainable parameter.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_elements <= 0:
        raise ValueError("max_elements must be positive.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    loss_a = _scalar_loss(loss_closure(batch_a), name="loss_closure")
    grads_a = torch.autograd.grad(loss_a, params, allow_unused=True)
    loss_b = _scalar_loss(loss_closure(batch_b), name="loss_closure")
    grads_b = torch.autograd.grad(loss_b, params, allow_unused=True)
    vector_a, vector_b = _paired_gradient_vectors(
        params,
        grads_a,
        grads_b,
        max_elements=max_elements,
    )
    if not torch.isfinite(vector_a).all() or not torch.isfinite(vector_b).all():
        raise FloatingPointError("The gradient probe produced a non-finite value.")

    difference = vector_a - vector_b
    mean = 0.5 * (vector_a + vector_b)
    noise = 0.5 * float(torch.dot(difference, difference))
    signal = float(torch.dot(mean, mean))
    ratio = noise / max(signal, float(epsilon))
    critical_batch = float(batch_size) * ratio
    mean_loss = 0.5 * (float(loss_a.detach()) + float(loss_b.detach()))
    if not all(math.isfinite(value) for value in (noise, signal, ratio, critical_batch, mean_loss)):
        raise FloatingPointError("The gradient-noise estimate was non-finite.")
    return GradientNoiseEstimate(
        critical_batch_size=critical_batch,
        noise=noise,
        signal=signal,
        noise_to_signal_ratio=ratio,
        batch_size=int(batch_size),
        sampled_parameters=int(vector_a.numel()),
        mean_loss=mean_loss,
    )


def _module_modes(model: torch.nn.Module | None) -> list[tuple[torch.nn.Module, bool]]:
    if model is None:
        return []
    return [(module, bool(module.training)) for module in model.modules()]


def _evaluate_loss(loss_closure: Callable[[], torch.Tensor | float]) -> float:
    with torch.no_grad():
        loss = loss_closure()
    value = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
    return value


def critical_sharpness_from_lr(critical_lr: float) -> float:
    """Convert a critical LR to critical sharpness using ``lambda_c = 2 / eta_c``."""

    value = float(critical_lr)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("critical_lr must be finite and positive.")
    return 2.0 / value


def estimate_critical_learning_rate(
    loss_closure: Callable[[], torch.Tensor | float],
    *,
    optimizer: Optimizer | None = None,
    direction: OptimizerDirection | None = None,
    model: torch.nn.Module | None = None,
    current_lr: float | None = None,
    previous_estimate: float | None = None,
    max_lr: float | None = None,
    loss_tolerance: float = 1e-4,
    expansion_steps: int = 12,
    binary_steps: int = 3,
    bracket_steps: int = 5,
) -> CriticalLREstimate:
    """Find the largest held-out non-increasing trial LR along the next update.

    The real training gradients must already be populated. The probe previews
    the upcoming optimizer direction, temporarily moves parameters along that
    direction, evaluates a deterministic held-out loss closure, and restores
    the parameters before returning.

    Built-in direction extraction supports SGD, Adam, and AdamW. Custom
    optimizers can pass an :class:`OptimizerDirection` directly.
    """

    if direction is None:
        if optimizer is None:
            raise ValueError("Provide optimizer or direction.")
        direction = optimizer_step_direction(optimizer, reference_lr=current_lr)
    reference_lr = float(direction.reference_lr)
    start_lr = reference_lr if current_lr is None else float(current_lr)
    if not math.isfinite(start_lr) or start_lr <= 0.0:
        raise ValueError("current_lr must be finite and positive.")
    if max_lr is not None and (not math.isfinite(max_lr) or max_lr <= 0.0):
        raise ValueError("max_lr must be finite and positive when supplied.")
    if loss_tolerance < 0.0:
        raise ValueError("loss_tolerance must be non-negative.")
    if expansion_steps <= 0 or binary_steps < 0 or bracket_steps < 2:
        raise ValueError(
            "expansion_steps must be positive, binary_steps non-negative, and "
            "bracket_steps at least 2."
        )

    modes = _module_modes(model)
    if model is not None:
        model.eval()
    applied_lr = 0.0
    evaluations = 0
    accepted_loss = float("nan")
    try:
        base_loss = _evaluate_loss(loss_closure)
        evaluations += 1
        if not math.isfinite(base_loss):
            raise FloatingPointError("The base held-out loss is non-finite.")
        loss_limit = base_loss + float(loss_tolerance) * max(abs(base_loss), 1e-12)

        def set_virtual_lr(target_lr: float) -> None:
            nonlocal applied_lr
            delta = float(target_lr) - applied_lr
            if delta == 0.0:
                return
            with torch.no_grad():
                for parameter, update in direction.parameters:
                    parameter.add_(update, alpha=-delta)
            applied_lr = float(target_lr)

        def evaluate_at(candidate_lr: float) -> float:
            nonlocal evaluations
            set_virtual_lr(candidate_lr)
            value = _evaluate_loss(loss_closure)
            evaluations += 1
            return value

        low = 0.0
        high: float | None = None
        probe = start_lr
        if previous_estimate is not None and math.isfinite(previous_estimate):
            probe = max(probe, float(previous_estimate))
        search_ceiling = probe * (2.0**expansion_steps)
        if max_lr is not None:
            search_ceiling = float(max_lr)
            probe = min(probe, search_ceiling)

        bracket_count = 1 if search_ceiling <= probe else int(bracket_steps)
        growth = (
            1.0
            if bracket_count == 1
            else (search_ceiling / probe) ** (1.0 / float(bracket_count - 1))
        )
        for bracket_index in range(bracket_count):
            if bracket_index == bracket_count - 1:
                probe = search_ceiling
            value = evaluate_at(probe)
            if (not math.isfinite(value)) or value > loss_limit:
                high = probe
                break
            low = probe
            accepted_loss = value
            if probe >= search_ceiling:
                break
            probe = min(search_ceiling, probe * growth)

        if high is None:
            critical_lr = low if low > 0.0 else probe
            if not math.isfinite(accepted_loss):
                accepted_loss = base_loss
            return CriticalLREstimate(
                critical_lr=critical_lr,
                critical_sharpness=critical_sharpness_from_lr(critical_lr),
                reference_lr=reference_lr,
                base_loss=base_loss,
                accepted_loss=accepted_loss,
                evaluations=evaluations,
                bracketed=False,
            )

        for _ in range(binary_steps):
            midpoint = math.sqrt(low * high) if low > 0.0 else 0.5 * high
            value = evaluate_at(midpoint)
            if math.isfinite(value) and value <= loss_limit:
                low = midpoint
                accepted_loss = value
            else:
                high = midpoint
        if not math.isfinite(accepted_loss):
            accepted_loss = base_loss
        critical_lr = max(0.0, low)
        if critical_lr <= 0.0:
            raise FloatingPointError(
                "The critical LR is below the probe's resolution; lower current_lr and retry."
            )
        return CriticalLREstimate(
            critical_lr=critical_lr,
            critical_sharpness=critical_sharpness_from_lr(critical_lr),
            reference_lr=reference_lr,
            base_loss=base_loss,
            accepted_loss=accepted_loss,
            evaluations=evaluations,
            bracketed=True,
        )
    finally:
        if applied_lr != 0.0:
            with torch.no_grad():
                for parameter, update in direction.parameters:
                    parameter.add_(update, alpha=applied_lr)
        for module, was_training in modes:
            module.training = was_training


__all__ = [
    "critical_sharpness_from_lr",
    "estimate_critical_learning_rate",
    "estimate_gradient_noise",
]
