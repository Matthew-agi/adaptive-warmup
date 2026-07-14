"""Preview the next PyTorch optimizer direction without changing its state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import Adam, AdamW, Optimizer, SGD


@dataclass(frozen=True)
class OptimizerDirection:
    """Parameter directions normalized to a reference learning rate.

    Applying ``parameter -= candidate_lr * direction`` preserves the relative
    learning-rate ratios of all optimizer parameter groups.
    """

    reference_lr: float
    parameters: tuple[tuple[torch.nn.Parameter, torch.Tensor], ...]
    optimizer_name: str


def _number(value: object) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def _reference_lr(optimizer: Optimizer, requested: float | None) -> float:
    if requested is not None:
        value = float(requested)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("reference_lr must be finite and positive.")
        return value
    for group in optimizer.param_groups:
        value = _number(group.get("lr", 0.0))
        if math.isfinite(value) and value > 0.0:
            return value
    raise ValueError("The optimizer has no positive learning rate.")


def _next_step(state: dict[object, object]) -> int:
    raw = state.get("step", 0)
    if torch.is_tensor(raw):
        return int(raw.detach().item()) + 1
    return int(raw) + 1


def _adam_direction(
    parameter: torch.nn.Parameter,
    group: dict[str, object],
    state: dict[object, object],
    *,
    decoupled_weight_decay: bool,
) -> torch.Tensor:
    grad = parameter.grad
    if grad is None:
        raise ValueError("Cannot construct a direction for a parameter without a gradient.")
    if grad.is_sparse:
        raise TypeError("Sparse gradients are not supported by the Adam direction preview.")
    if parameter.is_complex():
        raise TypeError("Complex parameters are not supported by the direction preview.")

    direction_grad = grad.detach()
    if bool(group.get("maximize", False)):
        direction_grad = -direction_grad
    weight_decay = float(group.get("weight_decay", 0.0))
    if weight_decay and not decoupled_weight_decay:
        direction_grad = direction_grad.add(parameter.detach(), alpha=weight_decay)

    beta1, beta2 = group["betas"]  # type: ignore[misc]
    beta1 = float(beta1)
    beta2 = float(beta2)
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None:
        exp_avg_next = direction_grad.clone().mul_(1.0 - beta1)
    else:
        exp_avg_next = exp_avg.detach().clone().mul_(beta1).add_(
            direction_grad,
            alpha=1.0 - beta1,
        )
    if exp_avg_sq is None:
        exp_avg_sq_next = direction_grad.square().mul_(1.0 - beta2)
    else:
        exp_avg_sq_next = exp_avg_sq.detach().clone().mul_(beta2).addcmul_(
            direction_grad,
            direction_grad,
            value=1.0 - beta2,
        )

    if bool(group.get("amsgrad", False)):
        previous_max = state.get("max_exp_avg_sq")
        if previous_max is not None:
            exp_avg_sq_next = torch.maximum(previous_max.detach(), exp_avg_sq_next)

    step = _next_step(state)
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    eps = float(group.get("eps", 1e-8))
    denominator = exp_avg_sq_next.sqrt().div_(math.sqrt(correction2)).add_(eps)
    direction = exp_avg_next.div_(correction1).div_(denominator)
    if weight_decay and decoupled_weight_decay:
        direction = direction.add(parameter.detach(), alpha=weight_decay)
    return direction


def _sgd_direction(
    parameter: torch.nn.Parameter,
    group: dict[str, object],
    state: dict[object, object],
) -> torch.Tensor:
    grad = parameter.grad
    if grad is None:
        raise ValueError("Cannot construct a direction for a parameter without a gradient.")
    if grad.is_sparse:
        raise TypeError("Sparse gradients are not supported by the SGD direction preview.")
    direction = grad.detach()
    if bool(group.get("maximize", False)):
        direction = -direction
    weight_decay = float(group.get("weight_decay", 0.0))
    if weight_decay:
        direction = direction.add(parameter.detach(), alpha=weight_decay)

    momentum = float(group.get("momentum", 0.0))
    if momentum:
        previous = state.get("momentum_buffer")
        if previous is None:
            buffer = direction.clone()
        else:
            dampening = float(group.get("dampening", 0.0))
            buffer = previous.detach().clone().mul_(momentum).add_(
                direction,
                alpha=1.0 - dampening,
            )
        if bool(group.get("nesterov", False)):
            direction = direction.add(buffer, alpha=momentum)
        else:
            direction = buffer
    return direction


def optimizer_step_direction(
    optimizer: Optimizer,
    *,
    reference_lr: float | None = None,
) -> OptimizerDirection:
    """Return the upcoming SGD, Adam, or AdamW update direction.

    Gradients must already be populated. Optimizer parameters and state are
    only read. Parameter-group learning-rate ratios are folded into each
    direction relative to ``reference_lr``.
    """

    ref = _reference_lr(optimizer, reference_lr)
    if isinstance(optimizer, AdamW):
        family = "adamw"
    elif isinstance(optimizer, Adam):
        family = "adam"
    elif isinstance(optimizer, SGD):
        family = "sgd"
    else:
        raise TypeError(
            "Built-in direction preview supports torch.optim.SGD, Adam, and AdamW. "
            "Use estimate_critical_learning_rate(..., direction=...) for a custom optimizer."
        )

    output: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for group in optimizer.param_groups:
        group_lr = _number(group.get("lr", 0.0))
        group_scale = group_lr / ref
        if group_scale == 0.0:
            continue
        for parameter in group["params"]:  # type: ignore[index]
            if not isinstance(parameter, torch.nn.Parameter) or parameter.grad is None:
                continue
            state = optimizer.state.get(parameter, {})
            if family in {"adam", "adamw"}:
                decoupled = family == "adamw" or bool(group.get("decoupled_weight_decay", False))
                raw_direction = _adam_direction(
                    parameter,
                    group,
                    state,
                    decoupled_weight_decay=decoupled,
                )
            else:
                raw_direction = _sgd_direction(parameter, group, state)
            output.append((parameter, raw_direction.mul(group_scale)))

    if not output:
        raise ValueError("No dense gradients were available to preview an optimizer step.")
    return OptimizerDirection(
        reference_lr=ref,
        parameters=tuple(output),
        optimizer_name=type(optimizer).__name__,
    )


def set_reference_lr(
    optimizer: Optimizer,
    new_reference_lr: float,
    *,
    current_reference_lr: float | None = None,
) -> float:
    """Scale every optimizer group LR while preserving group ratios."""

    new_value = float(new_reference_lr)
    if not math.isfinite(new_value) or new_value <= 0.0:
        raise ValueError("new_reference_lr must be finite and positive.")
    current = _reference_lr(optimizer, current_reference_lr)
    scale = new_value / current
    for group in optimizer.param_groups:
        old_lr = group.get("lr", 0.0)
        if torch.is_tensor(old_lr):
            old_lr.mul_(scale)
        else:
            group["lr"] = float(old_lr) * scale
    return new_value


__all__ = ["OptimizerDirection", "optimizer_step_direction", "set_reference_lr"]
