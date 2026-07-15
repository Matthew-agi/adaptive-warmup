import copy

import pytest
import torch

from adaptive_warmup import (
    ProbeResolutionError,
    critical_sharpness_from_lr,
    estimate_critical_learning_rate,
    estimate_gradient_noise,
)


def test_below_resolution_critical_lr_is_a_recoverable_probe_error() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.tensor(-1.0)

    with pytest.raises(ProbeResolutionError, match="below the probe's resolution"):
        estimate_critical_learning_rate(
            lambda: parameter.square(),
            optimizer=optimizer,
            current_lr=1.0,
            max_lr=1.0,
            bracket_steps=2,
            binary_steps=1,
        )

    torch.testing.assert_close(parameter, torch.tensor(1.0))


def test_gradient_noise_estimate_matches_scalar_calculation() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))

    def loss_for(target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (parameter - target).square()

    estimate = estimate_gradient_noise(
        [parameter],
        loss_for,
        torch.tensor(1.0),
        torch.tensor(3.0),
        batch_size=8,
    )

    assert estimate.noise == pytest.approx(2.0)
    assert estimate.signal == pytest.approx(4.0)
    assert estimate.noise_to_signal_ratio == pytest.approx(0.5)
    assert estimate.critical_batch_size == pytest.approx(4.0)
    assert estimate.mean_loss == pytest.approx(2.5)


def test_gradient_probe_does_not_overwrite_real_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    parameter.square().backward()
    original = parameter.grad.detach().clone()

    estimate_gradient_noise(
        [parameter],
        lambda target: (parameter - target).square(),
        torch.tensor(0.0),
        torch.tensor(1.0),
        batch_size=1,
    )

    torch.testing.assert_close(parameter.grad, original)


def test_critical_lr_finds_quadratic_directional_boundary_and_restores_state() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = 0.5 * model.weight.square().sum()
    loss.backward()
    parameter_before = model.weight.detach().clone()
    optimizer_state_before = copy.deepcopy(optimizer.state_dict())
    model.train()

    estimate = estimate_critical_learning_rate(
        lambda: 0.5 * model.weight.square().sum(),
        model=model,
        optimizer=optimizer,
        current_lr=0.01,
        max_lr=4.0,
        binary_steps=16,
    )

    assert estimate.bracketed is True
    assert estimate.critical_lr == pytest.approx(2.00005, abs=2e-4)
    assert estimate.critical_sharpness == pytest.approx(2.0 / estimate.critical_lr)
    assert model.training is True
    torch.testing.assert_close(model.weight, parameter_before, rtol=0, atol=5e-7)
    assert optimizer.state_dict() == optimizer_state_before


def test_critical_lr_respects_hard_search_cap() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    (0.5 * parameter.square()).backward()

    estimate = estimate_critical_learning_rate(
        lambda: 0.5 * parameter.square(),
        optimizer=optimizer,
        max_lr=1.0,
    )

    assert estimate.bracketed is False
    assert estimate.critical_lr == pytest.approx(1.0)


def test_critical_lr_default_search_uses_six_evaluations_near_previous_estimate() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    (0.5 * parameter.square()).backward()

    estimate = estimate_critical_learning_rate(
        lambda: 0.5 * parameter.square(),
        optimizer=optimizer,
        current_lr=1.0,
        previous_estimate=1.0,
    )

    assert estimate.evaluations == 6
    assert estimate.bracketed is True
    assert estimate.critical_lr < 2.0
    assert estimate.critical_sharpness == pytest.approx(2.0 / estimate.critical_lr)


def test_critical_lr_default_search_is_capped_at_nine_evaluations() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(0.0004)
    optimizer = torch.optim.SGD([parameter], lr=2.0)

    estimate = estimate_critical_learning_rate(
        lambda: 0.5 * parameter.square(),
        optimizer=optimizer,
        current_lr=2.0,
    )

    assert estimate.evaluations == 9
    assert estimate.bracketed is True
    assert estimate.critical_lr < 5_000.0


def test_critical_sharpness_conversion_uses_two_over_critical_lr() -> None:
    assert critical_sharpness_from_lr(0.25) == pytest.approx(8.0)


def test_critical_lr_restores_parameters_and_modes_after_closure_error() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Dropout(), torch.nn.Linear(2, 1))
    model.train()
    model[1].eval()
    modes_before = [module.training for module in model.modules()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 1)).square().sum().backward()
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    calls = 0

    def flaky_loss() -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("probe failed")
        return model(torch.ones(1, 1)).square().sum()

    with pytest.raises(RuntimeError, match="probe failed"):
        estimate_critical_learning_rate(
            flaky_loss,
            model=model,
            optimizer=optimizer,
        )

    for parameter, original in zip(model.parameters(), parameters_before):
        torch.testing.assert_close(parameter, original, rtol=0, atol=2e-7)
    assert [module.training for module in model.modules()] == modes_before
