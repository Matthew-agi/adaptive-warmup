import torch

from adaptive_warmup import optimizer_step_direction, set_reference_lr


def _previewed_parameter(optimizer: torch.optim.Optimizer) -> torch.Tensor:
    preview = optimizer_step_direction(optimizer)
    parameter, direction = preview.parameters[0]
    return parameter.detach() - preview.reference_lr * direction


def test_adamw_preview_matches_real_steps() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.1)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        expected = _previewed_parameter(optimizer)
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


def test_adam_preview_matches_real_steps() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.01, weight_decay=0.1, amsgrad=True)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        expected = _previewed_parameter(optimizer)
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


def test_sgd_momentum_preview_matches_real_steps() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.SGD(
        [parameter],
        lr=0.05,
        momentum=0.9,
        nesterov=True,
        weight_decay=0.01,
    )

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()
        expected = _previewed_parameter(optimizer)
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)


def test_reference_lr_scaling_preserves_param_group_ratios() -> None:
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "lr": 0.01},
            {"params": [second], "lr": 0.001},
        ]
    )

    set_reference_lr(optimizer, 0.02)

    assert optimizer.param_groups[0]["lr"] == 0.02
    assert optimizer.param_groups[1]["lr"] == 0.002
