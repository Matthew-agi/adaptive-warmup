#!/usr/bin/env python3
"""End-to-end adaptive warmup on a small synthetic regression problem."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adaptive_warmup import (
    AdaptiveWarmup,
    WarmupConfig,
    estimate_critical_learning_rate,
    estimate_gradient_noise,
    set_reference_lr,
)


def main() -> None:
    torch.manual_seed(7)
    features = torch.randn(4_096, 16)
    true_weights = torch.randn(16, 1)
    targets = features @ true_weights + 0.1 * torch.randn(4_096, 1)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    warmup = AdaptiveWarmup(
        initial_lr=1e-5,
        initial_batch_size=4,
        config=WarmupConfig(
            warmup_steps=30,
            measurement_interval=5,
            max_lr=0.05,
            max_batch_size=256,
        ),
    )

    def draw(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, features.shape[0], (batch_size,))
        return features[indices], targets[indices]

    def loss_for(batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, expected = batch
        return F.mse_loss(model(inputs), expected)

    for step in range(60):
        train_batch = draw(warmup.current_batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(train_batch)
        loss.backward()

        if warmup.should_measure(step):
            batch_estimate = estimate_gradient_noise(
                model.parameters(),
                loss_for,
                draw(warmup.current_batch_size),
                draw(warmup.current_batch_size),
                batch_size=warmup.current_batch_size,
            )
            held_out = draw(max(64, warmup.current_batch_size))
            lr_estimate = estimate_critical_learning_rate(
                lambda: loss_for(held_out),
                model=model,
                optimizer=optimizer,
                current_lr=float(optimizer.param_groups[0]["lr"]),
                previous_estimate=warmup.last_critical_lr,
                max_lr=warmup.config.max_lr,
                expansion_steps=10,
                binary_steps=8,
            )
            recommendation = warmup.observe(
                step,
                critical_lr=lr_estimate,
                critical_batch_size=batch_estimate,
            )
            set_reference_lr(optimizer, recommendation.learning_rate)
            print(
                f"step={step + 1:02d} phase={recommendation.phase:8s} "
                f"lr={recommendation.learning_rate:.3e} "
                f"batch={recommendation.batch_size:3d} "
                f"critical_lr={lr_estimate.critical_lr:.3e} "
                f"critical_batch={batch_estimate.critical_batch_size:.1f}"
            )

        optimizer.step()

    final_loss = float(loss_for((features, targets)).detach())
    print(f"final_loss={final_loss:.6f}")


if __name__ == "__main__":
    main()
