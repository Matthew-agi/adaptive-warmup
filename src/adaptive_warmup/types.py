"""Public result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GradientNoiseEstimate:
    """Two-minibatch estimate of gradient noise and critical batch size."""

    critical_batch_size: float
    noise: float
    signal: float
    noise_to_signal_ratio: float
    batch_size: int
    sampled_parameters: int
    mean_loss: float


@dataclass(frozen=True)
class CriticalLREstimate:
    """Largest accepted reference LR along the next optimizer direction."""

    critical_lr: float
    critical_sharpness: float
    reference_lr: float
    base_loss: float
    accepted_loss: float
    evaluations: int
    bracketed: bool


@dataclass(frozen=True)
class WarmupRecommendation:
    """Controller output to apply to the next optimizer update."""

    step: int
    phase: str
    learning_rate: float
    batch_size: int
    learning_rate_goal: float | None
    selected_critical_batch_size: float | None
    critical_lr_sample: float | None
    critical_batch_sample: float | None
    batch_size_cap: int | None
    batch_size_goal: int | None = None
