"""Adaptive batch-size and learning-rate warmup for PyTorch."""

from .controller import (
    AdaptiveWarmup,
    WarmupConfig,
    large_batch_utility,
    select_critical_batch_size,
)
from .directions import OptimizerDirection, optimizer_step_direction, set_reference_lr
from .estimates import (
    critical_sharpness_from_lr,
    estimate_critical_learning_rate,
    estimate_gradient_noise,
)
from .types import CriticalLREstimate, GradientNoiseEstimate, WarmupRecommendation

__all__ = [
    "AdaptiveWarmup",
    "CriticalLREstimate",
    "GradientNoiseEstimate",
    "OptimizerDirection",
    "WarmupConfig",
    "WarmupRecommendation",
    "critical_sharpness_from_lr",
    "estimate_critical_learning_rate",
    "estimate_gradient_noise",
    "large_batch_utility",
    "optimizer_step_direction",
    "select_critical_batch_size",
    "set_reference_lr",
]

__version__ = "0.3.0"
