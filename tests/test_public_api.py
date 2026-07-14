def test_public_api_imports() -> None:
    from adaptive_warmup import (  # noqa: F401
        AdaptiveWarmup,
        CriticalLREstimate,
        GradientNoiseEstimate,
        OptimizerDirection,
        WarmupConfig,
        WarmupRecommendation,
        estimate_critical_learning_rate,
        estimate_gradient_noise,
        optimizer_step_direction,
        set_reference_lr,
    )
