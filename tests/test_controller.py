import json

import pytest

from adaptive_warmup import (
    AdaptiveWarmup,
    WarmupConfig,
    large_batch_utility,
    select_critical_batch_size,
)


def test_single_critical_batch_hits_half_utility_at_same_batch() -> None:
    assert large_batch_utility(64.0, [64.0]) == pytest.approx(0.5)
    assert select_critical_batch_size([64.0], target_utility=0.5) == pytest.approx(64.0)


def test_distribution_aware_batch_selection() -> None:
    selected = select_critical_batch_size([64.0, 256.0], target_utility=0.5)
    assert selected == pytest.approx(128.0)


def test_controller_holds_batch_until_wsd_handoff_and_freezes_lr_search() -> None:
    config = WarmupConfig(
        warmup_steps=20,
        measurement_interval=5,
        batch_ema_beta=0.0,
        max_lr=0.1,
        max_batch_size=512,
    )
    controller = AdaptiveWarmup(initial_lr=1e-4, initial_batch_size=8, config=config)

    first = controller.observe(step=4, critical_lr=0.01, critical_batch_size=64.0)
    assert first.phase == "search"
    assert first.learning_rate_goal == pytest.approx(0.008)
    assert first.batch_size == 8
    assert first.batch_size_goal == 128

    frozen = controller.observe(step=9, critical_lr=0.02, critical_batch_size=256.0)
    assert frozen.learning_rate_goal == pytest.approx(0.016)
    assert frozen.batch_size == 8
    assert frozen.batch_size_goal == 256
    frozen_goal = frozen.learning_rate_goal

    settled = controller.observe(step=14, critical_lr=1.0, critical_batch_size=256.0)
    assert settled.phase == "settle"
    assert settled.learning_rate_goal == frozen_goal
    assert settled.batch_size == 8

    handoff = controller.complete_warmup(step=20)
    assert handoff.phase == "complete"
    assert handoff.batch_size == handoff.batch_size_goal
    assert handoff.batch_size <= 512


def test_batch_handoff_multiplier_is_configurable() -> None:
    controller = AdaptiveWarmup(
        initial_lr=1e-4,
        initial_batch_size=8,
        config=WarmupConfig(
            warmup_steps=10,
            measurement_interval=5,
            batch_multiplier=1.5,
            batch_round_to=8,
        ),
    )

    measured = controller.observe(step=4, critical_batch_size=64.0)
    assert measured.batch_size == 8
    assert measured.batch_size_goal == 96
    with pytest.raises(ValueError, match="before warmup"):
        controller.complete_warmup(step=9)
    assert controller.complete_warmup(step=10).batch_size == 96


def test_oom_report_creates_persistent_rounded_cap() -> None:
    controller = AdaptiveWarmup(
        initial_lr=1e-4,
        initial_batch_size=64,
        config=WarmupConfig(batch_round_to=8, oom_buffer_fraction=0.10),
    )

    recommendation = controller.report_oom(step=3, failed_batch_size=64)

    assert recommendation.batch_size == 56
    assert recommendation.batch_size_cap == 56
    controller.observe(step=4, critical_batch_size=1_000.0)
    assert controller.current_batch_size == 56
    assert controller.recommendation(step=4).batch_size_goal == 56


def test_controller_state_round_trip_is_json_serializable() -> None:
    config = WarmupConfig(warmup_steps=20, measurement_interval=5)
    source = AdaptiveWarmup(initial_lr=1e-4, initial_batch_size=8, config=config)
    source.observe(step=4, critical_lr=0.01, critical_batch_size=64.0)
    serialized = json.loads(json.dumps(source.state_dict()))

    restored = AdaptiveWarmup(initial_lr=9e-4, initial_batch_size=2, config=config)
    restored.load_state_dict(serialized)

    assert restored.state_dict() == source.state_dict()
