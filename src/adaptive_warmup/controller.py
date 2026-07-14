"""Checkpointable policy for turning noisy probes into warmup settings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from .types import CriticalLREstimate, GradientNoiseEstimate, WarmupRecommendation


@dataclass(frozen=True)
class WarmupConfig:
    """Policy settings with conservative defaults from the original trainer."""

    warmup_steps: int = 1_000
    measurement_interval: int = 5
    lr_freeze_fraction: float = 0.5
    lr_safety_factor: float = 0.8
    lr_ema_beta: float = 0.9
    batch_ema_beta: float = 0.995
    lr_outlier_factor: float = 4.0
    batch_outlier_factor: float = 4.0
    batch_window: int = 9
    batch_target_utility: float = 0.5
    batch_multiplier: float = 1.0
    batch_round_to: int = 8
    max_lr: float | None = None
    max_batch_size: int | None = None
    oom_buffer_fraction: float = 0.10
    early_lr_fraction: float = 0.10
    early_lr_max_increase: float = 0.10

    def __post_init__(self) -> None:
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        if self.measurement_interval <= 0:
            raise ValueError("measurement_interval must be positive.")
        if not 0.0 < self.lr_freeze_fraction <= 1.0:
            raise ValueError("lr_freeze_fraction must be in (0, 1].")
        if self.lr_safety_factor <= 0.0:
            raise ValueError("lr_safety_factor must be positive.")
        if not 0.0 <= self.lr_ema_beta < 1.0:
            raise ValueError("lr_ema_beta must be in [0, 1).")
        if not 0.0 <= self.batch_ema_beta < 1.0:
            raise ValueError("batch_ema_beta must be in [0, 1).")
        if self.lr_outlier_factor < 1.0 or self.batch_outlier_factor < 1.0:
            raise ValueError("outlier factors must be at least 1.")
        if self.batch_window <= 0:
            raise ValueError("batch_window must be positive.")
        if not 0.0 < self.batch_target_utility < 1.0:
            raise ValueError("batch_target_utility must be in (0, 1).")
        if self.batch_multiplier <= 0.0 or self.batch_round_to <= 0:
            raise ValueError("batch_multiplier and batch_round_to must be positive.")
        if self.max_lr is not None and self.max_lr <= 0.0:
            raise ValueError("max_lr must be positive when supplied.")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive when supplied.")
        if not 0.0 <= self.oom_buffer_fraction < 1.0:
            raise ValueError("oom_buffer_fraction must be in [0, 1).")
        if not 0.0 <= self.early_lr_fraction <= 1.0:
            raise ValueError("early_lr_fraction must be in [0, 1].")
        if self.early_lr_max_increase < 0.0:
            raise ValueError("early_lr_max_increase must be non-negative.")


def _positive(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(value) and value > 0.0]


def large_batch_utility(batch_size: float, critical_batch_sizes: list[float]) -> float:
    """Mean fraction of asymptotic large-batch utility for a candidate batch."""

    valid = _positive(critical_batch_sizes)
    if not valid or not math.isfinite(batch_size) or batch_size <= 0.0:
        return float("nan")
    batch = float(batch_size)
    return sum(batch / (batch + critical) for critical in valid) / len(valid)


def select_critical_batch_size(
    critical_batch_sizes: list[float],
    *,
    target_utility: float = 0.5,
    max_batch_size: float | None = None,
) -> float | None:
    """Select the smallest batch meeting a target mean large-batch utility."""

    valid = _positive(critical_batch_sizes)
    if not valid:
        return None
    if not 0.0 < target_utility < 1.0:
        raise ValueError("target_utility must be in (0, 1).")
    lower = 1.0
    upper = max(valid)
    upper = max(upper, (target_utility / (1.0 - target_utility)) * max(valid))
    if max_batch_size is not None:
        if max_batch_size <= 0.0:
            raise ValueError("max_batch_size must be positive when supplied.")
        upper = min(upper, float(max_batch_size))
    upper = max(lower, upper)
    if large_batch_utility(lower, valid) >= target_utility:
        return lower
    if large_batch_utility(upper, valid) < target_utility:
        return upper
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        if large_batch_utility(midpoint, valid) >= target_utility:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _round_batch_down(value: float, multiple: int) -> int:
    if not math.isfinite(value) or value <= 0.0:
        return 1
    rounded = int(math.floor(value / multiple)) * multiple
    return rounded if rounded > 0 else max(1, int(math.floor(value)))


def _clamp_sample(value: float, center: float | None, factor: float) -> float:
    if center is None or not math.isfinite(center) or center <= 0.0 or factor <= 1.0:
        return value
    return min(max(value, center / factor), center * factor)


def _theil_sen_log_slope(points: list[tuple[int, float]]) -> float | None:
    slopes: list[float] = []
    clean = [(int(step), math.log(value)) for step, value in points if value > 0.0]
    for index, (step_a, value_a) in enumerate(clean[:-1]):
        for step_b, value_b in clean[index + 1 :]:
            if step_b > step_a:
                slopes.append((value_b - value_a) / (step_b - step_a))
    return float(median(slopes)) if slopes else None


class AdaptiveWarmup:
    """Convert periodic measurements into LR and batch-size recommendations."""

    def __init__(
        self,
        *,
        initial_lr: float,
        initial_batch_size: int,
        config: WarmupConfig | None = None,
    ) -> None:
        if not math.isfinite(initial_lr) or initial_lr <= 0.0:
            raise ValueError("initial_lr must be finite and positive.")
        if initial_batch_size <= 0:
            raise ValueError("initial_batch_size must be positive.")
        self.config = config or WarmupConfig()
        self.current_lr = float(initial_lr)
        self.current_batch_size = int(initial_batch_size)
        self.lr_goal: float | None = None
        self.lr_ema: float | None = None
        self.first_critical_lr: float | None = None
        self.last_critical_lr: float | None = None
        self.recent_lr_points: list[tuple[int, float]] = []
        self.batch_ema: float | None = None
        self.selected_critical_batch: float | None = None
        self.recent_critical_batches: list[float] = []
        self.batch_size_cap: int | None = self.config.max_batch_size
        self.last_safe_batch_size = int(initial_batch_size)
        self.last_step = -1

    @property
    def lr_freeze_step(self) -> int:
        return max(1, int(self.config.warmup_steps * self.config.lr_freeze_fraction))

    def should_measure(self, step: int) -> bool:
        """Return whether a zero-based training step is a measurement step."""

        return 0 <= step < self.config.warmup_steps and (
            (step + 1) % self.config.measurement_interval == 0
        )

    def _phase(self, step: int) -> str:
        if step >= self.config.warmup_steps:
            return "complete"
        if step < self.lr_freeze_step:
            return "search"
        return "settle"

    def _recommendation(
        self,
        step: int,
        *,
        lr_sample: float | None = None,
        batch_sample: float | None = None,
    ) -> WarmupRecommendation:
        return WarmupRecommendation(
            step=int(step),
            phase=self._phase(step),
            learning_rate=float(self.current_lr),
            batch_size=int(self.current_batch_size),
            learning_rate_goal=self.lr_goal,
            selected_critical_batch_size=self.selected_critical_batch,
            critical_lr_sample=lr_sample,
            critical_batch_sample=batch_sample,
            batch_size_cap=self.batch_size_cap,
        )

    def recommendation(self, step: int) -> WarmupRecommendation:
        """Return current settings without consuming a measurement."""

        return self._recommendation(step)

    def _update_lr_goal(self, step: int, sample: float) -> float:
        filtered = _clamp_sample(sample, self.lr_ema, self.config.lr_outlier_factor)
        if self.first_critical_lr is None:
            self.first_critical_lr = filtered
        self.last_critical_lr = filtered
        if self.lr_ema is None:
            self.lr_ema = filtered
        else:
            beta = self.config.lr_ema_beta
            self.lr_ema = beta * self.lr_ema + (1.0 - beta) * filtered
        self.recent_lr_points.append((int(step), filtered))
        self.recent_lr_points = self.recent_lr_points[-7:]

        slope = _theil_sen_log_slope(self.recent_lr_points)
        forecast = float(self.lr_ema)
        if slope is not None:
            remaining = max(0, self.config.warmup_steps - (step + 1))
            forecast = math.exp(
                min(700.0, max(math.log(1e-12), math.log(self.lr_ema) + slope * remaining))
            )
        if self.first_critical_lr is not None:
            forecast = min(forecast, self.first_critical_lr)
        current_safe = self.config.lr_safety_factor * filtered
        forecast_safe = self.config.lr_safety_factor * forecast
        final_search_update = step + self.config.measurement_interval >= self.lr_freeze_step
        goal = current_safe if final_search_update else min(forecast_safe, current_safe)
        if self.config.max_lr is not None:
            goal = min(goal, self.config.max_lr)
        self.lr_goal = max(goal, 1e-12)
        return current_safe

    def _ramp_lr(self, step: int, current_safe: float | None) -> None:
        if self.lr_goal is None:
            return
        updates_left = max(
            1,
            int(
                math.ceil(
                    (max(1, self.config.warmup_steps) - step)
                    / self.config.measurement_interval
                )
            ),
        )
        proposed = self.current_lr + (self.lr_goal - self.current_lr) / updates_left
        early_steps = max(1, math.ceil(self.config.warmup_steps * self.config.early_lr_fraction))
        if step + 1 <= early_steps and proposed > self.current_lr:
            proposed = min(
                proposed,
                self.current_lr * (1.0 + self.config.early_lr_max_increase),
            )
        if current_safe is not None:
            proposed = min(proposed, current_safe)
        self.current_lr = max(float(proposed), 1e-12)

    def _update_batch(self, sample: float) -> None:
        filtered = _clamp_sample(sample, self.batch_ema, self.config.batch_outlier_factor)
        if self.batch_ema is None:
            self.batch_ema = filtered
        else:
            beta = self.config.batch_ema_beta
            self.batch_ema = beta * self.batch_ema + (1.0 - beta) * filtered
        self.recent_critical_batches.append(filtered)
        self.recent_critical_batches = self.recent_critical_batches[-self.config.batch_window :]
        self.selected_critical_batch = select_critical_batch_size(
            self.recent_critical_batches,
            target_utility=self.config.batch_target_utility,
        )
        if self.selected_critical_batch is None:
            return
        goal = _round_batch_down(
            self.selected_critical_batch * self.config.batch_multiplier,
            self.config.batch_round_to,
        )
        if self.batch_size_cap is not None:
            goal = min(goal, self.batch_size_cap)
        if self.config.max_batch_size is not None:
            goal = min(goal, self.config.max_batch_size)
        if goal > self.current_batch_size:
            self.last_safe_batch_size = self.current_batch_size
            self.current_batch_size = goal

    def observe(
        self,
        step: int,
        *,
        critical_lr: CriticalLREstimate | float | None = None,
        critical_batch_size: GradientNoiseEstimate | float | None = None,
    ) -> WarmupRecommendation:
        """Consume probe results and return settings for the upcoming update."""

        if step < 0:
            raise ValueError("step must be non-negative.")
        if step < self.last_step:
            raise ValueError("Measurements must be observed in non-decreasing step order.")
        self.last_step = int(step)
        raw_lr = (
            float(critical_lr.critical_lr)
            if isinstance(critical_lr, CriticalLREstimate)
            else (float(critical_lr) if critical_lr is not None else None)
        )
        raw_batch = (
            float(critical_batch_size.critical_batch_size)
            if isinstance(critical_batch_size, GradientNoiseEstimate)
            else (float(critical_batch_size) if critical_batch_size is not None else None)
        )
        valid_lr = raw_lr if raw_lr is not None and math.isfinite(raw_lr) and raw_lr > 0.0 else None
        valid_batch = (
            raw_batch
            if raw_batch is not None and math.isfinite(raw_batch) and raw_batch > 0.0
            else None
        )
        if step >= self.config.warmup_steps:
            return self._recommendation(step, lr_sample=valid_lr, batch_sample=valid_batch)

        current_safe: float | None = None
        if step < self.lr_freeze_step and valid_lr is not None:
            current_safe = self._update_lr_goal(step, valid_lr)
        self._ramp_lr(step, current_safe)
        if valid_batch is not None:
            self._update_batch(valid_batch)
        self.last_safe_batch_size = self.current_batch_size
        return self._recommendation(step, lr_sample=valid_lr, batch_sample=valid_batch)

    def report_oom(self, step: int, *, failed_batch_size: int) -> WarmupRecommendation:
        """Back off a failed batch and remember the resulting hardware cap."""

        if failed_batch_size <= 0:
            raise ValueError("failed_batch_size must be positive.")
        target = max(1, int(math.floor(failed_batch_size * (1.0 - self.config.oom_buffer_fraction))))
        target = _round_batch_down(float(target), self.config.batch_round_to)
        self.batch_size_cap = (
            target if self.batch_size_cap is None else min(self.batch_size_cap, target)
        )
        self.current_batch_size = min(self.current_batch_size, self.batch_size_cap)
        self.last_safe_batch_size = self.current_batch_size
        return self._recommendation(step)

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-serializable controller state for training checkpoints."""

        return {
            "version": 1,
            "config": asdict(self.config),
            "current_lr": self.current_lr,
            "current_batch_size": self.current_batch_size,
            "lr_goal": self.lr_goal,
            "lr_ema": self.lr_ema,
            "first_critical_lr": self.first_critical_lr,
            "last_critical_lr": self.last_critical_lr,
            "recent_lr_points": [list(point) for point in self.recent_lr_points],
            "batch_ema": self.batch_ema,
            "selected_critical_batch": self.selected_critical_batch,
            "recent_critical_batches": list(self.recent_critical_batches),
            "batch_size_cap": self.batch_size_cap,
            "last_safe_batch_size": self.last_safe_batch_size,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, Any], *, strict_config: bool = True) -> None:
        """Restore a state produced by :meth:`state_dict`."""

        if int(state.get("version", 0)) != 1:
            raise ValueError("Unsupported adaptive-warmup state version.")
        if strict_config and state.get("config") != asdict(self.config):
            raise ValueError("Checkpoint WarmupConfig does not match this controller.")
        self.current_lr = float(state["current_lr"])
        self.current_batch_size = int(state["current_batch_size"])
        self.lr_goal = None if state.get("lr_goal") is None else float(state["lr_goal"])
        self.lr_ema = None if state.get("lr_ema") is None else float(state["lr_ema"])
        self.first_critical_lr = (
            None if state.get("first_critical_lr") is None else float(state["first_critical_lr"])
        )
        self.last_critical_lr = (
            None if state.get("last_critical_lr") is None else float(state["last_critical_lr"])
        )
        self.recent_lr_points = [
            (int(step), float(value)) for step, value in state.get("recent_lr_points", [])
        ]
        self.batch_ema = None if state.get("batch_ema") is None else float(state["batch_ema"])
        self.selected_critical_batch = (
            None
            if state.get("selected_critical_batch") is None
            else float(state["selected_critical_batch"])
        )
        self.recent_critical_batches = [
            float(value) for value in state.get("recent_critical_batches", [])
        ]
        self.batch_size_cap = (
            None if state.get("batch_size_cap") is None else int(state["batch_size_cap"])
        )
        self.last_safe_batch_size = int(state["last_safe_batch_size"])
        self.last_step = int(state["last_step"])


__all__ = [
    "AdaptiveWarmup",
    "WarmupConfig",
    "large_batch_utility",
    "select_critical_batch_size",
]
