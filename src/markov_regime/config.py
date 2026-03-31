from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Interval = Literal["1hour", "1day"]


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "SPY"
    interval: Interval = "1hour"
    limit: int = 2500
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    n_states: int = 6
    covariance_type: str = "full"
    n_iter: int = 300
    random_state: int = 7
    min_covar: float = 1e-4


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 750
    validate_bars: int = 180
    test_bars: int = 180
    refit_stride_bars: int = 180


@dataclass(frozen=True)
class StrategyConfig:
    posterior_threshold: float = 0.65
    min_hold_bars: int = 6
    cooldown_bars: int = 3
    required_confirmations: int = 2
    confidence_gap: float = 0.05
    min_validation_edge: float = 0.0
    min_validation_samples: int = 20
    signal_horizon: int = 6
    allow_short: bool = False
    cost_bps: float = 2.0
    cost_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class SweepConfig:
    posterior_thresholds: tuple[float, ...] = (0.55, 0.6, 0.65, 0.7, 0.75)
    min_hold_bars: tuple[int, ...] = (1, 3, 6, 12)
    cooldown_bars: tuple[int, ...] = (0, 2, 4, 8)
    required_confirmations: tuple[int, ...] = (1, 2, 3, 4)


def bars_per_year(interval: Interval) -> int:
    if interval == "1day":
        return 252
    return 252 * 6

