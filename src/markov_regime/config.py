from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Interval = Literal["1hour", "4hour", "1day"]
ConsensusGateMode = Literal["hard", "entry_only"]
HistoricalProvider = Literal["auto", "fmp", "coinbase", "yahoo"]


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BTCUSD"
    interval: Interval = "4hour"
    limit: int = 2500
    start: str | None = None
    end: str | None = None
    provider: HistoricalProvider = "auto"


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
    purge_bars: int = 6
    validate_bars: int = 180
    embargo_bars: int = 6
    test_bars: int = 180
    refit_stride_bars: int = 180


@dataclass(frozen=True)
class StrategyConfig:
    posterior_threshold: float = 0.65
    min_hold_bars: int = 6
    cooldown_bars: int = 3
    required_confirmations: int = 2
    confidence_gap: float = 0.05
    require_daily_confirmation: bool = False
    require_consensus_confirmation: bool = False
    consensus_min_share: float = 0.67
    consensus_gate_mode: ConsensusGateMode = "entry_only"
    min_validation_edge: float = 0.0
    min_validation_samples: int = 20
    signal_horizon: int = 6
    scoring_horizons: tuple[int, ...] = (6, 12, 24)
    validation_shrinkage: float = 30.0
    min_consistent_horizons: int = 2
    allow_short: bool = False
    cost_bps: float = 10.0
    spread_bps: float = 4.0
    slippage_bps: float = 3.0
    impact_bps: float = 2.0
    range_impact_weight: float = 0.15
    volume_reference: float = 1_000_000.0
    cost_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0)


@dataclass(frozen=True)
class SweepConfig:
    posterior_thresholds: tuple[float, ...] = (0.55, 0.6, 0.65, 0.7, 0.75)
    min_hold_bars: tuple[int, ...] = (1, 3, 6, 12)
    cooldown_bars: tuple[int, ...] = (0, 2, 4, 8)
    required_confirmations: tuple[int, ...] = (1, 2, 3, 4)


def bars_per_year(interval: Interval) -> int:
    # This project is now crypto-first, so annualization assumes 24/7 bars.
    if interval == "1day":
        return 365
    if interval == "4hour":
        return 365 * 6
    return 365 * 24


def default_walk_forward_config(interval: Interval) -> WalkForwardConfig:
    if interval == "4hour":
        return WalkForwardConfig(
            train_bars=365 * 6,
            purge_bars=6,
            validate_bars=90 * 6,
            embargo_bars=6,
            test_bars=90 * 6,
            refit_stride_bars=90 * 6,
        )
    if interval == "1day":
        return WalkForwardConfig(
            train_bars=365,
            purge_bars=2,
            validate_bars=90,
            embargo_bars=2,
            test_bars=90,
            refit_stride_bars=90,
        )
    return WalkForwardConfig(
        train_bars=720,
        purge_bars=6,
        validate_bars=120,
        embargo_bars=6,
        test_bars=120,
        refit_stride_bars=120,
    )
