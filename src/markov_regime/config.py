from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Interval = Literal["1hour", "4hour", "1day"]
ConsensusGateMode = Literal["hard", "entry_only"]
HistoricalProvider = Literal["auto", "fmp", "coinbase", "yahoo"]
AssetClass = Literal["crypto", "equity"]


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


CRYPTO_ALIASES: set[str] = {
    "BTC",
    "BTC-USD",
    "BTCUSD",
    "ETH",
    "ETH-USD",
    "ETHUSD",
    "SOL",
    "SOL-USD",
    "SOLUSD",
    "DOGE",
    "DOGE-USD",
    "DOGEUSD",
    "ADA",
    "ADA-USD",
    "ADAUSD",
    "XRP",
    "XRP-USD",
    "XRPUSD",
    "BNB",
    "BNB-USD",
    "BNBUSD",
}


@dataclass(frozen=True)
class AssetDefaults:
    asset_class: AssetClass
    interval: Interval
    feature_pack: str
    limit: int
    robustness_symbols: tuple[str, ...]
    cost_bps: float
    spread_bps: float
    slippage_bps: float
    impact_bps: float
    require_daily_confirmation: bool
    provider: HistoricalProvider = "auto"


def infer_asset_class(symbol: str) -> AssetClass:
    cleaned = str(symbol).strip().upper()
    if cleaned in CRYPTO_ALIASES:
        return "crypto"
    if cleaned.endswith("-USD") and cleaned[:-4].isalpha():
        return "crypto"
    if cleaned.endswith("USD") and cleaned[:-3].isalpha() and len(cleaned[:-3]) >= 2:
        return "crypto"
    return "equity"


def asset_class_label(asset_class: AssetClass) -> str:
    return "crypto (24/7)" if asset_class == "crypto" else "equity / ETF (market hours)"


def default_robustness_basket(symbol: str, asset_class: AssetClass | None = None) -> tuple[str, ...]:
    resolved_asset_class = asset_class or infer_asset_class(symbol)
    cleaned = str(symbol).strip().upper()
    if resolved_asset_class == "crypto":
        candidates = [cleaned, "BTCUSD", "ETHUSD", "SOLUSD"]
    else:
        candidates = [cleaned, "SPY", "QQQ", "IWM"]
    deduped = list(dict.fromkeys(item for item in candidates if item))
    return tuple(deduped[:3] if len(deduped) >= 3 else deduped)


def default_asset_settings(symbol: str) -> AssetDefaults:
    asset_class = infer_asset_class(symbol)
    if asset_class == "crypto":
        return AssetDefaults(
            asset_class="crypto",
            interval="4hour",
            feature_pack="mean_reversion",
            limit=5000,
            robustness_symbols=default_robustness_basket(symbol, asset_class),
            cost_bps=10.0,
            spread_bps=4.0,
            slippage_bps=3.0,
            impact_bps=2.0,
            require_daily_confirmation=False,
            provider="auto",
        )
    return AssetDefaults(
        asset_class="equity",
        interval="1day",
        feature_pack="trend",
        limit=3000,
        robustness_symbols=default_robustness_basket(symbol, "equity"),
        cost_bps=0.0,
        spread_bps=1.0,
        slippage_bps=1.0,
        impact_bps=0.5,
        require_daily_confirmation=False,
        provider="auto",
    )


def bars_per_year(interval: Interval, asset_class: AssetClass = "crypto") -> int:
    if asset_class == "equity":
        if interval == "1day":
            return 252
        if interval == "4hour":
            return int(round(252 * (6.5 / 4.0)))
        return int(round(252 * 6.5))
    if interval == "1day":
        return 365
    if interval == "4hour":
        return 365 * 6
    return 365 * 24


def default_walk_forward_config(interval: Interval, asset_class: AssetClass = "crypto") -> WalkForwardConfig:
    if asset_class == "equity":
        if interval == "4hour":
            return WalkForwardConfig(
                train_bars=int(round(252 * (6.5 / 4.0))),
                purge_bars=2,
                validate_bars=int(round(63 * (6.5 / 4.0))),
                embargo_bars=2,
                test_bars=int(round(63 * (6.5 / 4.0))),
                refit_stride_bars=int(round(63 * (6.5 / 4.0))),
            )
        if interval == "1day":
            return WalkForwardConfig(
                train_bars=252,
                purge_bars=2,
                validate_bars=63,
                embargo_bars=2,
                test_bars=63,
                refit_stride_bars=63,
            )
        return WalkForwardConfig(
            train_bars=int(round(252 * 6.5)),
            purge_bars=6,
            validate_bars=int(round(63 * 6.5)),
            embargo_bars=6,
            test_bars=int(round(63 * 6.5)),
            refit_stride_bars=int(round(63 * 6.5)),
        )
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
