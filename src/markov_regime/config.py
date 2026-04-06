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

EQUITY_PEER_BASKETS: dict[str, tuple[str, ...]] = {
    "SPY": ("SPY", "QQQ", "IWM"),
    "QQQ": ("QQQ", "SPY", "IWM"),
    "IWM": ("IWM", "SPY", "QQQ"),
    "DIA": ("DIA", "SPY", "QQQ"),
    "AAPL": ("AAPL", "MSFT", "NVDA"),
    "MSFT": ("MSFT", "AAPL", "GOOGL"),
    "NVDA": ("NVDA", "AMD", "AVGO"),
    "AMD": ("AMD", "NVDA", "AVGO"),
    "AVGO": ("AVGO", "NVDA", "AMD"),
    "GOOGL": ("GOOGL", "MSFT", "META"),
    "GOOG": ("GOOG", "GOOGL", "MSFT"),
    "META": ("META", "GOOGL", "AMZN"),
    "AMZN": ("AMZN", "META", "WMT"),
    "TSLA": ("TSLA", "GM", "F"),
    "JPM": ("JPM", "BAC", "XLF"),
    "BAC": ("BAC", "JPM", "XLF"),
    "XLF": ("XLF", "JPM", "BAC"),
    "XLE": ("XLE", "XOM", "CVX"),
    "XOM": ("XOM", "CVX", "XLE"),
    "CVX": ("CVX", "XOM", "XLE"),
    "SMH": ("SMH", "NVDA", "AMD"),
    "SOXX": ("SOXX", "NVDA", "AMD"),
    "TLT": ("TLT", "IEF", "SHY"),
    "GLD": ("GLD", "SLV", "DBC"),
}

EQUITY_PEER_GROUP_LABELS: dict[str, str] = {
    "SPY": "broad U.S. index basket",
    "QQQ": "growth-heavy index basket",
    "IWM": "small-cap index basket",
    "DIA": "large-cap industrial basket",
    "AAPL": "mega-cap technology peer basket",
    "MSFT": "mega-cap software peer basket",
    "NVDA": "semiconductor leadership basket",
    "AMD": "semiconductor peer basket",
    "AVGO": "semiconductor infrastructure basket",
    "GOOGL": "internet platform peer basket",
    "GOOG": "internet platform peer basket",
    "META": "digital advertising peer basket",
    "AMZN": "consumer platform peer basket",
    "TSLA": "auto and EV peer basket",
    "JPM": "money-center bank basket",
    "BAC": "money-center bank basket",
    "XLF": "financial sector basket",
    "XLE": "energy sector basket",
    "XOM": "energy major basket",
    "CVX": "energy major basket",
    "SMH": "semiconductor ETF basket",
    "SOXX": "semiconductor ETF basket",
    "TLT": "duration-sensitive rates basket",
    "GLD": "precious-metals basket",
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
        candidates = list(EQUITY_PEER_BASKETS.get(cleaned, (cleaned, "SPY", "QQQ")))
    deduped = list(dict.fromkeys(item for item in candidates if item))
    return tuple(deduped[:3] if len(deduped) >= 3 else deduped)


def describe_robustness_basket(symbol: str, asset_class: AssetClass | None = None) -> tuple[tuple[str, ...], str]:
    resolved_asset_class = asset_class or infer_asset_class(symbol)
    cleaned = str(symbol).strip().upper()
    basket = default_robustness_basket(cleaned, resolved_asset_class)

    if resolved_asset_class == "crypto":
        if cleaned in {"BTCUSD", "ETHUSD", "SOLUSD"}:
            reason = (
                "Crypto majors are compared against BTCUSD, ETHUSD, and SOLUSD so the signal has to generalize "
                "across liquid 24/7 leaders instead of looking good on only one coin."
            )
        else:
            anchors = ", ".join(item for item in basket if item != cleaned)
            reason = (
                f"Crypto robustness keeps {cleaned} in the basket, then adds liquid majors like {anchors} "
                "so the result is checked against broader market behavior instead of one isolated series."
            )
        return basket, reason

    if cleaned in EQUITY_PEER_BASKETS:
        group_label = EQUITY_PEER_GROUP_LABELS.get(cleaned, "recognized equity peer basket")
        reason = (
            f"Equity robustness uses a {group_label}: {', '.join(basket)}. "
            "That keeps the check closer to the symbol's actual peer set than a generic crypto-style basket."
        )
        return basket, reason

    reason = (
        f"{cleaned} does not have a dedicated peer map yet, so the app falls back to a broad equity basket "
        f"of {', '.join(basket)}. That is a reasonable first-pass robustness check, but it is less specific "
        "than a curated sector peer set."
    )
    return basket, reason


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
