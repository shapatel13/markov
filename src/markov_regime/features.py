from __future__ import annotations

import numpy as np
import pandas as pd

FORWARD_HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 24, 72)

FEATURE_PACKS: dict[str, tuple[str, ...]] = {
    "baseline": (
        "log_return_1",
        "log_return_3",
        "log_return_6",
        "trend_12",
        "trend_24",
        "vol_12",
        "vol_24",
        "range_ratio",
        "return_z_24",
        "volume_z_24",
    ),
    "trend": (
        "log_return_1",
        "log_return_3",
        "log_return_6",
        "trend_12",
        "trend_24",
        "trend_gap_12_24",
        "ema_gap_12",
        "ema_gap_24",
        "return_z_24",
        "volume_z_24",
    ),
    "volatility": (
        "log_return_1",
        "vol_12",
        "vol_24",
        "downside_vol_24",
        "vol_ratio_12_24",
        "range_ratio",
        "atr_ratio_14",
        "compression_24",
        "return_z_24",
        "volume_z_24",
    ),
    "regime_mix": (
        "log_return_1",
        "log_return_3",
        "trend_12",
        "trend_24",
        "trend_gap_12_24",
        "vol_12",
        "vol_24",
        "downside_vol_24",
        "range_ratio",
        "atr_ratio_14",
        "ema_gap_24",
        "compression_24",
        "return_z_24",
        "volume_z_24",
    ),
}

FEATURE_COLUMNS: tuple[str, ...] = FEATURE_PACKS["baseline"]


def list_feature_packs() -> tuple[str, ...]:
    return tuple(FEATURE_PACKS.keys())


def get_feature_columns(feature_pack: str = "baseline") -> tuple[str, ...]:
    try:
        return FEATURE_PACKS[feature_pack]
    except KeyError as exc:  # pragma: no cover - guarded by callers and tests
        supported = ", ".join(sorted(FEATURE_PACKS))
        raise ValueError(f"Unknown feature pack '{feature_pack}'. Supported packs: {supported}.") from exc


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def _compute_true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    components = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1)


def build_feature_frame(
    prices: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Cannot build features from an empty price frame.")

    frame = prices.copy()
    frame["bar_return"] = frame["close"].pct_change().fillna(0.0)
    log_close = np.log(frame["close"])
    log_return_1 = log_close.diff(1)

    frame["log_return_1"] = log_return_1
    frame["log_return_3"] = log_close.diff(3)
    frame["log_return_6"] = log_close.diff(6)
    frame["trend_12"] = frame["close"].pct_change(12)
    frame["trend_24"] = frame["close"].pct_change(24)
    frame["trend_gap_12_24"] = frame["trend_12"] - frame["trend_24"]
    frame["vol_12"] = log_return_1.rolling(12).std()
    frame["vol_24"] = log_return_1.rolling(24).std()
    downside_returns = log_return_1.where(log_return_1 < 0.0, 0.0)
    frame["downside_vol_24"] = downside_returns.rolling(24).std()
    frame["vol_ratio_12_24"] = frame["vol_12"] / frame["vol_24"].replace(0.0, np.nan)
    frame["range_ratio"] = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
    frame["atr_ratio_14"] = _compute_true_range(frame).rolling(14).mean() / frame["close"].replace(0.0, np.nan)
    frame["ema_gap_12"] = frame["close"] / frame["close"].ewm(span=12, adjust=False).mean() - 1.0
    frame["ema_gap_24"] = frame["close"] / frame["close"].ewm(span=24, adjust=False).mean() - 1.0
    frame["compression_24"] = frame["range_ratio"].rolling(24).mean() / frame["vol_24"].replace(0.0, np.nan)
    frame["return_z_24"] = _rolling_zscore(log_return_1, 24)
    frame["volume_z_24"] = _rolling_zscore(frame["volume"].replace(0.0, np.nan), 24).fillna(0.0)

    for horizon in FORWARD_HORIZONS:
        frame[f"forward_return_{horizon}"] = frame["close"].shift(-horizon) / frame["close"] - 1.0

    frame = frame.replace([np.inf, -np.inf], np.nan)
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Requested feature columns are missing from the frame: {missing_columns}")
    usable = frame.dropna(subset=list(feature_columns)).reset_index(drop=True)
    return usable
