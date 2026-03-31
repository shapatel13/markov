from __future__ import annotations

import numpy as np
import pandas as pd

FORWARD_HORIZONS: tuple[int, ...] = (1, 3, 6, 12, 24, 72)
FEATURE_COLUMNS: tuple[str, ...] = (
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
)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Cannot build features from an empty price frame.")

    frame = prices.copy()
    frame["bar_return"] = frame["close"].pct_change().fillna(0.0)
    log_close = np.log(frame["close"])

    frame["log_return_1"] = log_close.diff(1)
    frame["log_return_3"] = log_close.diff(3)
    frame["log_return_6"] = log_close.diff(6)
    frame["trend_12"] = frame["close"].pct_change(12)
    frame["trend_24"] = frame["close"].pct_change(24)
    frame["vol_12"] = frame["log_return_1"].rolling(12).std()
    frame["vol_24"] = frame["log_return_1"].rolling(24).std()
    frame["range_ratio"] = (frame["high"] - frame["low"]) / frame["close"].replace(0.0, np.nan)
    frame["return_z_24"] = _rolling_zscore(frame["log_return_1"], 24)
    frame["volume_z_24"] = _rolling_zscore(frame["volume"].replace(0.0, np.nan), 24).fillna(0.0)

    for horizon in FORWARD_HORIZONS:
        frame[f"forward_return_{horizon}"] = frame["close"].shift(-horizon) / frame["close"] - 1.0

    frame = frame.replace([np.inf, -np.inf], np.nan)
    usable = frame.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    return usable
