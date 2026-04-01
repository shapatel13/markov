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
    "trend_strength": (
        "log_return_1",
        "trend_12",
        "trend_24",
        "trend_gap_12_24",
        "ema_gap_12",
        "ema_gap_24",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "donchian_position_20",
        "breakout_distance_20",
        "volume_z_24",
    ),
    "mean_reversion": (
        "log_return_1",
        "rsi_14",
        "stoch_k_14",
        "bollinger_z_20",
        "bollinger_bandwidth_20",
        "donchian_position_20",
        "distance_to_vwap_24",
        "return_z_24",
        "range_ratio",
        "volume_z_24",
    ),
    "vol_surface": (
        "log_return_1",
        "vol_12",
        "vol_24",
        "downside_vol_24",
        "vol_ratio_12_24",
        "realized_skew_24",
        "realized_kurt_24",
        "parkinson_vol_24",
        "garman_klass_vol_24",
        "bollinger_bandwidth_20",
        "compression_24",
    ),
    "regime_mix_v2": (
        "log_return_1",
        "log_return_3",
        "trend_12",
        "trend_24",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "vol_12",
        "vol_24",
        "realized_skew_24",
        "bollinger_z_20",
        "bollinger_bandwidth_20",
        "donchian_position_20",
        "distance_to_vwap_24",
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


def _wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False).mean()


def _compute_directional_indicators(frame: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = frame["high"].diff()
    down_move = frame["low"].shift(1) - frame["low"]
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0).fillna(0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0).fillna(0.0)

    true_range = _compute_true_range(frame)
    atr = _wilder_smooth(true_range, window).replace(0.0, np.nan)
    plus_di = _wilder_smooth(plus_dm, window) / atr
    minus_di = _wilder_smooth(minus_dm, window) / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = _wilder_smooth(dx, window)
    return adx, plus_di, minus_di


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = _wilder_smooth(gains, window)
    avg_loss = _wilder_smooth(losses, window)
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 1.0 - (1.0 / (1.0 + relative_strength))
    return rsi.fillna(0.5)


def _compute_stochastic_k(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    rolling_low = frame["low"].rolling(window).min()
    rolling_high = frame["high"].rolling(window).max()
    return (frame["close"] - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)


def _compute_bollinger_features(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    z_score = (close - rolling_mean) / rolling_std.replace(0.0, np.nan)
    bandwidth = (4.0 * rolling_std) / rolling_mean.replace(0.0, np.nan)
    return z_score, bandwidth


def _compute_donchian_features(frame: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    rolling_high = frame["high"].rolling(window).max().shift(1)
    rolling_low = frame["low"].rolling(window).min().shift(1)
    position = (frame["close"] - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
    breakout_distance = frame["close"] / rolling_high.replace(0.0, np.nan) - 1.0
    return position, breakout_distance


def _compute_rolling_vwap(close: pd.Series, volume: pd.Series, window: int = 24) -> pd.Series:
    weighted_price = close * volume.fillna(0.0)
    rolling_volume = volume.fillna(0.0).rolling(window).sum()
    return weighted_price.rolling(window).sum() / rolling_volume.replace(0.0, np.nan)


def _compute_parkinson_volatility(frame: pd.DataFrame, window: int = 24) -> pd.Series:
    log_range = np.log(frame["high"] / frame["low"].replace(0.0, np.nan))
    estimator = (log_range.pow(2) / (4.0 * np.log(2.0))).rolling(window).mean()
    return np.sqrt(estimator.clip(lower=0.0))


def _compute_garman_klass_volatility(frame: pd.DataFrame, window: int = 24) -> pd.Series:
    log_hl = np.log(frame["high"] / frame["low"].replace(0.0, np.nan))
    log_co = np.log(frame["close"] / frame["open"].replace(0.0, np.nan))
    estimator = (0.5 * log_hl.pow(2) - ((2.0 * np.log(2.0)) - 1.0) * log_co.pow(2)).rolling(window).mean()
    return np.sqrt(estimator.clip(lower=0.0))


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
    frame["adx_14"], frame["plus_di_14"], frame["minus_di_14"] = _compute_directional_indicators(frame, 14)
    frame["rsi_14"] = _compute_rsi(frame["close"], 14)
    frame["stoch_k_14"] = _compute_stochastic_k(frame, 14)
    frame["bollinger_z_20"], frame["bollinger_bandwidth_20"] = _compute_bollinger_features(frame["close"], 20)
    frame["donchian_position_20"], frame["breakout_distance_20"] = _compute_donchian_features(frame, 20)
    rolling_vwap_24 = _compute_rolling_vwap(frame["close"], frame["volume"].replace(0.0, np.nan), 24)
    frame["distance_to_vwap_24"] = frame["close"] / rolling_vwap_24.replace(0.0, np.nan) - 1.0
    frame["realized_skew_24"] = log_return_1.rolling(24).skew()
    frame["realized_kurt_24"] = log_return_1.rolling(24).kurt()
    frame["parkinson_vol_24"] = _compute_parkinson_volatility(frame, 24)
    frame["garman_klass_vol_24"] = _compute_garman_klass_volatility(frame, 24)

    for horizon in FORWARD_HORIZONS:
        frame[f"forward_return_{horizon}"] = frame["close"].shift(-horizon) / frame["close"] - 1.0

    frame = frame.replace([np.inf, -np.inf], np.nan)
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Requested feature columns are missing from the frame: {missing_columns}")
    usable = frame.dropna(subset=list(feature_columns)).reset_index(drop=True)
    return usable
