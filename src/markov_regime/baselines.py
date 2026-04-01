from __future__ import annotations

import numpy as np
import pandas as pd

from markov_regime.config import Interval, StrategyConfig
from markov_regime.strategy import build_buy_and_hold_frame, compute_metrics, estimate_execution_cost_bps


def _bars_held_from_position(signal_position: pd.Series) -> pd.Series:
    bars_held: list[int] = []
    current_position = 0
    held = 0
    for position in signal_position.fillna(0.0).astype(int):
        if position != 0:
            held = held + 1 if position == current_position else 1
        else:
            held = 0
        current_position = position
        bars_held.append(held)
    return pd.Series(bars_held, index=signal_position.index, dtype="int64")


def _finalize_baseline_frame(frame: pd.DataFrame, position: pd.Series, config: StrategyConfig, label: str) -> pd.DataFrame:
    baseline = frame.copy()
    position = position.fillna(0.0).astype(int)
    baseline["candidate_action"] = position
    baseline["signal_position"] = position
    baseline["bars_held"] = _bars_held_from_position(position)
    baseline["guardrail_reason"] = label
    baseline["turnover"] = baseline["signal_position"].diff().abs().fillna(abs(int(baseline["signal_position"].iloc[0])))
    baseline["gross_strategy_return"] = baseline["signal_position"].shift(1).fillna(0.0) * baseline["bar_return"]
    baseline["execution_cost_bps"] = estimate_execution_cost_bps(baseline, config)
    baseline["transaction_cost"] = baseline["turnover"] * (baseline["execution_cost_bps"] / 10_000.0)
    baseline["net_strategy_return"] = baseline["gross_strategy_return"] - baseline["transaction_cost"]
    baseline["asset_wealth"] = (1.0 + baseline["bar_return"]).cumprod()
    baseline["strategy_wealth"] = (1.0 + baseline["net_strategy_return"]).cumprod()
    return baseline


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    return _true_range(frame).ewm(alpha=1.0 / window, adjust=False).mean()


def build_ema_trend_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    ema_fast = frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=48, adjust=False).mean()
    position = ((frame["close"] > ema_slow) & (ema_fast > ema_slow)).astype(int)
    return _finalize_baseline_frame(frame, position, config, "ema_trend")


def build_vol_filtered_trend_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    ema_fast = frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=48, adjust=False).mean()
    realized_vol = frame["bar_return"].rolling(24).std()
    vol_threshold = realized_vol.rolling(96).median().fillna(realized_vol.expanding().median())
    position = ((frame["close"] > ema_slow) & (ema_fast > ema_slow) & (realized_vol <= vol_threshold)).astype(int)
    return _finalize_baseline_frame(frame, position, config, "vol_filtered_trend")


def build_breakout_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    entry_level = frame["high"].rolling(20).max().shift(1)
    exit_level = frame["low"].rolling(10).min().shift(1)
    position_values: list[int] = []
    current_position = 0
    for close, entry_trigger, exit_trigger in zip(frame["close"], entry_level, exit_level, strict=True):
        if np.isnan(entry_trigger) or np.isnan(exit_trigger):
            current_position = 0
        elif current_position == 0 and close > entry_trigger:
            current_position = 1
        elif current_position == 1 and close < exit_trigger:
            current_position = 0
        position_values.append(current_position)
    position = pd.Series(position_values, index=frame.index, dtype="int64")
    return _finalize_baseline_frame(frame, position, config, "breakout")


def build_atr_trend_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    ema_mid = frame["close"].ewm(span=24, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=72, adjust=False).mean()
    atr = _atr(frame, 14)
    atr_ratio = atr / frame["close"].replace(0.0, np.nan)
    atr_scaled_momentum = frame["close"].pct_change(12) / atr_ratio.replace(0.0, np.nan)
    position = (
        (frame["close"] > ema_slow)
        & (ema_mid > ema_slow)
        & (atr_scaled_momentum > 0.0)
    ).astype(int)
    return _finalize_baseline_frame(frame, position, config, "atr_trend")


def build_atr_breakout_stop_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    entry_level = frame["high"].rolling(14).max().shift(1)
    atr = _atr(frame, 14)
    daily_bias = frame.get("daily_trend_20", pd.Series(0.0, index=frame.index)).fillna(0.0)

    current_position = 0
    high_since_entry = np.nan
    positions: list[int] = []
    for close, entry_trigger, atr_now, daily_trend in zip(frame["close"], entry_level, atr, daily_bias, strict=True):
        if np.isnan(entry_trigger) or np.isnan(atr_now):
            current_position = 0
            high_since_entry = np.nan
        elif current_position == 0 and close > entry_trigger and daily_trend > -0.01:
            current_position = 1
            high_since_entry = float(close)
        elif current_position == 1:
            high_since_entry = max(float(high_since_entry), float(close)) if not np.isnan(high_since_entry) else float(close)
            running_stop = high_since_entry - 3.0 * float(atr_now)
            if close < running_stop:
                current_position = 0
                high_since_entry = np.nan
        positions.append(current_position)
    position = pd.Series(positions, index=frame.index, dtype="int64")
    return _finalize_baseline_frame(frame, position, config, "atr_breakout_stop")


def build_daily_trend_filter_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    ema_slow = frame["close"].ewm(span=48, adjust=False).mean()
    daily_trend = frame.get("daily_trend_20", frame.get("trend_24", pd.Series(0.0, index=frame.index))).fillna(0.0)
    daily_gap = frame.get("daily_ema_gap_20", frame.get("ema_gap_24", pd.Series(0.0, index=frame.index))).fillna(0.0)
    daily_adx = frame.get("daily_adx_14", frame.get("adx_14", pd.Series(0.0, index=frame.index))).fillna(0.0)
    position = (
        (frame["close"] > ema_slow)
        & (daily_trend > 0.0)
        & (daily_gap > 0.0)
        & (daily_adx > 0.05)
    ).astype(int)
    return _finalize_baseline_frame(frame, position, config, "daily_trend_filter")


def build_baseline_frames(frame: pd.DataFrame, config: StrategyConfig) -> dict[str, pd.DataFrame]:
    return {
        "buy_and_hold": build_buy_and_hold_frame(frame),
        "ema_trend": build_ema_trend_baseline(frame, config),
        "vol_filtered_trend": build_vol_filtered_trend_baseline(frame, config),
        "breakout": build_breakout_baseline(frame, config),
        "atr_trend": build_atr_trend_baseline(frame, config),
        "atr_breakout_stop": build_atr_breakout_stop_baseline(frame, config),
        "daily_trend_filter": build_daily_trend_filter_baseline(frame, config),
    }


def summarize_baselines(frame: pd.DataFrame, interval: Interval, config: StrategyConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for baseline_name, baseline_frame in build_baseline_frames(frame, config).items():
        metrics = compute_metrics(baseline_frame, interval)
        rows.append(
            {
                "baseline": baseline_name,
                "sharpe": metrics["sharpe"],
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
                "trades": metrics["trades"],
                "trade_win_rate": metrics["trade_win_rate"],
                "expectancy": metrics["expectancy"],
                "exposure": metrics["exposure"],
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
