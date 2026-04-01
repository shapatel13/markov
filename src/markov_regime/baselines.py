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


def build_baseline_frames(frame: pd.DataFrame, config: StrategyConfig) -> dict[str, pd.DataFrame]:
    return {
        "buy_and_hold": build_buy_and_hold_frame(frame),
        "ema_trend": build_ema_trend_baseline(frame, config),
        "vol_filtered_trend": build_vol_filtered_trend_baseline(frame, config),
        "breakout": build_breakout_baseline(frame, config),
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
