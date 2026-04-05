from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from markov_regime.config import Interval, StrategyConfig
from markov_regime.strategy import build_buy_and_hold_frame, compute_metrics, estimate_execution_cost_bps

BASELINE_DISPLAY_NAMES: dict[str, str] = {
    "buy_and_hold": "Buy and Hold",
    "ema_trend": "EMA Trend",
    "vol_filtered_trend": "Vol-Filtered Trend",
    "breakout": "Breakout",
    "atr_trend": "ATR Trend",
    "atr_breakout_stop": "ATR Breakout Stop",
    "daily_trend_filter": "Daily Trend Filter",
    "atr_causal_trend": "ATR Causal Trend",
    "daily_breakout_filter": "Daily Breakout Filter",
}


def baseline_display_name(name: str) -> str:
    return BASELINE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


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
    if "entry_trigger" not in baseline.columns:
        baseline["entry_trigger"] = baseline["high"].shift(1).fillna(baseline["high"])
    if "stop_level" not in baseline.columns:
        baseline["stop_level"] = baseline["low"].shift(1).fillna(baseline["low"])
    baseline["baseline_name"] = label
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
    baseline_frame = frame.assign(entry_trigger=ema_slow, stop_level=ema_slow)
    return _finalize_baseline_frame(baseline_frame, position, config, "ema_trend")


def build_vol_filtered_trend_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    ema_fast = frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=48, adjust=False).mean()
    realized_vol = frame["bar_return"].rolling(24).std()
    vol_threshold = realized_vol.rolling(96).median().fillna(realized_vol.expanding().median())
    position = ((frame["close"] > ema_slow) & (ema_fast > ema_slow) & (realized_vol <= vol_threshold)).astype(int)
    baseline_frame = frame.assign(entry_trigger=ema_slow, stop_level=ema_slow)
    return _finalize_baseline_frame(baseline_frame, position, config, "vol_filtered_trend")


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
    baseline_frame = frame.assign(entry_trigger=entry_level, stop_level=exit_level)
    return _finalize_baseline_frame(baseline_frame, position, config, "breakout")


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
    baseline_frame = frame.assign(entry_trigger=ema_slow, stop_level=(ema_slow - 2.0 * atr))
    return _finalize_baseline_frame(baseline_frame, position, config, "atr_trend")


def build_atr_breakout_stop_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    entry_level = frame["high"].rolling(14).max().shift(1)
    atr = _atr(frame, 14)
    daily_bias = frame.get("daily_trend_20", pd.Series(0.0, index=frame.index)).fillna(0.0)

    current_position = 0
    high_since_entry = np.nan
    positions: list[int] = []
    stop_levels: list[float] = []
    for close, entry_trigger, atr_now, daily_trend in zip(frame["close"], entry_level, atr, daily_bias, strict=True):
        if np.isnan(entry_trigger) or np.isnan(atr_now):
            current_position = 0
            high_since_entry = np.nan
            running_stop = np.nan
        elif current_position == 0 and close > entry_trigger and daily_trend > -0.01:
            current_position = 1
            high_since_entry = float(close)
            running_stop = high_since_entry - 3.0 * float(atr_now)
        elif current_position == 1:
            high_since_entry = max(float(high_since_entry), float(close)) if not np.isnan(high_since_entry) else float(close)
            running_stop = high_since_entry - 3.0 * float(atr_now)
            if close < running_stop:
                current_position = 0
                high_since_entry = np.nan
                running_stop = np.nan
        else:
            running_stop = np.nan
        positions.append(current_position)
        stop_levels.append(running_stop)
    position = pd.Series(positions, index=frame.index, dtype="int64")
    baseline_frame = frame.assign(entry_trigger=entry_level, stop_level=pd.Series(stop_levels, index=frame.index))
    return _finalize_baseline_frame(baseline_frame, position, config, "atr_breakout_stop")


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
    baseline_frame = frame.assign(entry_trigger=ema_slow, stop_level=ema_slow)
    return _finalize_baseline_frame(baseline_frame, position, config, "daily_trend_filter")


def build_atr_causal_trend_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    ema_slow = frame["close"].ewm(span=72, adjust=False).mean()
    atr = _atr(frame, 14)
    atr_ratio = (atr / frame["close"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    atr_momentum = frame.get("atr_momentum_24", frame["close"].pct_change(24) / atr_ratio.replace(0.0, np.nan)).fillna(0.0)
    causal_gap = frame.get("causal_gap_24", (frame["close"] - ema_slow) / ema_slow.replace(0.0, np.nan)).fillna(0.0)
    causal_slope = frame.get("causal_slope_24", ema_slow.pct_change(6)).fillna(0.0)
    position = (
        (frame["close"] > ema_slow)
        & (atr_momentum > 0.5)
        & (causal_gap > 0.0)
        & (causal_slope > 0.0)
    ).astype(int)
    baseline_frame = frame.assign(entry_trigger=ema_slow, stop_level=(ema_slow - 2.0 * atr))
    return _finalize_baseline_frame(baseline_frame, position, config, "atr_causal_trend")


def build_daily_breakout_filter_baseline(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    entry_level = frame["high"].rolling(20).max().shift(1)
    daily_trend = frame.get("daily_trend_20", frame.get("trend_24", pd.Series(0.0, index=frame.index))).fillna(0.0)
    daily_gap = frame.get("daily_ema_gap_20", frame.get("ema_gap_24", pd.Series(0.0, index=frame.index))).fillna(0.0)
    atr = _atr(frame, 14)

    current_position = 0
    peak_close = np.nan
    positions: list[int] = []
    stop_levels: list[float] = []
    for close, entry_trigger, trend_bias, gap_bias, atr_now in zip(
        frame["close"],
        entry_level,
        daily_trend,
        daily_gap,
        atr,
        strict=True,
    ):
        if np.isnan(entry_trigger) or np.isnan(atr_now):
            current_position = 0
            peak_close = np.nan
            trailing_stop = np.nan
        elif current_position == 0 and close > entry_trigger and trend_bias > 0.0 and gap_bias > 0.0:
            current_position = 1
            peak_close = float(close)
            trailing_stop = peak_close - 2.5 * float(atr_now)
        elif current_position == 1:
            peak_close = max(float(peak_close), float(close)) if not np.isnan(peak_close) else float(close)
            trailing_stop = peak_close - 2.5 * float(atr_now)
            if close < trailing_stop or trend_bias <= 0.0:
                current_position = 0
                peak_close = np.nan
                trailing_stop = np.nan
        else:
            trailing_stop = np.nan
        positions.append(current_position)
        stop_levels.append(trailing_stop)

    position = pd.Series(positions, index=frame.index, dtype="int64")
    baseline_frame = frame.assign(entry_trigger=entry_level, stop_level=pd.Series(stop_levels, index=frame.index))
    return _finalize_baseline_frame(baseline_frame, position, config, "daily_breakout_filter")


def build_baseline_frames(frame: pd.DataFrame, config: StrategyConfig) -> dict[str, pd.DataFrame]:
    return {
        "buy_and_hold": build_buy_and_hold_frame(frame),
        "ema_trend": build_ema_trend_baseline(frame, config),
        "vol_filtered_trend": build_vol_filtered_trend_baseline(frame, config),
        "breakout": build_breakout_baseline(frame, config),
        "atr_trend": build_atr_trend_baseline(frame, config),
        "atr_breakout_stop": build_atr_breakout_stop_baseline(frame, config),
        "daily_trend_filter": build_daily_trend_filter_baseline(frame, config),
        "atr_causal_trend": build_atr_causal_trend_baseline(frame, config),
        "daily_breakout_filter": build_daily_breakout_filter_baseline(frame, config),
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


def select_best_baseline_frame(
    frame: pd.DataFrame,
    interval: Interval,
    config: StrategyConfig,
    baseline_comparison: pd.DataFrame | None = None,
) -> tuple[str | None, pd.Series, pd.DataFrame]:
    comparison = baseline_comparison if baseline_comparison is not None and not baseline_comparison.empty else summarize_baselines(frame, interval, config)
    if comparison.empty:
        return None, pd.Series(dtype=object), pd.DataFrame()

    best_row = comparison.sort_values("sharpe", ascending=False).iloc[0]
    baseline_name = str(best_row["baseline"])
    frames = build_baseline_frames(frame, config)
    baseline_frame = frames.get(baseline_name, pd.DataFrame())
    return baseline_name, best_row, baseline_frame


def build_baseline_execution_plan(
    *,
    baseline_frame: pd.DataFrame,
    baseline_name: str,
    interval: Interval,
    live_price: float | None = None,
) -> dict[str, Any]:
    if baseline_frame.empty:
        return {
            "action": "No Entry",
            "severity": "warning",
            "summary": "The selected baseline did not produce a usable live signal frame.",
            "entry_guide": "There is no actionable baseline entry to follow right now.",
            "timing_note": "Baseline live guidance updates only on completed bars, just like the HMM research engine.",
            "held_position": "Flat",
            "engine_label": baseline_display_name(baseline_name),
        }

    latest = baseline_frame.iloc[-1]
    previous_position = int(baseline_frame["signal_position"].iloc[-2]) if len(baseline_frame) > 1 else 0
    current_position = int(latest.get("signal_position", 0))
    latest_close = float(latest.get("close", 0.0))
    latest_high = float(latest.get("high", latest_close))
    latest_low = float(latest.get("low", latest_close))
    entry_trigger = float(latest.get("entry_trigger", latest_high))
    stop_level = float(latest.get("stop_level", latest_low if current_position >= 0 else latest_high))
    reference_price = live_price if live_price is not None else latest_close
    bar_label = {"1hour": "1H", "4hour": "4H", "1day": "1D"}[interval]
    display_name = baseline_display_name(baseline_name)
    held_position = {1: "Long", 0: "Flat", -1: "Short"}.get(current_position, "Flat")

    if current_position == 1 and previous_position == 0:
        if "breakout" in baseline_name:
            setup_text = f"Wait for price to stay above the breakout trigger near {entry_trigger:,.2f}."
        else:
            setup_text = f"A conservative trigger is remaining above the trend line near {entry_trigger:,.2f}."
        return {
            "action": "Enter Long",
            "severity": "success",
            "summary": f"{display_name} approves a fresh long on the latest completed {bar_label} bar.",
            "entry_guide": (
                f"Aggressive entry: around {reference_price:,.2f}. "
                f"{setup_text} Initial invalidation is around {stop_level:,.2f}."
            ),
            "timing_note": f"This baseline only changes on completed {bar_label} bars. Do not treat intrabar spikes as confirmed entries.",
            "held_position": held_position,
            "engine_label": display_name,
        }
    if current_position == 1 and previous_position == 1:
        return {
            "action": "Hold Long",
            "severity": "success",
            "summary": f"{display_name} is already long and still supports holding the position.",
            "entry_guide": f"No fresh add is needed. The current trailing invalidation is around {stop_level:,.2f}.",
            "timing_note": f"Stay focused on completed {bar_label} closes. This baseline is trend-following, not tick-reactive.",
            "held_position": held_position,
            "engine_label": display_name,
        }
    if current_position == 0 and previous_position == 1:
        return {
            "action": "Exit to Flat",
            "severity": "warning",
            "summary": f"{display_name} has exited its prior long and is now flat.",
            "entry_guide": f"There is no fresh long entry right now. A new setup would need to reclaim roughly {entry_trigger:,.2f}.",
            "timing_note": f"The previous invalidation was around {stop_level:,.2f}, so the baseline is respecting its risk exit rather than forcing exposure.",
            "held_position": held_position,
            "engine_label": display_name,
        }
    if current_position == -1 and previous_position == 0:
        return {
            "action": "Enter Short",
            "severity": "success",
            "summary": f"{display_name} approves a fresh short on the latest completed {bar_label} bar.",
            "entry_guide": (
                f"Aggressive entry: around {reference_price:,.2f}. "
                f"Conservative trigger: only if price stays below {entry_trigger:,.2f}. "
                f"Invalidate back above {stop_level:,.2f}."
            ),
            "timing_note": f"This baseline only changes on completed {bar_label} bars.",
            "held_position": held_position,
            "engine_label": display_name,
        }
    if current_position == -1 and previous_position == -1:
        return {
            "action": "Hold Short",
            "severity": "success",
            "summary": f"{display_name} is already short and still supports holding the position.",
            "entry_guide": f"No fresh add is needed. The current invalidation is around {stop_level:,.2f}.",
            "timing_note": f"Stay focused on completed {bar_label} closes.",
            "held_position": held_position,
            "engine_label": display_name,
        }

    if "breakout" in baseline_name:
        no_entry_text = f"No baseline long is active. A new setup would need a completed {bar_label} close above roughly {entry_trigger:,.2f}."
    else:
        no_entry_text = f"No baseline long is active. The trend filter would need to improve back above roughly {entry_trigger:,.2f}."

    return {
        "action": "No Entry",
        "severity": "warning",
        "summary": f"{display_name} is flat on the latest completed {bar_label} bar.",
        "entry_guide": no_entry_text,
        "timing_note": "This is a valid outcome. A professional system should prefer flat over marginal exposure.",
        "held_position": held_position,
        "engine_label": display_name,
    }
