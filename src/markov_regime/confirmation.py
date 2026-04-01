from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from markov_regime.bootstrap import block_bootstrap_confidence_intervals
from markov_regime.baselines import summarize_baselines
from markov_regime.config import Interval, StrategyConfig
from markov_regime.strategy import (
    build_buy_and_hold_frame,
    build_trade_table,
    compute_metrics,
    estimate_execution_cost_bps,
    stress_test_transaction_costs,
    summarize_trade_table,
)
from markov_regime.walkforward import WalkForwardResult


def _guardrail_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        predictions.assign(guardrail_reason=lambda frame: frame["guardrail_reason"].replace("", "accepted"))
        .groupby("guardrail_reason", dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    counts["share"] = counts["count"] / max(len(predictions), 1)
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def _effective_confirmation_direction(frame: pd.DataFrame) -> pd.Series:
    candidate = frame["candidate_action"].fillna(0.0).astype(int)
    signal = frame["signal_position"].fillna(0.0).astype(int)
    return pd.Series(np.where(candidate != 0, candidate, signal), index=frame.index, dtype="int64")


def _recompute_bars_held(signal_position: pd.Series) -> pd.Series:
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


def align_confirmation_predictions(
    primary_predictions: pd.DataFrame,
    confirmation_predictions: pd.DataFrame,
    confirmation_interval: Interval = "1day",
) -> pd.DataFrame:
    left = primary_predictions.sort_values("timestamp").copy()
    right = confirmation_predictions.sort_values("timestamp").copy()
    right = right.assign(
        confirmation_timestamp=lambda frame: pd.to_datetime(frame["timestamp"]),
        confirmation_signal_position=lambda frame: frame["signal_position"].astype(int),
        confirmation_candidate_action=lambda frame: frame["candidate_action"].astype(int),
        confirmation_guardrail_reason=lambda frame: frame["guardrail_reason"].replace("", "accepted"),
        confirmation_max_posterior=lambda frame: frame["max_posterior"].astype(float),
        confirmation_confidence_gap=lambda frame: frame["confidence_gap"].astype(float),
    )
    right["confirmation_effective_direction"] = _effective_confirmation_direction(right)
    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.loc[
            :,
            [
                "timestamp",
                "confirmation_timestamp",
                "confirmation_signal_position",
                "confirmation_candidate_action",
                "confirmation_guardrail_reason",
                "confirmation_max_posterior",
                "confirmation_confidence_gap",
                "confirmation_effective_direction",
            ],
        ].rename(columns={"timestamp": "confirmation_source_timestamp"}),
        left_on="timestamp",
        right_on="confirmation_source_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["confirmation_interval"] = confirmation_interval
    return merged


def _confirmation_reason(status: str) -> str:
    return {
        "confirmed": "",
        "neutral": "",
        "blocked": "daily_confirmation_opposes",
        "unavailable": "daily_confirmation_unavailable",
        "no_primary_signal": "",
    }.get(status, "")


def apply_confirmation_overlay(signal_frame: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.require_daily_confirmation or "confirmation_effective_direction" not in signal_frame.columns:
        return signal_frame, pd.DataFrame()

    overlay = signal_frame.copy()
    overlay["base_signal_position"] = overlay["signal_position"].astype(int)
    overlay["base_candidate_action"] = overlay["candidate_action"].astype(int)
    overlay["base_guardrail_reason"] = overlay["guardrail_reason"].fillna("")
    overlay["confirmation_effective_direction"] = overlay["confirmation_effective_direction"].fillna(0.0)
    overlay["confirmation_signal_position"] = overlay["confirmation_signal_position"].fillna(0.0)
    overlay["confirmation_candidate_action"] = overlay["confirmation_candidate_action"].fillna(0.0)
    overlay["confirmation_guardrail_reason"] = overlay["confirmation_guardrail_reason"].fillna("unavailable")

    requested_direction = np.where(
        overlay["base_candidate_action"] != 0,
        overlay["base_candidate_action"],
        overlay["base_signal_position"],
    ).astype(int)
    overlay["requested_direction"] = requested_direction

    confirmation_available = overlay["confirmation_source_timestamp"].notna()
    confirmation_direction = overlay["confirmation_effective_direction"].astype(int)
    same_direction = (
        (requested_direction != 0)
        & confirmation_available
        & (confirmation_direction != 0)
        & (np.sign(requested_direction) == np.sign(confirmation_direction))
    )
    neutral_mask = (requested_direction != 0) & confirmation_available & (confirmation_direction == 0)
    blocked_mask = (
        (requested_direction != 0)
        & confirmation_available
        & (confirmation_direction != 0)
        & (np.sign(requested_direction) != np.sign(confirmation_direction))
    )
    unavailable_mask = (requested_direction != 0) & ~confirmation_available
    no_primary_signal = requested_direction == 0

    overlay["confirmation_status"] = np.select(
        [same_direction, neutral_mask, blocked_mask, unavailable_mask, no_primary_signal],
        ["confirmed", "neutral", "blocked", "unavailable", "no_primary_signal"],
        default="no_primary_signal",
    )
    overlay["confirmation_reason"] = overlay["confirmation_status"].map(_confirmation_reason)

    allow_mask = overlay["confirmation_status"].isin(["confirmed", "neutral"])
    overlay["signal_position"] = np.where(allow_mask, overlay["base_signal_position"], 0).astype(int)
    overlay["candidate_action"] = np.where(allow_mask, overlay["base_candidate_action"], 0).astype(int)
    overlay["guardrail_reason"] = np.where(
        overlay["confirmation_status"].isin(["blocked", "unavailable"]),
        overlay["confirmation_reason"],
        overlay["base_guardrail_reason"],
    )
    overlay["guardrail_reason"] = pd.Series(overlay["guardrail_reason"], index=overlay.index).fillna("").astype(str)
    overlay["bars_held"] = _recompute_bars_held(overlay["signal_position"])

    overlay["turnover"] = overlay["signal_position"].diff().abs().fillna(abs(int(overlay["signal_position"].iloc[0])))
    overlay["gross_strategy_return"] = overlay["signal_position"].shift(1).fillna(0.0) * overlay["bar_return"]
    overlay["execution_cost_bps"] = estimate_execution_cost_bps(overlay, config)
    overlay["transaction_cost"] = overlay["turnover"] * (overlay["execution_cost_bps"] / 10_000.0)
    overlay["net_strategy_return"] = overlay["gross_strategy_return"] - overlay["transaction_cost"]
    overlay["asset_wealth"] = (1.0 + overlay["bar_return"]).cumprod()
    overlay["strategy_wealth"] = (1.0 + overlay["net_strategy_return"]).cumprod()

    requested_mask = overlay["requested_direction"] != 0
    summary = (
        overlay.groupby("confirmation_status", dropna=False)
        .agg(
            bars=("timestamp", "size"),
            requested_bars=("requested_direction", lambda values: int((values != 0).sum())),
        )
        .reset_index()
    )
    summary["share_of_bars"] = summary["bars"] / max(len(overlay), 1)
    summary["share_of_requested"] = summary["requested_bars"] / max(int(requested_mask.sum()), 1)
    summary = summary.sort_values(["requested_bars", "bars"], ascending=False).reset_index(drop=True)
    return overlay, summary


def _recompute_fold_diagnostics(
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    interval: Interval,
) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics

    updated = diagnostics.copy()
    metric_columns = set(updated.columns)
    for fold_id, fold_frame in predictions.groupby("fold_id", sort=True):
        metrics = compute_metrics(fold_frame, interval)
        for metric_name, metric_value in metrics.items():
            if metric_name in metric_columns:
                updated.loc[updated["fold_id"] == fold_id, metric_name] = metric_value
    return updated


def apply_higher_timeframe_confirmation(
    primary_result: WalkForwardResult,
    confirmation_result: WalkForwardResult,
    *,
    interval: Interval,
    strategy_config: StrategyConfig,
    confirmation_interval: Interval = "1day",
) -> WalkForwardResult:
    if not strategy_config.require_daily_confirmation:
        return primary_result

    aligned = align_confirmation_predictions(primary_result.predictions, confirmation_result.predictions, confirmation_interval)
    confirmed_predictions, confirmation_summary = apply_confirmation_overlay(aligned, strategy_config)
    updated_metrics = compute_metrics(confirmed_predictions, interval)
    updated_diagnostics = _recompute_fold_diagnostics(confirmed_predictions, primary_result.fold_diagnostics, interval)
    updated_cost_stress = stress_test_transaction_costs(confirmed_predictions, strategy_config.cost_grid, interval, strategy_config)
    updated_bootstrap = block_bootstrap_confidence_intervals(
        confirmed_predictions["net_strategy_return"],
        interval=interval,
        block_length=max(strategy_config.signal_horizon * 2, 8),
    )
    updated_guardrail_summary = _guardrail_summary(confirmed_predictions)
    updated_trade_log = build_trade_table(confirmed_predictions)
    updated_trade_summary = summarize_trade_table(updated_trade_log)
    updated_baseline_comparison = summarize_baselines(confirmed_predictions, interval, strategy_config)

    return replace(
        primary_result,
        predictions=confirmed_predictions,
        fold_diagnostics=updated_diagnostics,
        metrics=updated_metrics,
        benchmark_metrics=compute_metrics(build_buy_and_hold_frame(confirmed_predictions), interval),
        cost_stress=updated_cost_stress,
        bootstrap=updated_bootstrap,
        guardrail_summary=updated_guardrail_summary,
        confirmation_summary=confirmation_summary,
        trade_log=updated_trade_log,
        trade_summary=updated_trade_summary,
        baseline_comparison=updated_baseline_comparison,
    )
