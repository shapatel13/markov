from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from markov_regime.bootstrap import block_bootstrap_confidence_intervals
from markov_regime.baselines import summarize_baselines
from markov_regime.config import DataConfig, Interval, ModelConfig, StrategyConfig, default_walk_forward_config
from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.data import DataFetchResult, fetch_price_data
from markov_regime.features import build_feature_frame
from markov_regime.strategy import (
    build_buy_and_hold_frame,
    build_trade_table,
    compute_metrics,
    estimate_execution_cost_bps,
    stress_test_transaction_costs,
    summarize_trade_table,
)
from markov_regime.walkforward import WalkForwardResult, run_walk_forward, suggest_walk_forward_config


@dataclass(frozen=True)
class ConsensusDiagnostics:
    members: pd.DataFrame
    timeline: pd.DataFrame
    summary: pd.DataFrame


def _position_label(value: int) -> str:
    return {1: "Long", 0: "Flat", -1: "Short"}.get(int(value), "Flat")


def _default_seed_values(base_seed: int) -> tuple[int, ...]:
    return tuple(sorted({max(1, base_seed - 4), base_seed, base_seed + 4}))


def _default_state_counts(base_n_states: int) -> tuple[int, ...]:
    return tuple(sorted({state for state in (base_n_states - 1, base_n_states, base_n_states + 1) if 5 <= state <= 9}))


def _resolve_feature_context(
    *,
    symbol: str,
    interval: Interval,
    limit: int,
    feature_columns: tuple[str, ...],
    auto_adjust_windows: bool,
    cache: dict[tuple[str, Interval, int, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame, object, bool]],
) -> tuple[DataFetchResult, pd.DataFrame, object, bool]:
    key = (symbol, interval, limit, feature_columns)
    cached = cache.get(key)
    if cached is not None:
        return cached

    fetched = fetch_price_data(DataConfig(symbol=symbol, interval=interval, limit=limit))
    feature_frame = build_feature_frame(fetched.frame, feature_columns=feature_columns)
    walk_config, was_adjusted = (
        suggest_walk_forward_config(len(feature_frame), default_walk_forward_config(interval))
        if auto_adjust_windows
        else (default_walk_forward_config(interval), False)
    )
    cache[key] = (fetched, feature_frame, walk_config, was_adjusted)
    return cache[key]


def _run_member(
    *,
    symbol: str,
    interval: Interval,
    limit: int,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    auto_adjust_windows: bool,
    cache: dict[tuple[str, Interval, int, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame, object, bool]],
) -> tuple[WalkForwardResult, DataFetchResult, bool]:
    fetched, feature_frame, walk_config, was_adjusted = _resolve_feature_context(
        symbol=symbol,
        interval=interval,
        limit=limit,
        feature_columns=feature_columns,
        auto_adjust_windows=auto_adjust_windows,
        cache=cache,
    )
    base_strategy = replace(strategy_config, require_daily_confirmation=False)
    result = run_walk_forward(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        interval=interval,
        model_config=model_config,
        walk_config=walk_config,
        strategy_config=base_strategy,
    )

    if interval == "4hour" and strategy_config.require_daily_confirmation:
        confirmation_fetched, confirmation_features, confirmation_walk_config, _ = _resolve_feature_context(
            symbol=symbol,
            interval="1day",
            limit=limit,
            feature_columns=feature_columns,
            auto_adjust_windows=auto_adjust_windows,
            cache=cache,
        )
        confirmation_result = run_walk_forward(
            feature_frame=confirmation_features,
            feature_columns=feature_columns,
            interval="1day",
            model_config=model_config,
            walk_config=confirmation_walk_config,
            strategy_config=base_strategy,
        )
        result = apply_higher_timeframe_confirmation(
            result,
            confirmation_result,
            interval=interval,
            strategy_config=strategy_config,
            confirmation_interval="1day",
        )
        fetched = fetched if fetched is not None else confirmation_fetched

    return result, fetched, was_adjusted


def _majority_vote(values: pd.Series) -> tuple[int, float, int, int, int]:
    valid = values.dropna().astype(int)
    if valid.empty:
        return 0, 0.0, 0, 0, 0
    counts = {
        -1: int((valid == -1).sum()),
        0: int((valid == 0).sum()),
        1: int((valid == 1).sum()),
    }
    majority_value = max(counts, key=lambda item: (counts[item], item == 0, item == 1))
    majority_share = counts[majority_value] / len(valid)
    return int(majority_value), float(majority_share), counts[1], counts[0], counts[-1]


def build_consensus_timeline(member_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not member_predictions:
        return pd.DataFrame()

    merged: pd.DataFrame | None = None
    position_columns: list[str] = []
    candidate_columns: list[str] = []

    for member_id, predictions in member_predictions.items():
        current = predictions.loc[:, ["timestamp", "close", "signal_position", "candidate_action"]].copy()
        current = current.rename(
            columns={
                "signal_position": f"signal_position__{member_id}",
                "candidate_action": f"candidate_action__{member_id}",
            }
        )
        position_columns.append(f"signal_position__{member_id}")
        candidate_columns.append(f"candidate_action__{member_id}")
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current.drop(columns=["close"]), on="timestamp", how="outer")

    assert merged is not None
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for row in merged.itertuples(index=False):
        row_dict = row._asdict()
        position_vote, position_share, long_votes, flat_votes, short_votes = _majority_vote(
            pd.Series([row_dict[column] for column in position_columns], dtype="float64")
        )
        candidate_vote, candidate_share, candidate_long_votes, candidate_flat_votes, candidate_short_votes = _majority_vote(
            pd.Series([row_dict[column] for column in candidate_columns], dtype="float64")
        )
        member_count = max(len(position_columns), 1)
        rows.append(
            {
                "timestamp": row_dict["timestamp"],
                "close": float(row_dict.get("close", np.nan)),
                "position_consensus": position_vote,
                "position_consensus_label": _position_label(position_vote),
                "position_consensus_share": position_share,
                "candidate_consensus": candidate_vote,
                "candidate_consensus_label": _position_label(candidate_vote),
                "candidate_consensus_share": candidate_share,
                "long_votes": long_votes,
                "flat_votes": flat_votes,
                "short_votes": short_votes,
                "candidate_long_votes": candidate_long_votes,
                "candidate_flat_votes": candidate_flat_votes,
                "candidate_short_votes": candidate_short_votes,
                "active_member_share": (long_votes + short_votes) / member_count,
                "active_candidate_share": (candidate_long_votes + candidate_short_votes) / member_count,
                "member_count": member_count,
            }
        )
    return pd.DataFrame(rows)


def summarize_consensus(members: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    if members.empty or timeline.empty:
        return pd.DataFrame(columns=["metric", "value", "interpretation"])

    latest = timeline.iloc[-1]
    latest_position_share = float(latest["position_consensus_share"])
    latest_candidate_share = float(latest["candidate_consensus_share"])
    avg_position_share = float(timeline["position_consensus_share"].mean())
    avg_candidate_share = float(timeline["candidate_consensus_share"].mean())
    median_member_sharpe = float(members["sharpe"].median())
    stability = float(members["stability_score"].median()) if "stability_score" in members.columns else 0.0

    rows = [
        {
            "metric": "Latest Held Consensus",
            "value": f"{latest['position_consensus_label']} ({latest_position_share:.0%})",
            "interpretation": "How strongly the current executed position agrees across nearby state counts and seeds.",
        },
        {
            "metric": "Latest Candidate Consensus",
            "value": f"{latest['candidate_consensus_label']} ({latest_candidate_share:.0%})",
            "interpretation": "How strongly the newest bar supports the same fresh action across nearby state counts and seeds.",
        },
        {
            "metric": "Average Held Consensus",
            "value": f"{avg_position_share:.0%}",
            "interpretation": "Average agreement on executed position across the full backtest path. Higher is more stable.",
        },
        {
            "metric": "Average Candidate Consensus",
            "value": f"{avg_candidate_share:.0%}",
            "interpretation": "Average agreement on fresh candidate actions. Lower values often signal fragile entry timing.",
        },
        {
            "metric": "Median Member Sharpe",
            "value": f"{median_member_sharpe:.2f}",
            "interpretation": "Median performance across the consensus members. This is a better trust check than the single best run.",
        },
        {
            "metric": "Median Member Stability",
            "value": f"{stability:.2f}",
            "interpretation": "Median state-stability score across consensus members.",
        },
        {
            "metric": "Consensus Members",
            "value": f"{len(members)}",
            "interpretation": "Total nearby models included in the consensus panel.",
        },
    ]
    return pd.DataFrame(rows)


def align_consensus_predictions(primary_predictions: pd.DataFrame, consensus_timeline: pd.DataFrame) -> pd.DataFrame:
    if consensus_timeline.empty:
        return primary_predictions
    right = consensus_timeline.sort_values("timestamp").rename(
        columns={
            "timestamp": "consensus_timestamp",
            "position_consensus": "consensus_position",
            "position_consensus_share": "consensus_position_share",
            "candidate_consensus": "consensus_candidate",
            "candidate_consensus_share": "consensus_candidate_share",
            "member_count": "consensus_member_count",
        }
    )
    merged = pd.merge_asof(
        primary_predictions.sort_values("timestamp"),
        right.loc[
            :,
            [
                "consensus_timestamp",
                "consensus_position",
                "consensus_position_share",
                "consensus_candidate",
                "consensus_candidate_share",
                "consensus_member_count",
            ],
        ],
        left_on="timestamp",
        right_on="consensus_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


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


def _consensus_reason(status: str) -> str:
    return {
        "confirmed": "",
        "weak_share": "consensus_weak_share",
        "flat_consensus": "consensus_flat",
        "opposed": "consensus_opposes",
        "unavailable": "consensus_unavailable",
        "no_primary_signal": "",
    }.get(status, "")


def _consensus_hold_reason(status: str) -> str:
    return {
        "weak_share": "consensus_hold_weak_share",
        "flat_consensus": "consensus_hold_flat",
        "opposed": "consensus_hold_opposed",
        "unavailable": "consensus_hold_unavailable",
    }.get(status, "")


def apply_consensus_overlay(signal_frame: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"consensus_position", "consensus_position_share", "consensus_candidate", "consensus_candidate_share"}
    if not config.require_consensus_confirmation or not required_columns.issubset(signal_frame.columns):
        return signal_frame, pd.DataFrame()

    overlay = signal_frame.copy()
    overlay["consensus_base_signal_position"] = overlay["signal_position"].astype(int)
    overlay["consensus_base_candidate_action"] = overlay["candidate_action"].astype(int)
    overlay["consensus_base_guardrail_reason"] = overlay["guardrail_reason"].fillna("")
    overlay["consensus_base_turnover"] = overlay["turnover"].fillna(0.0).astype(float) if "turnover" in overlay.columns else 0.0
    overlay["consensus_position"] = overlay["consensus_position"].fillna(0.0).astype(int)
    overlay["consensus_candidate"] = overlay["consensus_candidate"].fillna(0.0).astype(int)
    overlay["consensus_position_share"] = overlay["consensus_position_share"].fillna(0.0).astype(float)
    overlay["consensus_candidate_share"] = overlay["consensus_candidate_share"].fillna(0.0).astype(float)

    requested_direction = np.where(
        overlay["consensus_base_candidate_action"] != 0,
        overlay["consensus_base_candidate_action"],
        overlay["consensus_base_signal_position"],
    ).astype(int)
    use_candidate_consensus = overlay["consensus_base_candidate_action"] != 0
    consensus_direction = np.where(use_candidate_consensus, overlay["consensus_candidate"], overlay["consensus_position"]).astype(int)
    consensus_share = np.where(use_candidate_consensus, overlay["consensus_candidate_share"], overlay["consensus_position_share"]).astype(float)
    overlay["consensus_requested_direction"] = requested_direction
    overlay["consensus_effective_direction"] = consensus_direction
    overlay["consensus_effective_share"] = consensus_share

    available = overlay["consensus_timestamp"].notna()
    no_primary_signal = requested_direction == 0
    confirmed = (
        (requested_direction != 0)
        & available
        & (consensus_share >= config.consensus_min_share)
        & (consensus_direction != 0)
        & (np.sign(requested_direction) == np.sign(consensus_direction))
    )
    weak_share = (requested_direction != 0) & available & (consensus_share < config.consensus_min_share)
    flat_consensus = (requested_direction != 0) & available & (consensus_share >= config.consensus_min_share) & (consensus_direction == 0)
    opposed = (
        (requested_direction != 0)
        & available
        & (consensus_share >= config.consensus_min_share)
        & (consensus_direction != 0)
        & (np.sign(requested_direction) != np.sign(consensus_direction))
    )
    unavailable = (requested_direction != 0) & ~available

    overlay["consensus_status"] = np.select(
        [confirmed, weak_share, flat_consensus, opposed, unavailable, no_primary_signal],
        ["confirmed", "weak_share", "flat_consensus", "opposed", "unavailable", "no_primary_signal"],
        default="no_primary_signal",
    )
    overlay["consensus_reason"] = overlay["consensus_status"].map(_consensus_reason)

    allow_mask = overlay["consensus_status"] == "confirmed"
    entry_attempt = (overlay["consensus_base_turnover"] > 0.0) & (overlay["consensus_base_signal_position"] != 0)
    carry_existing_hold = (
        (config.consensus_gate_mode == "entry_only")
        & ~allow_mask
        & (overlay["consensus_status"].isin(["weak_share", "flat_consensus", "opposed", "unavailable"]))
        & (overlay["consensus_base_signal_position"] != 0)
        & ~entry_attempt
    )

    overlay["signal_position"] = np.where(
        allow_mask | carry_existing_hold,
        overlay["consensus_base_signal_position"],
        0,
    ).astype(int)
    overlay["candidate_action"] = np.where(allow_mask, overlay["consensus_base_candidate_action"], 0).astype(int)
    hold_reasons = overlay["consensus_status"].map(_consensus_hold_reason)
    overlay["guardrail_reason"] = np.where(
        carry_existing_hold,
        hold_reasons,
        np.where(
            overlay["consensus_status"].isin(["weak_share", "flat_consensus", "opposed", "unavailable"]),
            overlay["consensus_reason"],
            overlay["consensus_base_guardrail_reason"],
        ),
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

    requested_mask = overlay["consensus_requested_direction"] != 0
    summary = (
        overlay.groupby("consensus_status", dropna=False)
        .agg(
            bars=("timestamp", "size"),
            requested_bars=("consensus_requested_direction", lambda values: int((values != 0).sum())),
        )
        .reset_index()
    )
    summary["share_of_bars"] = summary["bars"] / max(len(overlay), 1)
    summary["share_of_requested"] = summary["requested_bars"] / max(int(requested_mask.sum()), 1)
    return overlay, summary.sort_values(["requested_bars", "bars"], ascending=False).reset_index(drop=True)


def apply_consensus_confirmation(
    primary_result: WalkForwardResult,
    diagnostics: ConsensusDiagnostics,
    *,
    interval: Interval,
    strategy_config: StrategyConfig,
) -> WalkForwardResult:
    if not strategy_config.require_consensus_confirmation:
        return primary_result
    aligned = align_consensus_predictions(primary_result.predictions, diagnostics.timeline)
    confirmed_predictions, consensus_summary = apply_consensus_overlay(aligned, strategy_config)
    updated_metrics = compute_metrics(confirmed_predictions, interval)
    updated_cost_stress = stress_test_transaction_costs(confirmed_predictions, strategy_config.cost_grid, interval, strategy_config)
    updated_bootstrap = block_bootstrap_confidence_intervals(
        confirmed_predictions["net_strategy_return"],
        interval=interval,
        block_length=max(strategy_config.signal_horizon * 2, 8),
    )
    updated_trade_log = build_trade_table(confirmed_predictions)
    updated_trade_summary = summarize_trade_table(updated_trade_log)
    updated_baseline_comparison = summarize_baselines(confirmed_predictions, interval, strategy_config)
    updated_fold_diagnostics = primary_result.fold_diagnostics.copy()
    for fold_id, fold_frame in confirmed_predictions.groupby("fold_id", sort=True):
        metrics = compute_metrics(fold_frame, interval)
        for metric_name, metric_value in metrics.items():
            if metric_name in updated_fold_diagnostics.columns:
                updated_fold_diagnostics.loc[updated_fold_diagnostics["fold_id"] == fold_id, metric_name] = metric_value
    guardrail_summary = (
        confirmed_predictions.assign(guardrail_reason=lambda frame: frame["guardrail_reason"].replace("", "accepted"))
        .groupby("guardrail_reason", dropna=False)
        .size()
        .rename("bars")
        .reset_index()
    )
    guardrail_summary["share"] = guardrail_summary["bars"] / max(len(confirmed_predictions), 1)
    return replace(
        primary_result,
        predictions=confirmed_predictions,
        fold_diagnostics=updated_fold_diagnostics,
        metrics=updated_metrics,
        benchmark_metrics=compute_metrics(build_buy_and_hold_frame(confirmed_predictions), interval),
        cost_stress=updated_cost_stress,
        bootstrap=updated_bootstrap,
        trade_log=updated_trade_log,
        trade_summary=updated_trade_summary,
        guardrail_summary=guardrail_summary.sort_values("bars", ascending=False).reset_index(drop=True),
        consensus_summary=consensus_summary,
        baseline_comparison=updated_baseline_comparison,
    )


def compare_consensus_gate_modes(
    primary_result: WalkForwardResult,
    diagnostics: ConsensusDiagnostics,
    *,
    interval: Interval,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    if diagnostics.timeline.empty:
        return pd.DataFrame()

    selected_mode = strategy_config.consensus_gate_mode if strategy_config.require_consensus_confirmation else "off"
    aligned_primary = align_consensus_predictions(primary_result.predictions, diagnostics.timeline)

    def _build_row(mode: str, label: str, result: WalkForwardResult) -> dict[str, object]:
        latest = result.predictions.iloc[-1]
        blocked_requested_share = 0.0
        if result.consensus_summary is not None and not result.consensus_summary.empty:
            blocked_requested_share = float(
                result.consensus_summary.loc[
                    result.consensus_summary["consensus_status"].isin(["weak_share", "flat_consensus", "opposed", "unavailable"]),
                    "share_of_requested",
                ].sum()
            )
        return {
            "mode": mode,
            "label": label,
            "selected": mode == selected_mode,
            "sharpe": float(result.metrics.get("sharpe", 0.0)),
            "annualized_return": float(result.metrics.get("annualized_return", 0.0)),
            "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            "trades": float(result.metrics.get("trades", 0.0)),
            "trade_win_rate": float(result.metrics.get("trade_win_rate", result.metrics.get("win_rate", 0.0))),
            "expectancy": float(result.metrics.get("expectancy", 0.0)),
            "exposure": float(result.metrics.get("exposure", 0.0)),
            "confidence_coverage": float(result.metrics.get("confidence_coverage", 0.0)),
            "latest_position": int(latest.get("signal_position", 0)),
            "latest_candidate": int(latest.get("candidate_action", 0)),
            "latest_guardrail": str(latest.get("guardrail_reason", "") or "accepted"),
            "latest_consensus_status": str(latest.get("consensus_status", "not_applied")),
            "latest_consensus_share": float(latest.get("consensus_effective_share", np.nan)),
            "blocked_requested_share": blocked_requested_share,
        }

    mode_rows = [
        _build_row(
            "off",
            "No Consensus",
            replace(primary_result, predictions=aligned_primary),
        )
    ]
    for mode, label in (("hard", "Hard Gate"), ("entry_only", "Entry-Only Gate")):
        mode_config = replace(strategy_config, require_consensus_confirmation=True, consensus_gate_mode=mode)
        mode_result = apply_consensus_confirmation(
            primary_result,
            diagnostics,
            interval=interval,
            strategy_config=mode_config,
        )
        mode_rows.append(_build_row(mode, label, mode_result))

    comparison = pd.DataFrame(mode_rows)
    order = {"off": 0, "hard": 1, "entry_only": 2}
    return comparison.sort_values(
        by="mode",
        key=lambda values: values.map(lambda item: order.get(str(item), len(order))),
        kind="stable",
    ).reset_index(drop=True)


def run_consensus_diagnostics(
    *,
    symbol: str,
    interval: Interval,
    limit: int,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    auto_adjust_windows: bool = True,
    seed_values: tuple[int, ...] | None = None,
    state_counts: tuple[int, ...] | None = None,
) -> ConsensusDiagnostics:
    seeds = seed_values or _default_seed_values(model_config.random_state)
    states = state_counts or _default_state_counts(model_config.n_states)
    cache: dict[tuple[str, Interval, int, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame, object, bool]] = {}
    member_rows: list[dict[str, object]] = []
    member_predictions: dict[str, pd.DataFrame] = {}

    for n_states in states:
        for seed in seeds:
            member_model_config = replace(model_config, n_states=n_states, random_state=seed)
            result, fetched, was_adjusted = _run_member(
                symbol=symbol,
                interval=interval,
                limit=limit,
                feature_columns=feature_columns,
                model_config=member_model_config,
                strategy_config=strategy_config,
                auto_adjust_windows=auto_adjust_windows,
                cache=cache,
            )
            member_id = f"states{n_states}_seed{seed}"
            member_rows.append(
                {
                    "member_id": member_id,
                    "n_states": n_states,
                    "random_state": seed,
                    "resolved_symbol": fetched.resolved_symbol,
                    "walk_adjusted": was_adjusted,
                    "sharpe": result.metrics["sharpe"],
                    "annualized_return": result.metrics["annualized_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "trades": result.metrics["trades"],
                    "trade_win_rate": result.metrics.get("trade_win_rate", 0.0),
                    "expectancy": result.metrics.get("expectancy", 0.0),
                    "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                    "latest_signal_position": int(result.predictions["signal_position"].iloc[-1]),
                    "latest_candidate_action": int(result.predictions["candidate_action"].iloc[-1]),
                    "latest_posterior": float(result.predictions["max_posterior"].iloc[-1]),
                    "converged_ratio": result.converged_ratio,
                }
            )
            member_predictions[member_id] = result.predictions.loc[:, ["timestamp", "close", "signal_position", "candidate_action"]].copy()

    members = pd.DataFrame(member_rows).sort_values(["n_states", "random_state"]).reset_index(drop=True)
    timeline = build_consensus_timeline(member_predictions)
    summary = summarize_consensus(members, timeline)
    return ConsensusDiagnostics(members=members, timeline=timeline, summary=summary)
