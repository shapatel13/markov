from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from markov_regime.config import DataConfig, Interval, ModelConfig, StrategyConfig, default_walk_forward_config
from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.data import DataFetchResult, fetch_price_data
from markov_regime.features import build_feature_frame
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
