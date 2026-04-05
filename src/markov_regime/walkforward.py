from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from markov_regime.bootstrap import block_bootstrap_confidence_intervals
from markov_regime.baselines import summarize_baselines
from markov_regime.config import Interval, ModelConfig, StrategyConfig, WalkForwardConfig
from markov_regime.model import (
    align_state_mapping,
    annotate_posteriors,
    apply_state_mapping,
    blend_reference_summary,
    fit_hmm,
    information_criteria,
    initial_state_mapping,
    summarize_states,
)
from markov_regime.strategy import (
    apply_trading_rules,
    attach_state_action_columns,
    build_buy_and_hold_frame,
    build_trade_table,
    compute_metrics,
    derive_state_actions,
    stress_test_transaction_costs,
    summarize_trade_table,
)


@dataclass(frozen=True)
class FoldWindow:
    fold_id: int
    train_start: int
    train_end: int
    validate_start: int
    validate_end: int
    test_start: int
    test_end: int


@dataclass
class WalkForwardResult:
    n_states: int
    predictions: pd.DataFrame
    fold_diagnostics: pd.DataFrame
    state_stability: pd.DataFrame
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    cost_stress: pd.DataFrame
    bootstrap: pd.DataFrame
    forward_returns: pd.DataFrame
    guardrail_summary: pd.DataFrame
    converged_ratio: float
    confirmation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    consensus_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)


def suggest_walk_forward_config(
    total_rows: int,
    requested: WalkForwardConfig,
    min_train_bars: int = 160,
    min_validate_bars: int = 48,
    min_test_bars: int = 48,
) -> tuple[WalkForwardConfig, bool]:
    required = (
        requested.train_bars
        + requested.purge_bars
        + requested.validate_bars
        + requested.embargo_bars
        + requested.test_bars
    )
    if total_rows >= required:
        return requested, False

    minimum_required = min_train_bars + requested.purge_bars + min_validate_bars + requested.embargo_bars + min_test_bars
    if total_rows < minimum_required:
        raise ValueError(
            f"Need at least {minimum_required} rows even for a reduced walk-forward run, received {total_rows}."
        )

    scalable_required = requested.train_bars + requested.validate_bars + requested.test_bars
    available_scalable = total_rows - requested.purge_bars - requested.embargo_bars
    train_ratio = requested.train_bars / scalable_required
    validate_ratio = requested.validate_bars / scalable_required
    proposed_train = max(min_train_bars, int(available_scalable * train_ratio))
    proposed_validate = max(min_validate_bars, int(available_scalable * validate_ratio))
    proposed_test = max(min_test_bars, available_scalable - proposed_train - proposed_validate)

    overflow = proposed_train + proposed_validate + proposed_test - total_rows
    if overflow > 0:
        reducible_train = max(0, proposed_train - min_train_bars)
        train_cut = min(reducible_train, overflow)
        proposed_train -= train_cut
        overflow -= train_cut

    if overflow > 0:
        reducible_validate = max(0, proposed_validate - min_validate_bars)
        validate_cut = min(reducible_validate, overflow)
        proposed_validate -= validate_cut
        overflow -= validate_cut

    if overflow > 0:
        proposed_test = max(min_test_bars, proposed_test - overflow)

    stride = min(requested.refit_stride_bars, proposed_test)
    adjusted = WalkForwardConfig(
        train_bars=proposed_train,
        purge_bars=requested.purge_bars,
        validate_bars=proposed_validate,
        embargo_bars=requested.embargo_bars,
        test_bars=proposed_test,
        refit_stride_bars=max(1, stride),
    )
    return adjusted, True


def generate_walk_forward_windows(total_rows: int, config: WalkForwardConfig) -> list[FoldWindow]:
    required = config.train_bars + config.purge_bars + config.validate_bars + config.embargo_bars + config.test_bars
    if total_rows < required:
        raise ValueError(
            f"Need at least {required} rows for walk-forward evaluation, received {total_rows}."
        )

    windows: list[FoldWindow] = []
    start = 0
    fold_id = 0
    while start + required <= total_rows:
        train_end = start + config.train_bars
        validate_start = train_end + config.purge_bars
        validate_end = validate_start + config.validate_bars
        test_start = validate_end + config.embargo_bars
        test_end = test_start + config.test_bars
        windows.append(
            FoldWindow(
                fold_id=fold_id,
                train_start=start,
                train_end=train_end,
                validate_start=validate_start,
                validate_end=validate_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        start += config.refit_stride_bars
        fold_id += 1
    return windows


def _alignment_for_initial_mapping(mapping: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_state": list(mapping.values()),
            "raw_state": list(mapping.keys()),
            "alignment_distance": 0.0,
        }
    )


def _summarize_forward_returns(predictions: pd.DataFrame) -> pd.DataFrame:
    horizons = (1, 3, 6, 12, 24, 72)
    rows: list[dict[str, float | int]] = []
    for canonical_state, subset in predictions.groupby("canonical_state", sort=True):
        for horizon in horizons:
            column = f"forward_return_{horizon}"
            valid = subset[column].dropna()
            rows.append(
                {
                    "canonical_state": int(canonical_state),
                    "horizon_bars": horizon,
                    "samples": int(valid.count()),
                    "mean_return": float(valid.mean()) if not valid.empty else 0.0,
                    "median_return": float(valid.median()) if not valid.empty else 0.0,
                    "hit_rate": float((valid > 0.0).mean()) if not valid.empty else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _guardrail_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        predictions.assign(guardrail_reason=lambda frame: frame["guardrail_reason"].replace("", "accepted"))
        .groupby("guardrail_reason")
        .size()
        .rename("bars")
        .reset_index()
    )
    counts["share"] = counts["bars"] / counts["bars"].sum()
    return counts.sort_values("bars", ascending=False).reset_index(drop=True)


def _state_stability_table(state_rows: pd.DataFrame) -> pd.DataFrame:
    stability_rows: list[dict[str, float | int]] = []
    for canonical_state, subset in state_rows.groupby("canonical_state", sort=True):
        sign_series = np.sign(subset["validation_edge"]).replace({np.nan: 0.0})
        sign_flips = int((sign_series.diff().fillna(0.0) != 0.0).sum())
        sign_flip_rate = float(sign_flips / max(len(subset) - 1, 1))
        mean_return_std = float(subset["mean_return"].std(ddof=0) if len(subset) > 1 else 0.0)
        alignment_distance = float(subset["alignment_distance"].mean())
        occupancy_std = float(subset["frequency"].std(ddof=0) if len(subset) > 1 else 0.0)
        stability_penalty = min(1.0, (alignment_distance + sign_flip_rate + mean_return_std * 100.0 + occupancy_std * 10.0) / 4.0)
        stability_rows.append(
            {
                "canonical_state": int(canonical_state),
                "folds_seen": int(len(subset)),
                "mean_return_mean": float(subset["mean_return"].mean()),
                "mean_return_std": mean_return_std,
                "persistence_mean": float(subset["persistence"].mean()),
                "occupancy_mean": float(subset["frequency"].mean()),
                "occupancy_std": occupancy_std,
                "validation_edge_mean": float(subset["validation_edge"].mean()),
                "sign_flip_rate": sign_flip_rate,
                "alignment_distance_mean": alignment_distance,
                "stability_score": 1.0 - stability_penalty,
            }
        )
    return pd.DataFrame(stability_rows).sort_values("canonical_state").reset_index(drop=True)


def run_walk_forward(
    feature_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    interval: Interval,
    model_config: ModelConfig,
    walk_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
) -> WalkForwardResult:
    windows = generate_walk_forward_windows(len(feature_frame), walk_config)
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int | bool]] = []
    state_rows: list[pd.DataFrame] = []
    reference_summary: pd.DataFrame | None = None

    for window in windows:
        train_frame = feature_frame.iloc[window.train_start:window.train_end].reset_index(drop=True)
        validate_frame = feature_frame.iloc[window.validate_start:window.validate_end].reset_index(drop=True)
        test_frame = feature_frame.iloc[window.test_start:window.test_end].reset_index(drop=True)

        fitted = fit_hmm(train_frame, feature_columns, model_config)
        train_annotated = annotate_posteriors(train_frame, fitted, feature_columns)
        validate_annotated = annotate_posteriors(validate_frame, fitted, feature_columns)
        test_annotated = annotate_posteriors(test_frame, fitted, feature_columns)

        train_summary = summarize_states(train_annotated, edge_horizon=strategy_config.signal_horizon)
        if reference_summary is None:
            mapping = initial_state_mapping(train_summary)
            alignment = _alignment_for_initial_mapping(mapping)
        else:
            mapping, alignment = align_state_mapping(reference_summary, train_summary)

        train_aligned = apply_state_mapping(train_annotated, mapping)
        validate_aligned = apply_state_mapping(validate_annotated, mapping)
        test_aligned = apply_state_mapping(test_annotated, mapping)

        canonical_train_summary = summarize_states(
            train_aligned,
            state_column="canonical_state",
            edge_horizon=strategy_config.signal_horizon,
        ).rename(columns={"state": "canonical_state", "validation_edge": "train_edge"})
        reference_summary = blend_reference_summary(reference_summary, canonical_train_summary)

        state_actions = derive_state_actions(validate_aligned, model_config.n_states, strategy_config)
        signal_frame = apply_trading_rules(test_aligned, state_actions, strategy_config)
        signal_frame = attach_state_action_columns(signal_frame, state_actions, model_config.n_states)
        signal_frame["fold_id"] = window.fold_id
        signal_frame["train_start_time"] = train_frame["timestamp"].iloc[0]
        signal_frame["train_end_time"] = train_frame["timestamp"].iloc[-1]
        signal_frame["validate_start_time"] = validate_frame["timestamp"].iloc[0]
        signal_frame["validate_end_time"] = validate_frame["timestamp"].iloc[-1]
        signal_frame["test_start_time"] = test_frame["timestamp"].iloc[0]
        signal_frame["test_end_time"] = test_frame["timestamp"].iloc[-1]
        signal_frame["oos_segment"] = "blind_test"
        signal_frame["is_blind_oos"] = True
        prediction_frames.append(signal_frame)

        fold_metrics = compute_metrics(signal_frame, interval)
        log_likelihood, aic, bic = information_criteria(fitted, train_frame, feature_columns)
        fold_rows.append(
            {
                "fold_id": window.fold_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "validate_start": window.validate_start,
                "validate_end": window.validate_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "train_start_time": str(pd.to_datetime(train_frame["timestamp"].iloc[0])),
                "train_end_time": str(pd.to_datetime(train_frame["timestamp"].iloc[-1])),
                "validate_start_time": str(pd.to_datetime(validate_frame["timestamp"].iloc[0])),
                "validate_end_time": str(pd.to_datetime(validate_frame["timestamp"].iloc[-1])),
                "test_start_time": str(pd.to_datetime(test_frame["timestamp"].iloc[0])),
                "test_end_time": str(pd.to_datetime(test_frame["timestamp"].iloc[-1])),
                "purge_bars": walk_config.purge_bars,
                "embargo_bars": walk_config.embargo_bars,
                "converged": fitted.converged,
                "optimizer_warning_count": len(fitted.fit_messages),
                "optimizer_warning_text": " | ".join(fitted.fit_messages[:2]),
                "log_likelihood": log_likelihood,
                "aic": aic,
                "bic": bic,
                **fold_metrics,
            }
        )

        state_detail = canonical_train_summary.merge(
            state_actions.loc[:, ["canonical_state", "action", "validation_edge"]],
            on="canonical_state",
            how="left",
        ).merge(alignment, on="canonical_state", how="left")
        state_detail["fold_id"] = window.fold_id
        state_rows.append(state_detail)

    predictions = pd.concat(prediction_frames).sort_values("timestamp").reset_index(drop=True)
    diagnostics = pd.DataFrame(fold_rows)
    state_detail_frame = pd.concat(state_rows).reset_index(drop=True)
    stability = _state_stability_table(state_detail_frame)
    metrics = compute_metrics(predictions, interval)
    benchmark_metrics = compute_metrics(build_buy_and_hold_frame(predictions), interval)
    cost_stress = stress_test_transaction_costs(predictions, strategy_config.cost_grid, interval, strategy_config)
    bootstrap = block_bootstrap_confidence_intervals(
        predictions["net_strategy_return"],
        interval=interval,
        block_length=max(strategy_config.signal_horizon * 2, 8),
    )
    forward_returns = _summarize_forward_returns(predictions)
    guardrail_summary = _guardrail_summary(predictions)
    converged_ratio = float(diagnostics["converged"].mean()) if not diagnostics.empty else 0.0
    trade_log = build_trade_table(predictions)
    trade_summary = summarize_trade_table(trade_log)
    baseline_comparison = summarize_baselines(predictions, interval, strategy_config)

    return WalkForwardResult(
        n_states=model_config.n_states,
        predictions=predictions,
        fold_diagnostics=diagnostics,
        state_stability=stability,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        cost_stress=cost_stress,
        bootstrap=bootstrap,
        forward_returns=forward_returns,
        guardrail_summary=guardrail_summary,
        converged_ratio=converged_ratio,
        trade_log=trade_log,
        trade_summary=trade_summary,
        baseline_comparison=baseline_comparison,
    )


def compare_state_counts(
    feature_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    interval: Interval,
    model_config: ModelConfig,
    walk_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
    state_range: range = range(5, 10),
) -> tuple[pd.DataFrame, dict[int, WalkForwardResult]]:
    results: dict[int, WalkForwardResult] = {}

    for n_states in state_range:
        current_model_config = replace(model_config, n_states=n_states)
        result = run_walk_forward(
            feature_frame=feature_frame,
            feature_columns=feature_columns,
            interval=interval,
            model_config=current_model_config,
            walk_config=walk_config,
            strategy_config=strategy_config,
        )
        results[n_states] = result

    return summarize_state_count_results(results), results


def summarize_state_count_results(results: dict[int, WalkForwardResult]) -> pd.DataFrame:
    comparison_rows: list[dict[str, float | int]] = []
    for n_states, result in sorted(results.items()):
        sharpe_ci = result.bootstrap.loc[result.bootstrap["metric"] == "sharpe"].iloc[0]
        comparison_rows.append(
            {
                "n_states": n_states,
                "sharpe": result.metrics["sharpe"],
                "annualized_return": result.metrics["annualized_return"],
                "max_drawdown": result.metrics["max_drawdown"],
                "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                "mean_bic": float(result.fold_diagnostics["bic"].mean()),
                "mean_aic": float(result.fold_diagnostics["aic"].mean()),
                "confidence_coverage": result.metrics["confidence_coverage"],
                "converged_ratio": result.converged_ratio,
                "bootstrap_sharpe_lower": float(sharpe_ci["lower"]),
                "bootstrap_sharpe_upper": float(sharpe_ci["upper"]),
            }
        )
    return pd.DataFrame(comparison_rows).sort_values("n_states").reset_index(drop=True)
