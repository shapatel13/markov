from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from markov_regime.config import Interval, StrategyConfig, SweepConfig, bars_per_year


def derive_state_actions(
    validate_frame: pd.DataFrame,
    n_states: int,
    config: StrategyConfig,
) -> pd.DataFrame:
    edge_column = f"forward_return_{config.signal_horizon}"
    rows: list[dict[str, float | int | str]] = []
    for canonical_state in range(n_states):
        subset = validate_frame.loc[validate_frame["canonical_state"] == canonical_state]
        sample_count = len(subset)
        validation_edge = float(subset[edge_column].mean()) if sample_count else 0.0
        avg_confidence = float(subset["max_posterior"].mean()) if sample_count else 0.0
        if sample_count < config.min_validation_samples:
            action = 0
            label = "insufficient_support"
        elif validation_edge > config.min_validation_edge:
            action = 1
            label = "risk_on"
        elif config.allow_short and validation_edge < -config.min_validation_edge:
            action = -1
            label = "risk_off"
        else:
            action = 0
            label = "flat"
        rows.append(
            {
                "canonical_state": canonical_state,
                "action": action,
                "label": label,
                "validation_edge": validation_edge,
                "samples": sample_count,
                "avg_confidence": avg_confidence,
            }
        )
    return pd.DataFrame(rows)


def _state_action_maps(state_actions: pd.DataFrame) -> tuple[dict[int, int], dict[int, float]]:
    action_map = {int(row.canonical_state): int(row.action) for row in state_actions.itertuples()}
    edge_map = {int(row.canonical_state): float(row.validation_edge) for row in state_actions.itertuples()}
    return action_map, edge_map


def _candidate_signal(
    row: pd.Series,
    action_map: dict[int, int],
    edge_map: dict[int, float],
    config: StrategyConfig,
) -> tuple[int, str]:
    state = int(row["canonical_state"])
    action = action_map.get(state, 0)
    validation_edge = edge_map.get(state, 0.0)

    if action == 0:
        return 0, "no_directional_edge"
    if float(row["max_posterior"]) < config.posterior_threshold:
        return 0, "posterior_below_threshold"
    if float(row["confidence_gap"]) < config.confidence_gap:
        return 0, "top_two_states_too_close"
    if abs(validation_edge) < config.min_validation_edge:
        return 0, "validation_edge_too_small"
    return action, ""


def apply_trading_rules(
    test_frame: pd.DataFrame,
    state_actions: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    action_map, edge_map = _state_action_maps(state_actions)
    rows: list[dict[str, object]] = []
    current_position = 0
    bars_held = 0
    cooldown_remaining = 0
    pending_action: int | None = None
    pending_confirmations = 0

    for row in test_frame.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        candidate_action, guardrail_reason = _candidate_signal(row_series, action_map, edge_map, config)
        effective_reason = guardrail_reason

        if candidate_action == current_position:
            desired_position = current_position
            pending_action = None
            pending_confirmations = 0
        elif candidate_action == 0:
            pending_action = None
            pending_confirmations = 0
            if current_position != 0 and bars_held >= config.min_hold_bars:
                desired_position = 0
            else:
                desired_position = current_position
                if current_position != 0:
                    effective_reason = effective_reason or "min_hold_active"
        else:
            if current_position != 0 and bars_held < config.min_hold_bars:
                desired_position = current_position
                effective_reason = effective_reason or "min_hold_active"
            elif cooldown_remaining > 0 and current_position == 0:
                desired_position = 0
                effective_reason = effective_reason or "cooldown_active"
            else:
                if pending_action != candidate_action:
                    pending_action = candidate_action
                    pending_confirmations = 1
                else:
                    pending_confirmations += 1

                if pending_confirmations >= config.required_confirmations:
                    desired_position = candidate_action
                else:
                    desired_position = current_position
                    effective_reason = effective_reason or "waiting_for_confirmations"

        previous_position = current_position
        position_changed = desired_position != previous_position
        current_position = desired_position

        if current_position != 0:
            bars_held = 1 if position_changed else bars_held + 1
        elif previous_position == 0:
            bars_held = 0
        else:
            bars_held = 0

        if position_changed and previous_position != 0 and current_position == 0:
            cooldown_remaining = config.cooldown_bars
        elif cooldown_remaining > 0 and current_position == 0:
            cooldown_remaining -= 1

        row_dict = row._asdict()
        row_dict["candidate_action"] = candidate_action
        row_dict["signal_position"] = current_position
        row_dict["bars_held"] = bars_held
        row_dict["cooldown_remaining"] = cooldown_remaining
        row_dict["guardrail_reason"] = effective_reason
        rows.append(row_dict)

    signal_frame = pd.DataFrame(rows)
    signal_frame["turnover"] = signal_frame["signal_position"].diff().abs().fillna(abs(signal_frame["signal_position"].iloc[0]))
    signal_frame["gross_strategy_return"] = signal_frame["signal_position"].shift(1).fillna(0.0) * signal_frame["bar_return"]
    signal_frame["execution_cost_bps"] = estimate_execution_cost_bps(signal_frame, config)
    signal_frame["transaction_cost"] = signal_frame["turnover"] * (signal_frame["execution_cost_bps"] / 10_000.0)
    signal_frame["net_strategy_return"] = signal_frame["gross_strategy_return"] - signal_frame["transaction_cost"]
    signal_frame["asset_wealth"] = (1.0 + signal_frame["bar_return"]).cumprod()
    signal_frame["strategy_wealth"] = (1.0 + signal_frame["net_strategy_return"]).cumprod()
    return signal_frame


def estimate_execution_cost_bps(
    signal_frame: pd.DataFrame,
    config: StrategyConfig,
    extra_bps: float = 0.0,
) -> pd.Series:
    if "range_ratio" in signal_frame.columns:
        range_ratio = signal_frame["range_ratio"].fillna(0.0)
    else:
        range_ratio = ((signal_frame["high"] - signal_frame["low"]) / signal_frame["close"].replace(0.0, np.nan)).fillna(0.0)

    volume = signal_frame.get("volume", pd.Series(config.volume_reference, index=signal_frame.index)).replace(0.0, np.nan)
    liquidity_penalty = np.sqrt(config.volume_reference / volume.fillna(config.volume_reference))
    liquidity_penalty = liquidity_penalty.replace([np.inf, -np.inf], 1.0).clip(lower=0.25, upper=4.0)
    range_component = (range_ratio * 10_000.0 * config.range_impact_weight).clip(lower=0.0)

    base_cost = config.cost_bps + config.spread_bps + config.slippage_bps + extra_bps
    return (base_cost + range_component + config.impact_bps * liquidity_penalty).astype(float)


def compute_metrics(signal_frame: pd.DataFrame, interval: Interval) -> dict[str, float]:
    returns = signal_frame["net_strategy_return"].fillna(0.0)
    if returns.empty:
        raise ValueError("Cannot compute metrics for an empty signal frame.")

    annualization = bars_per_year(interval)
    cumulative_wealth = (1.0 + returns).cumprod()
    total_return = float(cumulative_wealth.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (annualization / max(len(returns), 1)) - 1.0) if total_return > -1.0 else -1.0
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(annualization))
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(annualization)) if returns.std(ddof=0) > 0 else 0.0
    downside = returns.where(returns < 0.0, 0.0)
    sortino = float(returns.mean() / downside.std(ddof=0) * np.sqrt(annualization)) if downside.std(ddof=0) > 0 else 0.0
    drawdown = cumulative_wealth / cumulative_wealth.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
    active_mask = signal_frame["signal_position"].shift(1).fillna(0.0) != 0
    active_returns = returns.loc[active_mask]
    win_rate = float((active_returns > 0.0).mean()) if len(active_returns) else 0.0
    exposure = float(active_mask.mean())
    turnover = float(signal_frame["turnover"].sum())
    trades = int(((signal_frame["signal_position"] != signal_frame["signal_position"].shift(1)) & (signal_frame["signal_position"] != 0)).sum())
    confidence_coverage = float((signal_frame["candidate_action"] != 0).mean())

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "exposure": exposure,
        "turnover": turnover,
        "trades": float(trades),
        "confidence_coverage": confidence_coverage,
    }


def stress_test_transaction_costs(
    signal_frame: pd.DataFrame,
    cost_grid: tuple[float, ...],
    interval: Interval,
    config: StrategyConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    gross_return = signal_frame["gross_strategy_return"].fillna(0.0)
    turnover = signal_frame["turnover"].fillna(0.0)
    for cost_bps in cost_grid:
        stressed = signal_frame.copy()
        stressed["execution_cost_bps"] = estimate_execution_cost_bps(stressed, config, extra_bps=cost_bps)
        stressed["transaction_cost"] = turnover * (stressed["execution_cost_bps"] / 10_000.0)
        stressed["net_strategy_return"] = gross_return - stressed["transaction_cost"]
        stressed["strategy_wealth"] = (1.0 + stressed["net_strategy_return"]).cumprod()
        metrics = compute_metrics(stressed, interval)
        metrics["cost_bps"] = float(cost_bps)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def attach_state_action_columns(signal_frame: pd.DataFrame, state_actions: pd.DataFrame, n_states: int) -> pd.DataFrame:
    enriched = signal_frame.copy()
    action_lookup = state_actions.set_index("canonical_state")
    for state in range(n_states):
        action = int(action_lookup.loc[state, "action"]) if state in action_lookup.index else 0
        edge = float(action_lookup.loc[state, "validation_edge"]) if state in action_lookup.index else 0.0
        enriched[f"state_action_{state}"] = action
        enriched[f"validation_edge_{state}"] = edge
    return enriched


def state_actions_from_signal_frame(signal_frame: pd.DataFrame, n_states: int) -> pd.DataFrame:
    first_row = signal_frame.iloc[0]
    rows: list[dict[str, float | int | str]] = []
    for state in range(n_states):
        action = int(first_row.get(f"state_action_{state}", 0))
        validation_edge = float(first_row.get(f"validation_edge_{state}", 0.0))
        label = "risk_on" if action > 0 else "risk_off" if action < 0 else "flat"
        rows.append(
            {
                "canonical_state": state,
                "action": action,
                "label": label,
                "validation_edge": validation_edge,
                "samples": 0,
                "avg_confidence": 0.0,
            }
        )
    return pd.DataFrame(rows)


def replay_strategy(
    predictions: pd.DataFrame,
    n_states: int,
    config: StrategyConfig,
    interval: Interval,
) -> tuple[pd.DataFrame, dict[str, float]]:
    replay_frames: list[pd.DataFrame] = []
    for _, fold_frame in predictions.groupby("fold_id", sort=True):
        state_actions = state_actions_from_signal_frame(fold_frame, n_states)
        replayed = apply_trading_rules(fold_frame, state_actions, config)
        replayed = attach_state_action_columns(replayed, state_actions, n_states)
        replay_frames.append(replayed)
    combined = pd.concat(replay_frames).sort_values("timestamp").reset_index(drop=True)
    return combined, compute_metrics(combined, interval)


def parameter_sweep(
    predictions: pd.DataFrame,
    n_states: int,
    base_config: StrategyConfig,
    sweep_config: SweepConfig,
    interval: Interval,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for posterior_threshold, min_hold_bars, cooldown_bars, required_confirmations in product(
        sweep_config.posterior_thresholds,
        sweep_config.min_hold_bars,
        sweep_config.cooldown_bars,
        sweep_config.required_confirmations,
    ):
        config = replace(
            base_config,
            posterior_threshold=posterior_threshold,
            min_hold_bars=min_hold_bars,
            cooldown_bars=cooldown_bars,
            required_confirmations=required_confirmations,
        )
        _, metrics = replay_strategy(predictions, n_states, config, interval)
        rows.append(
            {
                "posterior_threshold": posterior_threshold,
                "min_hold_bars": min_hold_bars,
                "cooldown_bars": cooldown_bars,
                "required_confirmations": required_confirmations,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def build_buy_and_hold_frame(frame: pd.DataFrame) -> pd.DataFrame:
    benchmark = frame.copy()
    benchmark["candidate_action"] = 1
    benchmark["signal_position"] = 1
    benchmark["turnover"] = 0.0
    benchmark["execution_cost_bps"] = 0.0
    benchmark["transaction_cost"] = 0.0
    benchmark["gross_strategy_return"] = benchmark["bar_return"]
    benchmark["net_strategy_return"] = benchmark["bar_return"]
    benchmark["asset_wealth"] = (1.0 + benchmark["bar_return"]).cumprod()
    benchmark["strategy_wealth"] = benchmark["asset_wealth"]
    return benchmark
