from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from markov_regime.config import Interval, StrategyConfig, SweepConfig, bars_per_year


def _horizon_weights(horizons: tuple[int, ...]) -> dict[int, float]:
    if not horizons:
        return {}
    raw_weights = np.array([1.0 / np.sqrt(max(horizon, 1)) for horizon in horizons], dtype=float)
    normalized = raw_weights / raw_weights.sum()
    return {horizon: float(weight) for horizon, weight in zip(horizons, normalized, strict=True)}


def _confidence_interval(mean: float, std: float, sample_count: int, reliability: float) -> tuple[float, float]:
    if sample_count <= 1 or std <= 0.0:
        return mean * reliability, mean * reliability
    stderr = std / np.sqrt(sample_count)
    margin = 1.96 * stderr * reliability
    center = mean * reliability
    return center - margin, center + margin


def derive_state_actions(
    validate_frame: pd.DataFrame,
    n_states: int,
    config: StrategyConfig,
) -> pd.DataFrame:
    weights = _horizon_weights(config.scoring_horizons)
    rows: list[dict[str, float | int | str]] = []

    for canonical_state in range(n_states):
        subset = validate_frame.loc[validate_frame["canonical_state"] == canonical_state]
        sample_count = len(subset)
        avg_confidence = float(subset["max_posterior"].mean()) if sample_count else 0.0

        weighted_score = 0.0
        weighted_lower = 0.0
        weighted_upper = 0.0
        available_weight = 0.0
        positive_horizons = 0
        negative_horizons = 0
        row: dict[str, float | int | str] = {
            "canonical_state": canonical_state,
            "samples": sample_count,
            "avg_confidence": avg_confidence,
        }

        for horizon in config.scoring_horizons:
            column = f"forward_return_{horizon}"
            valid = subset[column].dropna() if column in subset.columns else pd.Series(dtype=float)
            horizon_samples = int(valid.count())
            mean_edge = float(valid.mean()) if horizon_samples else 0.0
            edge_std = float(valid.std(ddof=0)) if horizon_samples > 1 else 0.0
            reliability = float(horizon_samples / (horizon_samples + config.validation_shrinkage)) if horizon_samples else 0.0
            shrunk_edge = mean_edge * reliability
            edge_lower, edge_upper = _confidence_interval(mean_edge, edge_std, horizon_samples, reliability)
            weight = weights.get(horizon, 0.0)

            if horizon_samples:
                weighted_score += weight * shrunk_edge
                weighted_lower += weight * edge_lower
                weighted_upper += weight * edge_upper
                available_weight += weight

            if edge_lower > config.min_validation_edge:
                positive_horizons += 1
            if edge_upper < -config.min_validation_edge:
                negative_horizons += 1

            row[f"samples_{horizon}"] = horizon_samples
            row[f"mean_edge_{horizon}"] = mean_edge
            row[f"edge_std_{horizon}"] = edge_std
            row[f"reliability_{horizon}"] = reliability
            row[f"shrunk_edge_{horizon}"] = shrunk_edge
            row[f"edge_lower_{horizon}"] = edge_lower
            row[f"edge_upper_{horizon}"] = edge_upper

        if available_weight > 0.0:
            score = weighted_score / available_weight
            score_lower = weighted_lower / available_weight
            score_upper = weighted_upper / available_weight
        else:
            score = 0.0
            score_lower = 0.0
            score_upper = 0.0

        consistent_horizons = max(positive_horizons, negative_horizons)
        if sample_count < config.min_validation_samples:
            action = 0
            label = "insufficient_support"
        elif positive_horizons >= config.min_consistent_horizons and score_lower > config.min_validation_edge:
            action = 1
            label = "risk_on"
        elif config.allow_short and negative_horizons >= config.min_consistent_horizons and score_upper < -config.min_validation_edge:
            action = -1
            label = "risk_off"
        elif consistent_horizons < config.min_consistent_horizons:
            action = 0
            label = "inconsistent_across_horizons"
        elif abs(score) <= config.min_validation_edge:
            action = 0
            label = "validation_edge_too_small"
        else:
            action = 0
            label = "flat"

        row.update(
            {
                "action": action,
                "label": label,
                "validation_edge": score,
                "score_lower": score_lower,
                "score_upper": score_upper,
                "positive_horizons": positive_horizons,
                "negative_horizons": negative_horizons,
                "consistent_horizons": consistent_horizons,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _state_action_maps(
    state_actions: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, float], dict[int, float], dict[int, float]]:
    action_map = {int(row.canonical_state): int(row.action) for row in state_actions.itertuples()}
    edge_map = {int(row.canonical_state): float(row.validation_edge) for row in state_actions.itertuples()}
    lower_map = {int(row.canonical_state): float(getattr(row, "score_lower", row.validation_edge)) for row in state_actions.itertuples()}
    upper_map = {int(row.canonical_state): float(getattr(row, "score_upper", row.validation_edge)) for row in state_actions.itertuples()}
    return action_map, edge_map, lower_map, upper_map


def _candidate_signal(
    row: pd.Series,
    action_map: dict[int, int],
    edge_map: dict[int, float],
    lower_map: dict[int, float],
    upper_map: dict[int, float],
    config: StrategyConfig,
) -> tuple[int, str]:
    state = int(row["canonical_state"])
    action = action_map.get(state, 0)
    validation_edge = edge_map.get(state, 0.0)
    score_lower = lower_map.get(state, validation_edge)
    score_upper = upper_map.get(state, validation_edge)

    if action == 0:
        return 0, "no_directional_edge"
    if float(row["max_posterior"]) < config.posterior_threshold:
        return 0, "posterior_below_threshold"
    if float(row["confidence_gap"]) < config.confidence_gap:
        return 0, "top_two_states_too_close"
    if action > 0 and score_lower <= config.min_validation_edge:
        return 0, "validation_edge_too_small"
    if action < 0 and score_upper >= -config.min_validation_edge:
        return 0, "validation_edge_too_small"
    return action, ""


def apply_trading_rules(
    test_frame: pd.DataFrame,
    state_actions: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    action_map, edge_map, lower_map, upper_map = _state_action_maps(state_actions)
    rows: list[dict[str, object]] = []
    current_position = 0
    bars_held = 0
    cooldown_remaining = 0
    pending_action: int | None = None
    pending_confirmations = 0

    for row in test_frame.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        candidate_action, guardrail_reason = _candidate_signal(
            row_series,
            action_map,
            edge_map,
            lower_map,
            upper_map,
            config,
        )
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


TRADE_LOG_COLUMNS = [
    "trade_id",
    "direction",
    "entry_signal_time",
    "first_active_time",
    "exit_signal_time",
    "status",
    "bars_held",
    "gross_return",
    "net_return",
    "mfe",
    "mae",
    "avg_posterior",
    "entry_price",
    "exit_price",
    "exit_reason",
]


def build_trade_table(signal_frame: pd.DataFrame) -> pd.DataFrame:
    if signal_frame.empty:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    frame = signal_frame.reset_index(drop=True).copy()
    frame["holding_position"] = frame["signal_position"].shift(1).fillna(0.0).astype(int)
    active_mask = frame["holding_position"] != 0
    if not active_mask.any():
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    start_flags = active_mask & (frame["holding_position"] != frame["holding_position"].shift(1).fillna(0.0).astype(int))
    frame["trade_id"] = start_flags.cumsum().where(active_mask, 0).astype(int)

    rows: list[dict[str, Any]] = []
    active_frame = frame.loc[active_mask].copy()
    for trade_id, trade_slice in active_frame.groupby("trade_id", sort=True):
        direction = int(trade_slice["holding_position"].iloc[0])
        first_active_index = int(trade_slice.index[0])
        last_active_index = int(trade_slice.index[-1])
        entry_signal_index = max(first_active_index - 1, 0)
        entry_signal_time = pd.to_datetime(frame.loc[entry_signal_index, "timestamp"])
        first_active_time = pd.to_datetime(trade_slice["timestamp"].iloc[0])
        current_signal = int(frame.loc[last_active_index, "signal_position"])
        closed = current_signal != direction
        exit_signal_time = pd.to_datetime(frame.loc[last_active_index, "timestamp"]) if closed else pd.NaT

        gross_returns = trade_slice["gross_strategy_return"].fillna(0.0)
        net_returns = trade_slice["net_strategy_return"].fillna(0.0)
        trade_wealth = (1.0 + net_returns).cumprod()
        gross_return = float((1.0 + gross_returns).prod() - 1.0)
        net_return = float(trade_wealth.iloc[-1] - 1.0)
        mfe = float(trade_wealth.max() - 1.0)
        mae = float(trade_wealth.min() - 1.0)

        if closed:
            if current_signal == 0:
                exit_reason = "flat_exit"
            elif np.sign(current_signal) != np.sign(direction):
                exit_reason = "reversal"
            else:
                exit_reason = str(frame.loc[last_active_index].get("guardrail_reason", "position_changed") or "position_changed")
        else:
            exit_reason = "open"

        rows.append(
            {
                "trade_id": int(trade_id),
                "direction": direction,
                "entry_signal_time": entry_signal_time,
                "first_active_time": first_active_time,
                "exit_signal_time": exit_signal_time,
                "status": "closed" if closed else "open",
                "bars_held": int(len(trade_slice)),
                "gross_return": gross_return,
                "net_return": net_return,
                "mfe": mfe,
                "mae": mae,
                "avg_posterior": float(trade_slice["max_posterior"].mean()) if "max_posterior" in trade_slice.columns else 0.0,
                "entry_price": float(frame.loc[entry_signal_index, "close"]) if "close" in frame.columns else np.nan,
                "exit_price": float(frame.loc[last_active_index, "close"]) if closed and "close" in frame.columns else np.nan,
                "exit_reason": exit_reason,
            }
        )

    return pd.DataFrame(rows, columns=TRADE_LOG_COLUMNS)


def _empty_trade_metrics() -> dict[str, float]:
    return {
        "trade_win_rate": 0.0,
        "avg_trade_return": 0.0,
        "median_trade_return": 0.0,
        "avg_winner_return": 0.0,
        "avg_loser_return": 0.0,
        "expectancy": 0.0,
        "profit_factor": 0.0,
        "avg_bars_held": 0.0,
        "median_bars_held": 0.0,
        "avg_mfe": 0.0,
        "avg_mae": 0.0,
        "closed_trade_count": 0.0,
        "open_trade_count": 0.0,
    }


def compute_trade_metrics(trade_table: pd.DataFrame) -> dict[str, float]:
    if trade_table.empty:
        return _empty_trade_metrics()

    closed_trades = trade_table.loc[trade_table["status"] == "closed"].copy()
    winners = closed_trades.loc[closed_trades["net_return"] > 0.0, "net_return"]
    losers = closed_trades.loc[closed_trades["net_return"] < 0.0, "net_return"]
    gross_profit = float(winners.sum())
    gross_loss = float(abs(losers.sum()))
    closed_count = float(len(closed_trades))
    open_count = float((trade_table["status"] == "open").sum())

    if closed_count:
        trade_win_rate = float((closed_trades["net_return"] > 0.0).mean())
        avg_trade_return = float(closed_trades["net_return"].mean())
        median_trade_return = float(closed_trades["net_return"].median())
    else:
        trade_win_rate = 0.0
        avg_trade_return = 0.0
        median_trade_return = 0.0

    avg_winner_return = float(winners.mean()) if not winners.empty else 0.0
    avg_loser_return = float(losers.mean()) if not losers.empty else 0.0
    expectancy = float(avg_trade_return)
    if gross_profit > 0.0 and gross_loss == 0.0:
        profit_factor = 25.0
    elif gross_loss > 0.0:
        profit_factor = float(min(gross_profit / gross_loss, 25.0))
    else:
        profit_factor = 0.0

    return {
        "trade_win_rate": trade_win_rate,
        "avg_trade_return": avg_trade_return,
        "median_trade_return": median_trade_return,
        "avg_winner_return": avg_winner_return,
        "avg_loser_return": avg_loser_return,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "avg_bars_held": float(trade_table["bars_held"].mean()) if not trade_table.empty else 0.0,
        "median_bars_held": float(trade_table["bars_held"].median()) if not trade_table.empty else 0.0,
        "avg_mfe": float(trade_table["mfe"].mean()) if not trade_table.empty else 0.0,
        "avg_mae": float(trade_table["mae"].mean()) if not trade_table.empty else 0.0,
        "closed_trade_count": closed_count,
        "open_trade_count": open_count,
    }


def summarize_trade_table(trade_table: pd.DataFrame) -> pd.DataFrame:
    metrics = compute_trade_metrics(trade_table)
    rows = [
        {"metric": "Total trades", "value": float(len(trade_table))},
        {"metric": "Closed trades", "value": metrics["closed_trade_count"]},
        {"metric": "Open trades", "value": metrics["open_trade_count"]},
        {"metric": "Trade win rate", "value": metrics["trade_win_rate"]},
        {"metric": "Average trade return", "value": metrics["avg_trade_return"]},
        {"metric": "Median trade return", "value": metrics["median_trade_return"]},
        {"metric": "Average winner", "value": metrics["avg_winner_return"]},
        {"metric": "Average loser", "value": metrics["avg_loser_return"]},
        {"metric": "Expectancy", "value": metrics["expectancy"]},
        {"metric": "Profit factor", "value": metrics["profit_factor"]},
        {"metric": "Average bars held", "value": metrics["avg_bars_held"]},
        {"metric": "Median bars held", "value": metrics["median_bars_held"]},
        {"metric": "Average MFE", "value": metrics["avg_mfe"]},
        {"metric": "Average MAE", "value": metrics["avg_mae"]},
    ]
    return pd.DataFrame(rows)


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
    bar_win_rate = float((active_returns > 0.0).mean()) if len(active_returns) else 0.0
    exposure = float(active_mask.mean())
    turnover = float(signal_frame["turnover"].sum())
    confidence_coverage = float((signal_frame["candidate_action"] != 0).mean())

    trade_table = build_trade_table(signal_frame)
    trade_metrics = compute_trade_metrics(trade_table)
    trades = float(len(trade_table))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": trade_metrics["trade_win_rate"],
        "bar_win_rate": bar_win_rate,
        "exposure": exposure,
        "turnover": turnover,
        "trades": trades,
        "confidence_coverage": confidence_coverage,
        **trade_metrics,
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
    action_lookup = state_actions.set_index("canonical_state") if not state_actions.empty else pd.DataFrame()
    for state in range(n_states):
        if state in getattr(action_lookup, "index", []):
            state_row = action_lookup.loc[state]
            action = int(state_row.get("action", 0))
            edge = float(state_row.get("validation_edge", 0.0))
            score_lower = float(state_row.get("score_lower", edge))
            score_upper = float(state_row.get("score_upper", edge))
            consistent_horizons = int(state_row.get("consistent_horizons", 0))
        else:
            action = 0
            edge = 0.0
            score_lower = 0.0
            score_upper = 0.0
            consistent_horizons = 0
        enriched[f"state_action_{state}"] = action
        enriched[f"validation_edge_{state}"] = edge
        enriched[f"score_lower_{state}"] = score_lower
        enriched[f"score_upper_{state}"] = score_upper
        enriched[f"consistent_horizons_{state}"] = consistent_horizons
    return enriched


def state_actions_from_signal_frame(signal_frame: pd.DataFrame, n_states: int) -> pd.DataFrame:
    first_row = signal_frame.iloc[0]
    rows: list[dict[str, float | int | str]] = []
    for state in range(n_states):
        action = int(first_row.get(f"state_action_{state}", 0))
        validation_edge = float(first_row.get(f"validation_edge_{state}", 0.0))
        score_lower = float(first_row.get(f"score_lower_{state}", validation_edge))
        score_upper = float(first_row.get(f"score_upper_{state}", validation_edge))
        consistent_horizons = int(first_row.get(f"consistent_horizons_{state}", 0))
        label = "risk_on" if action > 0 else "risk_off" if action < 0 else "flat"
        rows.append(
            {
                "canonical_state": state,
                "action": action,
                "label": label,
                "validation_edge": validation_edge,
                "score_lower": score_lower,
                "score_upper": score_upper,
                "consistent_horizons": consistent_horizons,
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
    if config.require_daily_confirmation and "confirmation_effective_direction" in combined.columns:
        from markov_regime.confirmation import apply_confirmation_overlay

        combined, _ = apply_confirmation_overlay(combined, config)
    if config.require_consensus_confirmation and "consensus_position" in combined.columns:
        from markov_regime.consensus import apply_consensus_overlay

        combined, _ = apply_consensus_overlay(combined, config)
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
    benchmark["guardrail_reason"] = "accepted"
    benchmark["turnover"] = 0.0
    benchmark["execution_cost_bps"] = 0.0
    benchmark["transaction_cost"] = 0.0
    benchmark["gross_strategy_return"] = benchmark["bar_return"]
    benchmark["net_strategy_return"] = benchmark["bar_return"]
    benchmark["asset_wealth"] = (1.0 + benchmark["bar_return"]).cumprod()
    benchmark["strategy_wealth"] = benchmark["asset_wealth"]
    return benchmark
