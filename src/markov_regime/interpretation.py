from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from markov_regime.config import Interval, StrategyConfig, WalkForwardConfig

POSITION_LABELS: dict[int, str] = {1: "Long", 0: "Flat", -1: "Short"}

CONTROL_HELP: dict[str, str] = {
    "live_engine_mode": "Controls what the top live recommendation follows. `auto` uses the promoted engine, `baseline` forces the strongest simple reference, `hmm_ensemble` uses the seed/state-count consensus overlay, and `hmm_research` forces the raw HMM even when it is not promoted.",
    "interval": "Primary research timeframe. `4hour` is the main trading lane, `1day` is slower confirmation, and `1hour` is the noisier baseline. Defaults now lean toward a 12-month train and 3-month blind test style on the higher timeframes.",
    "provider": "Historical bar source. `auto` keeps Financial Modeling Prep as the primary source, then uses Coinbase deep-history backfill only when FMP's intraday crypto sample is too short, with Yahoo only as a last-resort fallback.",
    "feature_pack": "Chooses what market features the HMM sees. Richer packs can improve regime separation, but they can also be more fragile.",
    "limit": "How many bars to keep after fetching and resampling. More history usually makes walk-forward conclusions less fragile, and `auto` provider mode may reach beyond FMP's recent intraday cap when needed.",
    "states": "Number of HMM regimes. Too few can blur distinct behavior; too many can over-segment noise.",
    "train_bars": "Bars used to fit each rolling HMM. Larger windows are steadier but slower to adapt.",
    "purge_bars": "Bars removed between train and validate windows to reduce leakage around the split.",
    "validate_bars": "Bars used to map latent states into trade actions before the true test window begins.",
    "embargo_bars": "Extra gap between validation and test windows to further reduce look-ahead contamination.",
    "test_bars": "Bars used for true out-of-sample scoring in each fold.",
    "refit_stride": "How often the model refits. Shorter stride adapts faster; longer stride gives smoother, slower changes.",
    "posterior_threshold": "Minimum state probability needed before the strategy trusts the regime assignment enough to consider trading.",
    "min_hold_bars": "Minimum bars to keep a position after entry. Higher values reduce churn but can delay exits.",
    "cooldown_bars": "Bars to stay flat after a position closes. Higher values reduce whipsaws but may miss quick re-entries.",
    "required_confirmations": "Consecutive qualifying bars required before a new position can open. Higher values are stricter.",
    "confidence_gap": "Minimum gap between the top two posterior state probabilities. Higher values force cleaner separation between competing regimes.",
    "allow_short": "When enabled, validated bearish regimes may map to short trades instead of only flattening exposure. Leave this off unless you explicitly want two-sided signals.",
    "require_daily_confirmation": "When enabled on `4hour`, the strategy only executes exposure when the latest daily lane agrees with the 4H direction.",
    "require_consensus_confirmation": "When enabled, the strategy only executes exposure when nearby seeds and state counts broadly agree with the current direction.",
    "consensus_min_share": "Minimum agreement share required across the consensus panel before a trade is allowed. Higher values are stricter.",
    "consensus_gate_mode": "Controls whether weak consensus fully blocks exposure or only blocks fresh entries while allowing existing holds to continue. `entry_only` is the recommended default because it kept more of the useful signal in testing than `hard` mode.",
    "cost_bps": "Trading fee assumption in basis points. Default is now 10 bps (0.10%) before spread and slippage to avoid unrealistically cheap crypto backtests.",
    "spread_bps": "Bid/ask spread assumption in basis points. Wider spreads punish turnover-heavy variants.",
    "slippage_bps": "Execution slippage assumption in basis points. Higher values test whether the edge survives imperfect fills.",
    "impact_bps": "Liquidity impact penalty in basis points. Higher values stress strategies that need more urgent execution.",
    "scoring_horizons": "Forward-return horizons used to score each latent state. Multiple horizons reduce the odds that one lucky window defines the trade label.",
    "validation_shrinkage": "Amount of shrinkage applied to state validation edges. Higher values pull weak, small-sample edges back toward zero.",
    "min_consistent_horizons": "How many forward horizons must agree before a state becomes tradable. Higher values prefer consistency over coverage.",
}


def position_label(value: int | float) -> str:
    return POSITION_LABELS.get(int(value), "Flat")


def _bars_to_days(interval: Interval, bars: int) -> float:
    if interval == "1hour":
        return bars / 24.0
    if interval == "4hour":
        return bars / 6.0
    return float(bars)


def _duration_label(interval: Interval, bars: int) -> str:
    days = _bars_to_days(interval, bars)
    if days >= 365:
        return f"{days / 365.0:.1f} years"
    if days >= 60:
        return f"{days / 30.0:.1f} months"
    if days >= 2:
        return f"{days:.0f} days"
    return f"{bars} bars"


def _sample_band(interval: Interval, available_rows: int) -> str:
    thresholds = {
        "1hour": (2500, 6000),
        "4hour": (1800, 3200),
        "1day": (500, 900),
    }
    thin, deeper = thresholds[interval]
    if available_rows < thin:
        return "thin"
    if available_rows < deeper:
        return "usable"
    return "deep"


def _bootstrap_interval(bootstrap: pd.DataFrame, metric: str = "sharpe") -> tuple[float | None, float | None]:
    if bootstrap.empty or "metric" not in bootstrap.columns:
        return None, None
    row = bootstrap.loc[bootstrap["metric"] == metric]
    if row.empty:
        return None, None
    return float(row.iloc[0]["lower"]), float(row.iloc[0]["upper"])


def _median_robustness_sharpe(robustness: pd.DataFrame) -> float | None:
    if robustness.empty or "status" not in robustness.columns or "sharpe" not in robustness.columns:
        return None
    ok_rows = robustness.loc[robustness["status"] == "ok", "sharpe"]
    if ok_rows.empty:
        return None
    return float(ok_rows.median())


def _best_baseline_sharpe(baseline_comparison: pd.DataFrame) -> float | None:
    if baseline_comparison.empty or "sharpe" not in baseline_comparison.columns:
        return None
    return float(baseline_comparison["sharpe"].max())


def first_sentence(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if ". " in cleaned:
        return cleaned.split(". ", 1)[0].rstrip(".") + "."
    return cleaned


def _status_label(status: str) -> str:
    mapping = {
        "confirmed": "Confirmed",
        "neutral": "Neutral",
        "blocked": "Blocked",
        "unavailable": "Unavailable",
        "no_primary_signal": "No Primary Signal",
        "weak_share": "Weak Share",
        "flat_consensus": "Consensus Flat",
        "opposed": "Consensus Opposes",
        "no_primary_signal": "No Primary Signal",
    }
    return mapping.get(status, status.replace("_", " ").title())


def describe_guardrail(reason: str, *, current_position: int, candidate_action: int) -> str:
    reason_key = reason or "accepted"
    if reason_key == "accepted":
        if candidate_action != 0 and current_position == candidate_action:
            return "The latest bar clears the guardrails and agrees with the currently held position."
        if candidate_action != 0 and current_position == 0:
            return "The latest bar clears the guardrails and supports a fresh entry."
        return "No active guardrail block is being applied on the latest bar."
    if reason_key == "no_directional_edge":
        if current_position != 0:
            return "The latest bar does not show enough validated directional edge for a fresh trade, even though an older position may still be held."
        return "The current regime does not have enough validated directional edge, so the strategy prefers no trade."
    if reason_key == "posterior_below_threshold":
        return "The HMM is not confident enough in the current state assignment, so the strategy prefers no trade."
    if reason_key == "top_two_states_too_close":
        return "The top two regime probabilities are too close together, so the state assignment is treated as ambiguous."
    if reason_key == "validation_edge_too_small":
        return "The mapped state edge is too small to justify a trade after validation."
    if reason_key == "waiting_for_confirmations":
        return "The latest bar points in a direction, but the strategy is still waiting for enough consecutive confirmations before entering."
    if reason_key == "min_hold_active":
        return "The strategy is respecting the minimum hold rule, so it is not allowed to exit or reverse immediately."
    if reason_key == "cooldown_active":
        return "The strategy is in cooldown after a recent exit, so it is intentionally waiting before re-entering."
    if reason_key == "consensus_weak_share":
        return "Nearby seeds and state counts do not agree strongly enough, so the strategy prefers no trade."
    if reason_key == "consensus_flat":
        return "The nearby-model consensus panel prefers no trade, so the strategy stands down."
    if reason_key == "consensus_opposes":
        return "The nearby-model consensus panel leans the other way, so the trade is blocked."
    if reason_key == "consensus_unavailable":
        return "Consensus diagnostics were not available, so the trade is blocked."
    if reason_key == "consensus_hold_weak_share":
        return "Consensus is too weak for a fresh entry, but the strategy is allowing an existing position to continue."
    if reason_key == "consensus_hold_flat":
        return "The consensus panel prefers no new trade, but the strategy is preserving an existing hold instead of force-flattening."
    if reason_key == "consensus_hold_opposed":
        return "Consensus now leans the other way, but this softer mode is preserving the existing hold instead of force-flattening."
    if reason_key == "consensus_hold_unavailable":
        return "Consensus diagnostics are unavailable, so new entries are blocked while existing exposure is allowed to continue."
    return f"Latest guardrail status: {reason_key.replace('_', ' ')}."


def build_execution_plan(
    *,
    latest_row: Mapping[str, Any],
    interval: Interval,
    live_price: float | None = None,
) -> dict[str, str]:
    current_position = int(latest_row.get("signal_position", 0))
    candidate_action = int(latest_row.get("candidate_action", 0))
    guardrail_reason = str(latest_row.get("guardrail_reason", "") or "accepted")
    latest_close = float(latest_row.get("close", 0.0))
    latest_high = float(latest_row.get("high", latest_close))
    latest_low = float(latest_row.get("low", latest_close))
    signal_time = pd.to_datetime(latest_row.get("timestamp")) if latest_row.get("timestamp") is not None else None
    reference_price = live_price if live_price is not None else latest_close
    bar_label = {"1hour": "1H", "4hour": "4H", "1day": "1D"}[interval]

    if current_position == 0 and candidate_action == -1 and guardrail_reason == "accepted":
        return {
            "action": "Enter Short",
            "severity": "success",
            "summary": f"The latest completed {bar_label} bar qualifies for a fresh short entry.",
            "entry_guide": (
                f"Aggressive entry: around {reference_price:,.2f}. "
                f"Conservative trigger: only if price is still below the last completed {bar_label} low at {latest_low:,.2f}. "
                f"Invalidate back above {latest_high:,.2f}."
            ),
            "timing_note": f"This strategy acts on completed {bar_label} bars, so treat intrabar moves before {signal_time} as noise rather than confirmed signals." if signal_time is not None else f"This strategy acts on completed {bar_label} bars, not intrabar spikes.",
        }
    if current_position == 0 and candidate_action == 1 and guardrail_reason == "accepted":
        return {
            "action": "Enter Long",
            "severity": "success",
            "summary": f"The latest completed {bar_label} bar qualifies for a fresh long entry.",
            "entry_guide": (
                f"Aggressive entry: around {reference_price:,.2f}. "
                f"Conservative trigger: only if price is still above the last completed {bar_label} high at {latest_high:,.2f}. "
                f"Invalidate below {latest_low:,.2f}."
            ),
            "timing_note": f"This strategy acts on completed {bar_label} bars, so treat intrabar spikes before {signal_time} as noise rather than confirmed signals." if signal_time is not None else f"This strategy acts on completed {bar_label} bars, not intrabar spikes.",
        }
    if current_position == 1 and candidate_action == 1 and guardrail_reason == "accepted":
        return {
            "action": "Hold Long",
            "severity": "success",
            "summary": f"The strategy is already long and the latest completed {bar_label} bar still supports staying long.",
            "entry_guide": "No fresh entry is needed because the model is already carrying exposure.",
            "timing_note": f"Only treat a new completed {bar_label} bar as actionable. This is not a tick-by-tick execution model.",
        }
    if current_position == -1 and candidate_action == -1 and guardrail_reason == "accepted":
        return {
            "action": "Hold Short",
            "severity": "success",
            "summary": f"The strategy is already short and the latest completed {bar_label} bar still supports staying short.",
            "entry_guide": "No fresh entry is needed because the model is already carrying short exposure.",
            "timing_note": f"Only treat a new completed {bar_label} bar as actionable. This is not a tick-by-tick execution model.",
        }
    if current_position == 1 and candidate_action == 0:
        return {
            "action": "Hold / No Add",
            "severity": "warning",
            "summary": f"The model is still carrying a long from earlier, but the latest completed {bar_label} bar does not justify a fresh add.",
            "entry_guide": "Do not add here. Wait for the next completed bar to restore an accepted long candidate before considering any new entry.",
            "timing_note": describe_guardrail(guardrail_reason, current_position=current_position, candidate_action=candidate_action),
        }
    if current_position == -1 and candidate_action == 0:
        return {
            "action": "Hold Short / No Add",
            "severity": "warning",
            "summary": f"The model is still carrying a short from earlier, but the latest completed {bar_label} bar does not justify a fresh add.",
            "entry_guide": "Do not add here. Wait for the next completed bar to restore an accepted short candidate before considering any new entry.",
            "timing_note": describe_guardrail(guardrail_reason, current_position=current_position, candidate_action=candidate_action),
        }
    if current_position == 0 and guardrail_reason == "waiting_for_confirmations":
        direction = "long" if candidate_action >= 0 else "short"
        reference = latest_high if direction == "long" else latest_low
        return {
            "action": "Wait",
            "severity": "warning",
            "summary": f"A directional setup is emerging, but the model still needs more consecutive {bar_label} confirmations before entering.",
            "entry_guide": (
                f"Wait for another completed {bar_label} bar that keeps the {direction} candidate alive. "
                f"A conservative reference is the last {bar_label} {'high' if direction == 'long' else 'low'} at {reference:,.2f}."
            ),
            "timing_note": "This is a confirmation delay, not a missed trade. The filter is trying to reduce one-bar noise.",
        }
    return {
        "action": "No Entry",
        "severity": "warning",
        "summary": f"No fresh trade is approved on the latest completed {bar_label} bar.",
        "entry_guide": "There is no valid live entry level right now because the model prefers staying flat over taking a marginal trade.",
        "timing_note": describe_guardrail(guardrail_reason, current_position=current_position, candidate_action=candidate_action),
    }


def build_trust_snapshot(
    *,
    metrics: Mapping[str, float],
    bootstrap: pd.DataFrame,
    state_stability: pd.DataFrame,
    robustness: pd.DataFrame,
    interval: Interval,
    available_rows: int,
    walk_adjusted: bool,
) -> dict[str, Any]:
    sharpe = float(metrics.get("sharpe", 0.0))
    trades = float(metrics.get("trades", 0.0))
    stability = float(state_stability["stability_score"].median()) if not state_stability.empty else 0.0
    sharpe_lower, sharpe_upper = _bootstrap_interval(bootstrap, "sharpe")
    robustness_median = _median_robustness_sharpe(robustness)
    sample_band = _sample_band(interval, available_rows)

    score = 0
    reasons: list[str] = []

    if sharpe > 0.0:
        score += 2
        reasons.append("positive backtest Sharpe")
    else:
        score -= 2
        reasons.append("negative backtest Sharpe")

    if sharpe_lower is not None and sharpe_lower > 0.0:
        score += 2
        reasons.append("bootstrap lower bound stays above zero")
    elif sharpe > 0.0:
        score -= 1
        reasons.append("bootstrap still crosses zero")

    if stability >= 0.6:
        score += 1
        reasons.append("state stability is decent")
    elif stability < 0.3:
        score -= 1
        reasons.append("state stability is fragile")

    if robustness_median is not None and robustness_median > 0.0:
        score += 1
        reasons.append("cross-asset robustness is positive")
    elif robustness_median is not None and robustness_median < 0.0:
        reasons.append("cross-asset robustness is weak")

    if trades >= 8:
        score += 1
        reasons.append("trade count is at least somewhat informative")
    elif trades < 5:
        reasons.append("trade count is very small")

    if sample_band == "deep" and not walk_adjusted:
        score += 1
        reasons.append("sample depth is healthier")
    elif sample_band == "thin" or walk_adjusted:
        score -= 1
        reasons.append("sample is thin or windows were auto-reduced")

    if score >= 5:
        verdict = "Robust"
        severity = "success"
    elif score >= 1:
        verdict = "Promising but fragile"
        severity = "warning"
    elif score >= -1:
        verdict = "Mixed"
        severity = "warning"
    else:
        verdict = "Weak"
        severity = "error"

    summary = (
        f"{verdict}: {'; '.join(reasons[:4])}. "
        "Treat annualized returns and headline Sharpe as provisional until bootstrap, stability, and robustness all agree."
    )
    return {
        "verdict": verdict,
        "severity": severity,
        "summary": summary,
        "sample_band": sample_band,
        "robustness_median_sharpe": robustness_median,
        "bootstrap_sharpe_lower": sharpe_lower,
        "bootstrap_sharpe_upper": sharpe_upper,
        "stability_score": stability,
    }


def build_metric_interpretation_rows(
    *,
    latest_row: Mapping[str, Any],
    metrics: Mapping[str, float],
    bootstrap: pd.DataFrame,
    state_stability: pd.DataFrame,
    robustness: pd.DataFrame,
    interval: Interval,
    available_rows: int,
    walk_adjusted: bool,
) -> pd.DataFrame:
    current_position = int(latest_row.get("signal_position", 0))
    candidate_action = int(latest_row.get("candidate_action", 0))
    current_state = int(latest_row.get("canonical_state", -1))
    guardrail_reason = str(latest_row.get("guardrail_reason", "") or "accepted")
    posterior = float(latest_row.get("max_posterior", 0.0))
    confidence_gap = float(latest_row.get("confidence_gap", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    annualized_return = float(metrics.get("annualized_return", 0.0))
    coverage = float(metrics.get("confidence_coverage", 0.0))
    bar_win_rate = float(metrics.get("bar_win_rate", 0.0))
    trade_win_rate = float(metrics.get("trade_win_rate", metrics.get("win_rate", 0.0)))
    expectancy = float(metrics.get("expectancy", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    trades = float(metrics.get("trades", 0.0))
    snapshot = build_trust_snapshot(
        metrics=metrics,
        bootstrap=bootstrap,
        state_stability=state_stability,
        robustness=robustness,
        interval=interval,
        available_rows=available_rows,
        walk_adjusted=walk_adjusted,
    )
    sharpe_lower = snapshot["bootstrap_sharpe_lower"]
    sharpe_upper = snapshot["bootstrap_sharpe_upper"]
    stability = snapshot["stability_score"]
    robustness_median = snapshot["robustness_median_sharpe"]
    sample_band = snapshot["sample_band"]
    confirmation_status = str(latest_row.get("confirmation_status", "") or "")
    confirmation_direction = int(latest_row.get("confirmation_effective_direction", 0)) if "confirmation_effective_direction" in latest_row else 0
    confirmation_interval = str(latest_row.get("confirmation_interval", "")) if "confirmation_interval" in latest_row else ""
    consensus_status = str(latest_row.get("consensus_status", "") or "")
    consensus_direction = int(latest_row.get("consensus_effective_direction", 0)) if "consensus_effective_direction" in latest_row else 0
    consensus_share = float(latest_row.get("consensus_effective_share", 0.0)) if "consensus_effective_share" in latest_row else 0.0

    if confirmation_status == "confirmed":
        confirmation_text = f"The {confirmation_interval} lane agrees with the current 4H direction."
    elif confirmation_status == "neutral":
        confirmation_text = f"The {confirmation_interval} lane is neutral. It is not adding extra confirmation, but it is also not vetoing the 4H trade."
    elif confirmation_status == "blocked":
        confirmation_text = f"The {confirmation_interval} lane points the other way, so it blocks the 4H trade."
    elif confirmation_status == "unavailable":
        confirmation_text = f"No usable {confirmation_interval} confirmation bar was available for this timestamp."
    elif confirmation_status:
        confirmation_text = f"The {confirmation_interval} confirmation filter is present but not active on this bar."
    else:
        confirmation_text = ""

    if consensus_status == "confirmed":
        consensus_text = "Nearby seeds and state counts agree strongly enough with the current direction."
    elif consensus_status == "weak_share":
        consensus_text = "Nearby models do not agree strongly enough, so the trade is being filtered out as fragile."
    elif consensus_status == "flat_consensus":
        consensus_text = "The consensus panel prefers no trade, so the strategy stands down."
    elif consensus_status == "opposed":
        consensus_text = "The consensus panel leans the other way, so the strategy blocks the trade."
    elif consensus_status == "unavailable":
        consensus_text = "Consensus diagnostics were not available for this bar."
    elif consensus_status:
        consensus_text = "The consensus filter is present but not active on this bar."
    else:
        consensus_text = ""

    if posterior >= 0.9:
        posterior_text = "Very high state confidence, but remember this is confidence in the state label, not proof of edge."
    elif posterior >= 0.75:
        posterior_text = "Solid state confidence. The model sees a reasonably clear regime assignment."
    elif posterior >= 0.6:
        posterior_text = "Moderate state confidence. The assignment is usable, but not especially clean."
    else:
        posterior_text = "Low state confidence. Regime assignment is weak and easier to flip on the next bar."

    if sharpe >= 2.0:
        sharpe_text = "Excellent on paper."
    elif sharpe >= 1.0:
        sharpe_text = "Good on paper."
    elif sharpe > 0.0:
        sharpe_text = "Positive, but not especially strong."
    elif sharpe > -1.0:
        sharpe_text = "Slightly negative. The strategy is not clearly adding value."
    else:
        sharpe_text = "Clearly negative. The current setup looks weak."
    if sharpe_lower is not None and sharpe_upper is not None:
        if sharpe_lower > 0.0:
            sharpe_text += f" Bootstrap support is better than average because the Sharpe interval stays above zero ({sharpe_lower:.2f} to {sharpe_upper:.2f})."
        else:
            sharpe_text += f" Bootstrap still crosses zero ({sharpe_lower:.2f} to {sharpe_upper:.2f}), so this reading is not yet trustworthy."

    annual_return_text = "Positive annualized return on this sample." if annualized_return > 0 else "Negative annualized return on this sample."
    if sample_band == "thin" or walk_adjusted:
        annual_return_text += " Because the sample is short or auto-adjusted, treat the annualized figure as illustrative rather than predictive."

    if stability >= 0.75:
        stability_text = "State alignment looks strong across retrains."
    elif stability >= 0.5:
        stability_text = "State stability is decent, but not ironclad."
    elif stability >= 0.3:
        stability_text = "State stability is fragile. Regime meaning may drift across refits."
    else:
        stability_text = "State stability is weak. The regime story is not very durable across retrains."

    if coverage < 0.15:
        coverage_text = "Very selective. The strategy mostly prefers no trade."
    elif coverage < 0.4:
        coverage_text = "Cautious. The strategy is willing to stay flat on many bars."
    elif coverage < 0.7:
        coverage_text = "Moderately active. The strategy finds tradable bars with some regularity."
    else:
        coverage_text = "Very active. This can help capture moves, but it also raises overtrading risk."

    if trades < 5:
        trades_text = "Too few trades to be very persuasive. A handful of winners can dominate the backtest."
    elif trades < 15:
        trades_text = "Still a fairly small trade count. Interpret performance carefully."
    elif trades < 40:
        trades_text = "Reasonable trade count for a first research pass."
    else:
        trades_text = "Healthy trade count, though execution assumptions still matter."

    if bar_win_rate >= 0.6:
        bar_win_rate_text = "A good share of active bars made money, but this can still overstate strategy quality if losses cluster into a few bad trades."
    elif bar_win_rate >= 0.5:
        bar_win_rate_text = "Slightly positive at the bar level. Useful, but less important than trade expectancy."
    else:
        bar_win_rate_text = "Weak at the bar level. Even before grouping into trades, the edge is not very clean."

    if trade_win_rate >= 0.6:
        trade_win_rate_text = "Strong trade hit rate on this sample, though it still needs enough trade count and a healthy payoff profile."
    elif trade_win_rate >= 0.45:
        trade_win_rate_text = "Middle-of-the-road trade hit rate. Profitability will depend on winners being larger than losers."
    else:
        trade_win_rate_text = "Low trade hit rate. The strategy needs unusually strong winner size to overcome this."

    if expectancy > 0.01:
        expectancy_text = "Positive average trade expectancy. Each closed trade has added meaningful value on average."
    elif expectancy > 0.0:
        expectancy_text = "Slightly positive trade expectancy. Encouraging, but still fragile."
    elif expectancy > -0.01:
        expectancy_text = "Near-flat expectancy. The strategy is not clearly earning enough per trade yet."
    else:
        expectancy_text = "Negative expectancy. The average closed trade is losing money."

    if profit_factor >= 2.0:
        profit_factor_text = "Strong payoff balance. Gross winners are comfortably outweighing gross losers."
    elif profit_factor >= 1.2:
        profit_factor_text = "Healthy enough payoff balance for a research candidate."
    elif profit_factor >= 1.0:
        profit_factor_text = "Barely above break-even. Small cost changes could erase the edge."
    elif profit_factor > 0.0:
        profit_factor_text = "Gross losers outweigh or nearly match gross winners."
    else:
        profit_factor_text = "No demonstrated payoff edge yet at the trade level."

    if robustness_median is None:
        robustness_text = "No usable robustness basket result was available."
        robustness_value = "n/a"
    else:
        robustness_value = f"{robustness_median:.2f}"
        if robustness_median > 0.5:
            robustness_text = "Cross-asset robustness looks constructive."
        elif robustness_median > 0.0:
            robustness_text = "Cross-asset robustness is mildly positive, but not overwhelming."
        elif robustness_median > -0.5:
            robustness_text = "Cross-asset robustness is mixed to weak."
        else:
            robustness_text = "Cross-asset robustness is poor. This often signals overfitting to the primary symbol."

    if sample_band == "deep" and not walk_adjusted:
        sample_text = "Sample depth is healthier for this timeframe and the requested windows fit without shrinkage."
        sample_value = "deep"
    elif sample_band == "usable" and not walk_adjusted:
        sample_text = "Sample depth is usable, but still not abundant."
        sample_value = "usable"
    else:
        sample_text = "Sample is thin or auto-adjusted. High Sharpe and annualized return can be badly inflated in this regime."
        sample_value = "thin / adjusted" if walk_adjusted else sample_band

    rows = [
        {
            "metric": "Research Verdict",
            "value": snapshot["verdict"],
            "interpretation": snapshot["summary"],
        },
        {
            "metric": "Current State",
            "value": str(current_state),
            "interpretation": "Latent state IDs are cluster labels, not direct market labels. Always interpret them through validated edge and stability, not by number alone.",
        },
        {
            "metric": "Held Position",
            "value": position_label(current_position),
            "interpretation": (
                "This is the executed position the strategy is currently carrying after all guardrails and confirmation filters. "
                + ("It already has market exposure." if current_position != 0 else "It is currently flat.")
            ),
        },
        {
            "metric": "Latest Candidate",
            "value": position_label(candidate_action),
            "interpretation": (
                "This is the action the newest bar would support before hold/cooldown mechanics are considered. "
                + ("A fresh trade is allowed." if guardrail_reason == "accepted" and candidate_action != 0 else "The latest bar is not currently strong enough for a fresh trade.")
            ),
        },
        {
            "metric": "Guardrail Status",
            "value": guardrail_reason.replace("_", " "),
            "interpretation": describe_guardrail(guardrail_reason, current_position=current_position, candidate_action=candidate_action),
        },
        {
            "metric": "Posterior",
            "value": f"{posterior:.2f}",
            "interpretation": posterior_text + f" The top-two state gap on the latest bar is {confidence_gap:.2f}.",
        },
        {
            "metric": "Sharpe",
            "value": f"{sharpe:.2f}",
            "interpretation": sharpe_text,
        },
        {
            "metric": "Annualized Return",
            "value": f"{annualized_return:.1%}",
            "interpretation": annual_return_text,
        },
        {
            "metric": "Bootstrap Sharpe CI",
            "value": "n/a" if sharpe_lower is None or sharpe_upper is None else f"{sharpe_lower:.2f} to {sharpe_upper:.2f}",
            "interpretation": "This is the block-bootstrap uncertainty range for Sharpe. If it crosses zero, the edge is still statistically fragile.",
        },
        {
            "metric": "State Stability",
            "value": f"{stability:.2f}",
            "interpretation": stability_text,
        },
        {
            "metric": "Confidence Coverage",
            "value": f"{coverage:.1%}",
            "interpretation": coverage_text,
        },
        {
            "metric": "Bar Win Rate",
            "value": f"{bar_win_rate:.1%}",
            "interpretation": bar_win_rate_text,
        },
        {
            "metric": "Trade Win Rate",
            "value": f"{trade_win_rate:.1%}",
            "interpretation": trade_win_rate_text,
        },
        {
            "metric": "Trades",
            "value": f"{trades:.0f}",
            "interpretation": trades_text,
        },
        {
            "metric": "Expectancy",
            "value": f"{expectancy:.2%}",
            "interpretation": expectancy_text,
        },
        {
            "metric": "Profit Factor",
            "value": f"{profit_factor:.2f}",
            "interpretation": profit_factor_text,
        },
        {
            "metric": "Cross-Asset Robustness",
            "value": robustness_value,
            "interpretation": robustness_text,
        },
        {
            "metric": "Sample Quality",
            "value": sample_value,
            "interpretation": sample_text,
        },
    ]
    if confirmation_status:
        rows.insert(
            5,
            {
                "metric": "Daily Confirmation",
                "value": f"{_status_label(confirmation_status)} ({position_label(confirmation_direction)})",
                "interpretation": confirmation_text,
            },
        )
    if consensus_status:
        insertion_index = 6 if confirmation_status else 5
        rows.insert(
            insertion_index,
            {
                "metric": "Consensus Filter",
                "value": f"{_status_label(consensus_status)} ({position_label(consensus_direction)}, {consensus_share:.0%})",
                "interpretation": consensus_text,
            },
        )
    return pd.DataFrame(rows)


def build_hmm_loss_breakdown(
    *,
    strategy_metrics: Mapping[str, float],
    ensemble_metrics: Mapping[str, float] | None,
    baseline_row: Mapping[str, Any] | None,
    promotion_gates: pd.DataFrame,
    nested_holdout: Mapping[str, Any] | None = None,
    robustness: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    strategy_sharpe = float(strategy_metrics.get("sharpe", 0.0))
    strategy_trades = float(strategy_metrics.get("trades", 0.0))
    robustness_median = _median_robustness_sharpe(robustness if robustness is not None else pd.DataFrame())
    sharpe_lower, sharpe_upper = _bootstrap_interval(bootstrap if bootstrap is not None else pd.DataFrame(), "sharpe")
    nested_status = str((nested_holdout or {}).get("status", "unavailable"))
    outer_holdout_sharpe = float((nested_holdout or {}).get("outer_holdout_sharpe", 0.0)) if nested_status == "ok" else None

    baseline_name = "unavailable"
    baseline_sharpe = None
    if baseline_row is not None:
        baseline_name = str(baseline_row.get("baseline", "baseline"))
        baseline_sharpe_raw = baseline_row.get("sharpe")
        baseline_sharpe = float(baseline_sharpe_raw) if baseline_sharpe_raw is not None else None

    ensemble_sharpe = None
    ensemble_trades = None
    if ensemble_metrics is not None:
        ensemble_sharpe = float(ensemble_metrics.get("sharpe", 0.0))
        ensemble_trades = float(ensemble_metrics.get("trades", 0.0))

    failed_gates = (
        promotion_gates.loc[promotion_gates["status"] == "fail", "gate"].tolist()
        if not promotion_gates.empty and {"status", "gate"}.issubset(promotion_gates.columns)
        else []
    )

    rows: list[dict[str, str]] = []

    if baseline_sharpe is None:
        rows.append(
            {
                "factor": "Baseline Bar",
                "status": "info",
                "detail": "No baseline comparison was available, so the HMM was judged without a simple reference bar.",
            }
        )
    elif strategy_sharpe > baseline_sharpe:
        rows.append(
            {
                "factor": "Baseline Bar",
                "status": "ok",
                "detail": f"The raw HMM beat `{baseline_name}` on Sharpe ({strategy_sharpe:.2f} vs {baseline_sharpe:.2f}), so baseline competition was not the blocker.",
            }
        )
    else:
        rows.append(
            {
                "factor": "Baseline Bar",
                "status": "fail",
                "detail": f"The strongest simple baseline `{baseline_name}` still beat the raw HMM on Sharpe ({baseline_sharpe:.2f} vs {strategy_sharpe:.2f}).",
            }
        )

    if ensemble_sharpe is None:
        rows.append(
            {
                "factor": "Ensemble Check",
                "status": "info",
                "detail": "The consensus-filtered HMM ensemble was not evaluated on this run.",
            }
        )
    elif baseline_sharpe is not None and ensemble_sharpe > baseline_sharpe and ensemble_sharpe > 0.0:
        rows.append(
            {
                "factor": "Ensemble Check",
                "status": "ok",
                "detail": f"The consensus-filtered HMM improved enough to clear the baseline bar ({ensemble_sharpe:.2f} vs {baseline_sharpe:.2f}).",
            }
        )
    elif ensemble_sharpe > strategy_sharpe:
        rows.append(
            {
                "factor": "Ensemble Check",
                "status": "warning",
                "detail": (
                    f"Consensus improved the raw HMM ({ensemble_sharpe:.2f} vs {strategy_sharpe:.2f})"
                    + (
                        f", but it still trailed `{baseline_name}` at {baseline_sharpe:.2f}."
                        if baseline_sharpe is not None
                        else "."
                    )
                ),
            }
        )
    else:
        ensemble_trade_text = f" with only {ensemble_trades:.0f} trades" if ensemble_trades is not None else ""
        rows.append(
            {
                "factor": "Ensemble Check",
                "status": "fail",
                "detail": f"The consensus-filtered HMM did not rescue the edge ({ensemble_sharpe:.2f} Sharpe{ensemble_trade_text}).",
            }
        )

    if outer_holdout_sharpe is None:
        rows.append(
            {
                "factor": "Outer Holdout",
                "status": "fail",
                "detail": "There was no clean nested outer holdout left after inner selection, so the sweep could not be blindly confirmed.",
            }
        )
    elif outer_holdout_sharpe > 0.0:
        rows.append(
            {
                "factor": "Outer Holdout",
                "status": "ok",
                "detail": f"The untouched outer holdout stayed positive at {outer_holdout_sharpe:.2f} Sharpe.",
            }
        )
    else:
        rows.append(
            {
                "factor": "Outer Holdout",
                "status": "fail",
                "detail": f"The untouched outer holdout did not confirm the edge ({outer_holdout_sharpe:.2f} Sharpe).",
            }
        )

    if sharpe_lower is None or sharpe_upper is None:
        rows.append(
            {
                "factor": "Bootstrap Support",
                "status": "info",
                "detail": "No Sharpe bootstrap interval was available for this run.",
            }
        )
    elif sharpe_lower > 0.0:
        rows.append(
            {
                "factor": "Bootstrap Support",
                "status": "ok",
                "detail": f"The bootstrap Sharpe interval stayed above zero ({sharpe_lower:.2f} to {sharpe_upper:.2f}).",
            }
        )
    else:
        rows.append(
            {
                "factor": "Bootstrap Support",
                "status": "fail",
                "detail": f"The bootstrap Sharpe interval still crossed zero ({sharpe_lower:.2f} to {sharpe_upper:.2f}).",
            }
        )

    if robustness_median is None:
        rows.append(
            {
                "factor": "Cross-Asset Robustness",
                "status": "info",
                "detail": "No successful robustness basket run was available.",
            }
        )
    elif robustness_median > 0.0:
        rows.append(
            {
                "factor": "Cross-Asset Robustness",
                "status": "ok",
                "detail": f"The median BTC/ETH/SOL robustness Sharpe stayed positive at {robustness_median:.2f}.",
            }
        )
    else:
        rows.append(
            {
                "factor": "Cross-Asset Robustness",
                "status": "fail",
                "detail": f"The median BTC/ETH/SOL robustness Sharpe was weak at {robustness_median:.2f}.",
            }
        )

    if strategy_trades >= 5:
        rows.append(
            {
                "factor": "Trade Count",
                "status": "ok",
                "detail": f"The HMM produced {strategy_trades:.0f} trades, which is enough to be more than a one-trade anecdote.",
            }
        )
    else:
        rows.append(
            {
                "factor": "Trade Count",
                "status": "fail",
                "detail": f"The HMM only produced {strategy_trades:.0f} trades, so a small number of outcomes can dominate the result.",
            }
        )

    if failed_gates:
        rows.append(
            {
                "factor": "Promotion Gates",
                "status": "fail",
                "detail": "Failed promotion gates: " + ", ".join(failed_gates[:5]) + ("." if len(failed_gates) <= 5 else ", ..."),
            }
        )
    else:
        rows.append(
            {
                "factor": "Promotion Gates",
                "status": "ok",
                "detail": "The run cleared every promotion gate.",
            }
        )

    return pd.DataFrame(rows)


def summarize_hmm_loss_breakdown(breakdown: pd.DataFrame) -> dict[str, str]:
    if breakdown.empty:
        return {
            "headline": "Why HMM Lost",
            "severity": "info",
            "summary": "No HMM loss breakdown was available for this run.",
        }

    failed = breakdown.loc[breakdown["status"] == "fail", "factor"].tolist() if "status" in breakdown.columns else []
    warnings = breakdown.loc[breakdown["status"] == "warning", "factor"].tolist() if "status" in breakdown.columns else []
    primary = failed[:3] if failed else warnings[:2]
    reason_text = ", ".join(primary) if primary else "no major blockers were recorded"

    if failed:
        severity = "warning"
        summary = f"The HMM lost the live seat mainly because of: {reason_text}."
    elif warnings:
        severity = "info"
        summary = f"The HMM is close, but still needs work around: {reason_text}."
    else:
        severity = "success"
        summary = "The HMM did not lose on the major quality checks tracked here."

    return {
        "headline": "Why HMM Lost",
        "severity": severity,
        "summary": summary,
    }


def build_promotion_gate_rows(
    *,
    metrics: Mapping[str, float],
    bootstrap: pd.DataFrame,
    state_stability: pd.DataFrame,
    robustness: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    interval: Interval,
    available_rows: int,
    walk_adjusted: bool,
    fold_count: int,
    nested_holdout: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    sharpe = float(metrics.get("sharpe", 0.0))
    trades = float(metrics.get("trades", 0.0))
    sharpe_lower, _ = _bootstrap_interval(bootstrap, "sharpe")
    stability = float(state_stability["stability_score"].median()) if not state_stability.empty else 0.0
    robustness_median = _median_robustness_sharpe(robustness)
    best_baseline_sharpe = _best_baseline_sharpe(baseline_comparison)
    sample_band = _sample_band(interval, available_rows)
    nested_status = str((nested_holdout or {}).get("status", "unavailable"))
    outer_holdout_sharpe = (
        float((nested_holdout or {}).get("outer_holdout_sharpe", 0.0))
        if nested_status == "ok"
        else None
    )

    rows = [
        {
            "gate": "Positive OOS Sharpe",
            "status": "pass" if sharpe > 0.0 else "fail",
            "detail": f"Current out-of-sample Sharpe is {sharpe:.2f}.",
        },
        {
            "gate": "Enough Walk-Forward Folds",
            "status": "pass" if fold_count >= 3 else "fail",
            "detail": f"Observed {fold_count} walk-forward folds. Fewer than 3 folds is usually too thin for holdout-aware promotion.",
        },
        {
            "gate": "Nested Holdout Available",
            "status": "pass" if nested_status == "ok" else "fail",
            "detail": (
                "Inner folds selected settings and a reserved outer fold remained available for blind confirmation."
                if nested_status == "ok"
                else "There were not enough folds to reserve a clean outer holdout after inner selection."
            ),
        },
        {
            "gate": "Nested Holdout Sharpe Positive",
            "status": "pass" if outer_holdout_sharpe is not None and outer_holdout_sharpe > 0.0 else "fail",
            "detail": (
                f"Outer holdout Sharpe is {outer_holdout_sharpe:.2f}."
                if outer_holdout_sharpe is not None
                else "Nested outer holdout Sharpe is unavailable because no clean outer holdout remained."
            ),
        },
        {
            "gate": "Bootstrap Lower Bound Above Zero",
            "status": "pass" if sharpe_lower is not None and sharpe_lower > 0.0 else "fail",
            "detail": (
                f"Sharpe lower confidence bound is {sharpe_lower:.2f}."
                if sharpe_lower is not None
                else "No Sharpe bootstrap interval was available."
            ),
        },
        {
            "gate": "Enough Closed Trades",
            "status": "pass" if trades >= 5 else "fail",
            "detail": f"Observed {trades:.0f} trades. Fewer than 5 trades is usually too thin to trust.",
        },
        {
            "gate": "State Stability",
            "status": "pass" if stability >= 0.5 else "fail",
            "detail": f"Median state stability is {stability:.2f}.",
        },
        {
            "gate": "Cross-Asset Robustness",
            "status": "pass" if robustness_median is not None and robustness_median > 0.0 else "fail",
            "detail": (
                f"Median robustness Sharpe is {robustness_median:.2f}."
                if robustness_median is not None
                else "No successful robustness basket run was available."
            ),
        },
        {
            "gate": "Beats Best Simple Baseline",
            "status": "pass" if best_baseline_sharpe is not None and sharpe > best_baseline_sharpe else "fail",
            "detail": (
                f"Best simple baseline Sharpe is {best_baseline_sharpe:.2f}."
                if best_baseline_sharpe is not None
                else "No baseline comparison was available."
            ),
        },
        {
            "gate": "No Auto-Reduced Windows",
            "status": "pass" if not walk_adjusted else "fail",
            "detail": "Requested windows ran as specified." if not walk_adjusted else "Windows had to be auto-reduced to fit the available sample.",
        },
        {
            "gate": "Sample Depth",
            "status": "pass" if sample_band != "thin" else "fail",
            "detail": f"Current sample depth is classified as `{sample_band}` for {interval}.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_promotion_gates(gates: pd.DataFrame) -> dict[str, str]:
    if gates.empty:
        return {
            "verdict": "Unavailable",
            "severity": "warning",
            "summary": "Promotion gates are unavailable for this run.",
        }
    passed = int((gates["status"] == "pass").sum())
    total = int(len(gates))
    if passed == total:
        verdict = "Eligible"
        severity = "success"
        summary = "This run clears all promotion gates. It is strong enough to consider for default-candidate review."
    elif passed >= max(total - 2, 1):
        verdict = "Near Miss"
        severity = "warning"
        summary = "This run clears most promotion gates, but it still has important weak spots before it should influence defaults."
    else:
        verdict = "Not Ready"
        severity = "error"
        summary = "This run does not clear enough promotion gates to justify changing defaults. Treat it as research only."
    return {
        "verdict": verdict,
        "severity": severity,
        "summary": summary + f" Passed {passed} of {total} gates.",
    }


def recommend_strategy_engine(
    *,
    strategy_metrics: Mapping[str, float],
    baseline_comparison: pd.DataFrame,
    promotion_summary: Mapping[str, str] | None = None,
) -> dict[str, str]:
    strategy_sharpe = float(strategy_metrics.get("sharpe", 0.0))
    promotion_verdict = str((promotion_summary or {}).get("verdict", "Unavailable"))

    if baseline_comparison is not None and not baseline_comparison.empty:
        best_baseline = baseline_comparison.sort_values("sharpe", ascending=False).iloc[0]
        baseline_name = str(best_baseline.get("baseline", "baseline"))
        baseline_sharpe = float(best_baseline.get("sharpe", 0.0))
    else:
        baseline_name = "unavailable"
        baseline_sharpe = float("-inf")

    if promotion_verdict == "Eligible" and strategy_sharpe > baseline_sharpe:
        return {
            "engine": "hmm",
            "severity": "success",
            "headline": "Use HMM, not baseline",
            "summary": (
                f"The current HMM run clears the promotion gates and beats the best simple baseline "
                f"({strategy_sharpe:.2f} vs {baseline_sharpe:.2f})."
            ),
            "best_baseline": baseline_name,
        }

    if promotion_verdict != "Eligible" and baseline_name != "unavailable" and baseline_sharpe > 0.0:
        return {
            "engine": "baseline",
            "severity": "info",
            "headline": "Use baseline, not HMM",
            "summary": (
                f"The HMM is not promoted yet, so the simpler live reference is `{baseline_name}` "
                f"with Sharpe {baseline_sharpe:.2f}."
            ),
            "best_baseline": baseline_name,
        }

    if baseline_name != "unavailable" and baseline_sharpe >= max(strategy_sharpe, 0.0):
        return {
            "engine": "baseline",
            "severity": "info",
            "headline": "Use baseline, not HMM",
            "summary": (
                f"The best simple baseline `{baseline_name}` is at least as strong "
                f"on Sharpe ({baseline_sharpe:.2f} vs {strategy_sharpe:.2f})."
            ),
            "best_baseline": baseline_name,
        }

    if promotion_verdict != "Eligible":
        return {
            "engine": "cash",
            "severity": "warning",
            "headline": "Use no active deployment yet",
            "summary": (
                "The HMM is still research-only, and the simple baselines are not compelling enough to justify "
                "promoting a live default either."
            ),
            "best_baseline": baseline_name,
        }

    return {
        "engine": "research",
        "severity": "warning",
        "headline": "Research further before promotion",
        "summary": "The current run is promising, but it still needs a cleaner edge over the baseline bar before promotion.",
        "best_baseline": baseline_name,
    }


def resolve_live_engine_mode(
    *,
    requested_mode: str,
    engine_recommendation: Mapping[str, str],
    best_baseline: str | None = None,
    consensus_available: bool = False,
) -> dict[str, str]:
    requested = str(requested_mode or "auto")
    recommended_engine = str(engine_recommendation.get("engine", "cash"))
    baseline_name = best_baseline or str(engine_recommendation.get("best_baseline", "unavailable"))

    if requested == "baseline":
        if baseline_name and baseline_name != "unavailable":
            return {
                "engine": "baseline",
                "mode": "baseline",
                "headline": "Baseline Mode",
                "summary": f"The live cards are following `{baseline_name}` directly, regardless of the current HMM promotion status.",
            }
        return {
            "engine": "cash",
            "mode": "baseline",
            "headline": "Baseline Mode Unavailable",
            "summary": "No usable baseline was available to force as the live engine, so the app is staying flat.",
        }

    if requested == "hmm_research":
        return {
            "engine": "hmm",
            "mode": "hmm_research",
            "headline": "HMM Research Mode",
            "summary": "The live cards are following the HMM directly, even if it is not yet promoted. Treat the output as research-only.",
        }
    if requested == "hmm_ensemble":
        if consensus_available:
            return {
                "engine": "hmm_ensemble",
                "mode": "hmm_ensemble",
                "headline": "HMM Ensemble Mode",
                "summary": "The live cards are following the consensus-filtered HMM ensemble, which requires agreement across nearby seeds and state counts.",
            }
        return {
            "engine": "cash",
            "mode": "hmm_ensemble",
            "headline": "HMM Ensemble Unavailable",
            "summary": "Consensus diagnostics were not available, so the app cannot force the ensemble live mode on this run.",
        }

    if recommended_engine == "hmm":
        if consensus_available:
            return {
                "engine": "hmm_ensemble",
                "mode": "auto",
                "headline": "Auto Mode -> HMM Ensemble",
                "summary": "The HMM cleared the current promotion logic, and the live cards are following the consensus-filtered ensemble version for extra stability.",
            }
        return {
            "engine": "hmm",
            "mode": "auto",
            "headline": "Auto Mode -> HMM",
            "summary": "The HMM cleared the current promotion logic, so the live cards are following the HMM.",
        }
    if recommended_engine == "baseline" and baseline_name and baseline_name != "unavailable":
        return {
            "engine": "baseline",
            "mode": "auto",
            "headline": "Auto Mode -> Baseline",
            "summary": f"The HMM is still research-only on this run, so the live cards are following `{baseline_name}` instead.",
        }
    return {
        "engine": "cash",
        "mode": "auto",
        "headline": "Auto Mode -> No Active Deployment",
        "summary": "Neither the HMM nor the simple baselines are strong enough to justify an active live recommendation right now.",
    }


def build_control_interpretation_rows(
    *,
    interval: Interval,
    feature_pack: str,
    walk_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
    history_provider: str | None = None,
) -> pd.DataFrame:
    total_friction = strategy_config.cost_bps + strategy_config.spread_bps + strategy_config.slippage_bps + strategy_config.impact_bps

    if interval == "4hour":
        interval_text = "Primary higher-timeframe trading lane. Usually a good balance between stability and responsiveness for BTC."
    elif interval == "1day":
        interval_text = "Slower swing lane. Cleaner regimes, but fewer samples and slower reactions."
    else:
        interval_text = "Fast baseline lane. Useful for comparison, but more vulnerable to noise and overtrading."

    if strategy_config.posterior_threshold >= 0.75:
        posterior_text = "Strict. It will reject many bars unless the state assignment is very clean."
    elif strategy_config.posterior_threshold >= 0.65:
        posterior_text = "Balanced to cautious. A sensible research default."
    else:
        posterior_text = "Loose. It will admit more bars, but also more ambiguous state calls."

    if strategy_config.min_hold_bars >= 12:
        hold_text = "Sticky. Good for riding slower regimes, but exits will be slower."
    elif strategy_config.min_hold_bars >= 6:
        hold_text = "Moderate. Helps reduce churn without locking positions too long."
    else:
        hold_text = "Fast. More responsive, but easier to whipsaw."

    if strategy_config.cooldown_bars >= 8:
        cooldown_text = "Strict cooldown. Reduces rapid re-entry and whipsaw risk."
    elif strategy_config.cooldown_bars >= 3:
        cooldown_text = "Moderate cooldown. A balanced anti-whipsaw setting."
    elif strategy_config.cooldown_bars > 0:
        cooldown_text = "Light cooldown. Allows fairly quick re-entry."
    else:
        cooldown_text = "No cooldown. Most reactive, but also easiest to churn."

    if strategy_config.required_confirmations >= 4:
        confirmation_text = "Very cautious. Needs repeated evidence before entering."
    elif strategy_config.required_confirmations >= 2:
        confirmation_text = "Balanced. Helps filter one-bar noise."
    else:
        confirmation_text = "Aggressive. Will react to the first qualifying bar."

    if strategy_config.confidence_gap >= 0.1:
        gap_text = "Strict state-separation requirement. Ambiguous regimes will be filtered out."
    elif strategy_config.confidence_gap >= 0.05:
        gap_text = "Balanced separation filter."
    else:
        gap_text = "Loose separation filter. More trades, but more ambiguity."

    if total_friction >= 15:
        friction_text = "Heavy friction assumption. Good for stress testing whether the edge survives realistic costs."
    elif total_friction >= 8:
        friction_text = "Moderate friction assumption."
    else:
        friction_text = "Light friction assumption. Be careful not to overstate live tradability."

    rows = [
        {
            "control": "Interval",
            "value": interval,
            "interpretation": interval_text,
        },
        {
            "control": "Historical Provider",
            "value": history_provider or "n/a",
            "interpretation": (
                "Auto mode keeps Financial Modeling Prep as the primary source and only switches to Coinbase deep-history backfill when the intraday sample would otherwise be too thin, with Yahoo reserved as a last-resort fallback."
                if history_provider == "auto"
                else "The selected provider controls the historical bar source for the research run. Keep it fixed during comparisons so source changes do not masquerade as alpha."
            ),
        },
        {
            "control": "Feature Pack",
            "value": feature_pack,
            "interpretation": "This controls the market description the HMM sees. Better feature packs can matter more than tweaking entry thresholds.",
        },
        {
            "control": "Train Window",
            "value": f"{walk_config.train_bars} bars",
            "interpretation": f"About {_duration_label(interval, walk_config.train_bars)} of history per refit. Longer windows stabilize the model but adapt more slowly.",
        },
        {
            "control": "Validation Window",
            "value": f"{walk_config.validate_bars} bars",
            "interpretation": f"About {_duration_label(interval, walk_config.validate_bars)} used to label states with forward-return evidence before scoring the test slice.",
        },
        {
            "control": "Test Window",
            "value": f"{walk_config.test_bars} bars",
            "interpretation": f"About {_duration_label(interval, walk_config.test_bars)} of true out-of-sample scoring per fold.",
        },
        {
            "control": "Refit Stride",
            "value": f"{walk_config.refit_stride_bars} bars",
            "interpretation": f"The model refits roughly every {_duration_label(interval, walk_config.refit_stride_bars)}.",
        },
        {
            "control": "Purge / Embargo",
            "value": f"{walk_config.purge_bars} / {walk_config.embargo_bars} bars",
            "interpretation": "These leakage buffers reduce contamination around train/validate/test boundaries.",
        },
        {
            "control": "Posterior Threshold",
            "value": f"{strategy_config.posterior_threshold:.2f}",
            "interpretation": posterior_text,
        },
        {
            "control": "Top-Two Gap",
            "value": f"{strategy_config.confidence_gap:.2f}",
            "interpretation": gap_text,
        },
        {
            "control": "Required Confirmations",
            "value": f"{strategy_config.required_confirmations}",
            "interpretation": confirmation_text,
        },
        {
            "control": "Scoring Horizons",
            "value": ", ".join(str(horizon) for horizon in strategy_config.scoring_horizons),
            "interpretation": "These forward horizons vote on whether a state is truly tradable. Multiple horizons reduce one-window luck.",
        },
        {
            "control": "Validation Shrinkage",
            "value": f"{strategy_config.validation_shrinkage:.0f}",
            "interpretation": "Higher shrinkage pulls thin-sample validation edges back toward zero, which makes the state labels more conservative.",
        },
          {
              "control": "Consistent Horizons Required",
              "value": f"{strategy_config.min_consistent_horizons}",
              "interpretation": "This is the number of scored horizons that must agree before a state can be labeled as directional.",
          },
          {
              "control": "Allow Shorts",
              "value": "on" if strategy_config.allow_short else "off",
              "interpretation": (
                  "Validated bearish regimes can become actual short trades instead of only flattening the book."
                  if strategy_config.allow_short
                  else "Bearish regimes can only flatten exposure, not open outright short positions."
              ),
          },
          {
              "control": "Daily Confirmation Filter",
              "value": "on" if strategy_config.require_daily_confirmation else "off",
              "interpretation": (
                "Daily confirmation is enforcing agreement between the slower daily lane and the 4H execution lane."
                if strategy_config.require_daily_confirmation
                else "Daily confirmation is off, so the 4H lane trades on its own."
            ),
        },
        {
            "control": "Consensus Filter",
            "value": "on" if strategy_config.require_consensus_confirmation else "off",
            "interpretation": (
                "Consensus confirmation is enforcing agreement across nearby seeds and state counts before the trade is allowed."
                if strategy_config.require_consensus_confirmation
                else "Consensus confirmation is off, so the selected model can trade without nearby-model agreement."
            ),
        },
        {
            "control": "Consensus Gate Mode",
            "value": strategy_config.consensus_gate_mode,
            "interpretation": (
                "Hard mode fully blocks weak-consensus trades."
                if strategy_config.consensus_gate_mode == "hard"
                else "Entry-only mode blocks weak new entries but can keep an existing position alive through mild disagreement."
            ),
        },
        {
            "control": "Consensus Min Share",
            "value": f"{strategy_config.consensus_min_share:.0%}",
            "interpretation": "This is the minimum share of nearby models that must agree before the consensus filter allows a trade.",
        },
        {
            "control": "Min Hold",
            "value": f"{strategy_config.min_hold_bars} bars",
            "interpretation": hold_text,
        },
        {
            "control": "Cooldown",
            "value": f"{strategy_config.cooldown_bars} bars",
            "interpretation": cooldown_text,
        },
        {
            "control": "Total Friction Assumption",
            "value": f"{total_friction:.1f} bps",
            "interpretation": friction_text,
        },
    ]
    return pd.DataFrame(rows)
