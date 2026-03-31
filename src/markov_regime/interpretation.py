from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from markov_regime.config import Interval, StrategyConfig, WalkForwardConfig

POSITION_LABELS: dict[int, str] = {1: "Long", 0: "Flat", -1: "Short"}

CONTROL_HELP: dict[str, str] = {
    "interval": "Primary research timeframe. `4hour` is the main trading lane, `1day` is slower confirmation, and `1hour` is the noisier baseline.",
    "feature_pack": "Chooses what market features the HMM sees. Richer packs can improve regime separation, but they can also be more fragile.",
    "limit": "How many vendor bars to fetch. More history usually makes walk-forward conclusions less fragile.",
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
    "require_daily_confirmation": "When enabled on `4hour`, the strategy only executes exposure when the latest daily lane agrees with the 4H direction.",
    "cost_bps": "Trading fee assumption in basis points. This is direct fee friction before spread and slippage.",
    "spread_bps": "Bid/ask spread assumption in basis points. Wider spreads punish turnover-heavy variants.",
    "slippage_bps": "Execution slippage assumption in basis points. Higher values test whether the edge survives imperfect fills.",
    "impact_bps": "Liquidity impact penalty in basis points. Higher values stress strategies that need more urgent execution.",
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
        "1hour": (1500, 4000),
        "4hour": (700, 1500),
        "1day": (500, 1200),
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


def first_sentence(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if ". " in cleaned:
        return cleaned.split(". ", 1)[0].rstrip(".") + "."
    return cleaned


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
    return f"Latest guardrail status: {reason_key.replace('_', ' ')}."


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
            "metric": "Trades",
            "value": f"{trades:.0f}",
            "interpretation": trades_text,
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
                "value": f"{confirmation_status} ({position_label(confirmation_direction)})",
                "interpretation": confirmation_text,
            },
        )
    return pd.DataFrame(rows)


def build_control_interpretation_rows(
    *,
    interval: Interval,
    feature_pack: str,
    walk_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
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
            "control": "Daily Confirmation Filter",
            "value": "on" if strategy_config.require_daily_confirmation else "off",
            "interpretation": (
                "Daily confirmation is enforcing agreement between the slower daily lane and the 4H execution lane."
                if strategy_config.require_daily_confirmation
                else "Daily confirmation is off, so the 4H lane trades on its own."
            ),
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
