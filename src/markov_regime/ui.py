from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = ["#0f766e", "#ef4444", "#d97706", "#1d4ed8", "#166534", "#7c3aed", "#c2410c", "#475569", "#be123c"]


def plot_equity_curve(predictions: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=predictions["timestamp"], y=predictions["asset_wealth"], name="Asset"))
    figure.add_trace(go.Scatter(x=predictions["timestamp"], y=predictions["strategy_wealth"], name="Strategy"))
    figure.update_layout(
        title="Walk-Forward Equity Curve",
        template="plotly_white",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def plot_regime_timeline(predictions: pd.DataFrame) -> go.Figure:
    figure = px.scatter(
        predictions,
        x="timestamp",
        y="close",
        color="canonical_state",
        color_discrete_sequence=PALETTE,
        title="Price Colored by Canonical State",
    )
    figure.update_traces(marker=dict(size=6))
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure


def plot_model_comparison(comparison: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=comparison["n_states"], y=comparison["stability_score"], name="Median Stability"))
    figure.add_trace(go.Scatter(x=comparison["n_states"], y=comparison["sharpe"], name="Sharpe", mode="lines+markers", yaxis="y2"))
    figure.update_layout(
        title="State Count Comparison (5-9)",
        template="plotly_white",
        yaxis=dict(title="Stability"),
        yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
        xaxis=dict(title="States"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def plot_cost_stress(cost_stress: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=cost_stress["cost_bps"], y=cost_stress["sharpe"], mode="lines+markers", name="Sharpe"))
    figure.add_trace(
        go.Scatter(
            x=cost_stress["cost_bps"],
            y=cost_stress["annualized_return"],
            mode="lines+markers",
            name="Annualized Return",
            yaxis="y2",
        )
    )
    figure.update_layout(
        title="Transaction Cost Stress Test (Dynamic Base + Extra Bps)",
        template="plotly_white",
        xaxis=dict(title="Cost (bps)"),
        yaxis=dict(title="Sharpe"),
        yaxis2=dict(title="Annualized Return", overlaying="y", side="right"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def plot_state_stability(stability: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=stability["canonical_state"], y=stability["stability_score"], name="Stability Score"))
    figure.add_trace(
        go.Scatter(
            x=stability["canonical_state"],
            y=stability["sign_flip_rate"],
            mode="lines+markers",
            name="Sign Flip Rate",
            yaxis="y2",
        )
    )
    figure.update_layout(
        title="State Stability Across Retrains",
        template="plotly_white",
        xaxis=dict(title="Canonical State"),
        yaxis=dict(title="Stability Score"),
        yaxis2=dict(title="Sign Flip Rate", overlaying="y", side="right"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def plot_forward_return_heatmap(forward_returns: pd.DataFrame) -> go.Figure:
    pivot = forward_returns.pivot(index="canonical_state", columns="horizon_bars", values="mean_return")
    figure = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Regime-Conditioned Mean Forward Returns",
        labels=dict(x="Horizon (bars)", y="Canonical State", color="Mean Return"),
    )
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure


def plot_guardrail_summary(guardrails: pd.DataFrame) -> go.Figure:
    figure = px.bar(
        guardrails,
        x="guardrail_reason",
        y="share",
        color="guardrail_reason",
        title="Why the Strategy Stayed Flat",
        color_discrete_sequence=PALETTE,
    )
    figure.update_layout(template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    return figure


def plot_robustness_results(robustness: pd.DataFrame) -> go.Figure:
    if robustness.empty or "status" not in robustness.columns:
        figure = go.Figure()
        figure.update_layout(
            title="Cross-Asset Robustness vs Buy-and-Hold",
            template="plotly_white",
            annotations=[dict(text="No robustness basket configured", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure
    ok_rows = robustness.loc[robustness["status"] == "ok"].copy()
    if ok_rows.empty:
        figure = go.Figure()
        figure.update_layout(
            title="Cross-Asset Robustness vs Buy-and-Hold",
            template="plotly_white",
            annotations=[dict(text="No successful robustness runs", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure
    figure = px.bar(
        ok_rows,
        x="resolved_symbol",
        y=["sharpe", "benchmark_sharpe"],
        barmode="group",
        title="Cross-Asset Robustness vs Buy-and-Hold",
    )
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure


def plot_timeframe_comparison(timeframes: pd.DataFrame) -> go.Figure:
    if timeframes.empty or "status" not in timeframes.columns:
        figure = go.Figure()
        figure.update_layout(
            title="Timeframe Comparison",
            template="plotly_white",
            annotations=[dict(text="No timeframe comparison available", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure

    ok_rows = timeframes.loc[timeframes["status"] == "ok"].copy()
    if ok_rows.empty:
        figure = go.Figure()
        figure.update_layout(
            title="Timeframe Comparison",
            template="plotly_white",
            annotations=[dict(text="No successful timeframe runs", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure

    interval_order = {"4hour": 0, "1day": 1, "1hour": 2}
    ok_rows = ok_rows.sort_values(
        by="interval",
        key=lambda values: values.map(lambda item: interval_order.get(str(item), len(interval_order))),
        kind="stable",
    )

    figure = go.Figure()
    figure.add_trace(go.Bar(x=ok_rows["interval"], y=ok_rows["stability_score"], name="Stability"))
    figure.add_trace(go.Scatter(x=ok_rows["interval"], y=ok_rows["sharpe"], mode="lines+markers", name="Sharpe", yaxis="y2"))
    figure.update_layout(
        title="Timeframe Comparison",
        template="plotly_white",
        yaxis=dict(title="Stability"),
        yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
        xaxis=dict(title="Interval"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def plot_feature_pack_comparison(feature_comparison: pd.DataFrame) -> go.Figure:
    if feature_comparison.empty or "status" not in feature_comparison.columns:
        figure = go.Figure()
        figure.update_layout(
            title="Feature Pack Comparison",
            template="plotly_white",
            annotations=[dict(text="No feature comparison available", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure

    ok_rows = feature_comparison.loc[feature_comparison["status"] == "ok"].copy()
    if ok_rows.empty:
        figure = go.Figure()
        figure.update_layout(
            title="Feature Pack Comparison",
            template="plotly_white",
            annotations=[dict(text="No successful feature-pack runs", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure

    figure = go.Figure()
    figure.add_trace(go.Bar(x=ok_rows["feature_pack"], y=ok_rows["stability_score"], name="Stability"))
    figure.add_trace(go.Scatter(x=ok_rows["feature_pack"], y=ok_rows["sharpe"], mode="lines+markers", name="Sharpe", yaxis="y2"))
    figure.update_layout(
        title="Feature Pack Comparison",
        template="plotly_white",
        yaxis=dict(title="Stability"),
        yaxis2=dict(title="Sharpe", overlaying="y", side="right"),
        xaxis=dict(title="Feature Pack"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def plot_baseline_comparison(baseline_comparison: pd.DataFrame) -> go.Figure:
    if baseline_comparison.empty:
        figure = go.Figure()
        figure.update_layout(
            title="Baseline Comparison",
            template="plotly_white",
            annotations=[dict(text="No baseline comparison available", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure

    figure = go.Figure()
    figure.add_trace(go.Bar(x=baseline_comparison["baseline"], y=baseline_comparison["sharpe"], name="Sharpe"))
    figure.add_trace(
        go.Scatter(
            x=baseline_comparison["baseline"],
            y=baseline_comparison["annualized_return"],
            mode="lines+markers",
            name="Annualized Return",
            yaxis="y2",
        )
    )
    figure.update_layout(
        title="Baseline Comparison",
        template="plotly_white",
        yaxis=dict(title="Sharpe"),
        yaxis2=dict(title="Annualized Return", overlaying="y", side="right"),
        xaxis=dict(title="Baseline"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return figure


def plot_consensus_timeline(consensus_timeline: pd.DataFrame) -> go.Figure:
    if consensus_timeline.empty:
        figure = go.Figure()
        figure.update_layout(
            title="Consensus Timeline",
            template="plotly_white",
            annotations=[dict(text="No consensus diagnostics available", x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")],
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return figure

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=consensus_timeline["timestamp"], y=consensus_timeline["close"], name="Close"))
    figure.add_trace(
        go.Scatter(
            x=consensus_timeline["timestamp"],
            y=consensus_timeline["position_consensus_share"],
            mode="lines",
            name="Held Consensus Share",
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=consensus_timeline["timestamp"],
            y=consensus_timeline["candidate_consensus_share"],
            mode="lines",
            name="Candidate Consensus Share",
            yaxis="y2",
            line=dict(dash="dot"),
        )
    )
    figure.update_layout(
        title="Consensus Timeline",
        template="plotly_white",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Consensus Share", overlaying="y", side="right", range=[0.0, 1.0]),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h"),
    )
    return figure


def sensitivity_aggregate(sweep_results: pd.DataFrame, parameter: str, metric: str) -> pd.DataFrame:
    aggregated = (
        sweep_results.groupby(parameter, as_index=False)[metric]
        .median()
        .rename(columns={metric: f"median_{metric}"})
    )
    return aggregated


def plot_sensitivity(sweep_results: pd.DataFrame, parameter: str, metric: str) -> go.Figure:
    aggregated = sensitivity_aggregate(sweep_results, parameter, metric)
    figure = px.line(
        aggregated,
        x=parameter,
        y=f"median_{metric}",
        markers=True,
        title=f"{metric.replace('_', ' ').title()} vs {parameter.replace('_', ' ').title()}",
    )
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure
