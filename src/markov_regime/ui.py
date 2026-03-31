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
        title="Transaction Cost Stress Test",
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

