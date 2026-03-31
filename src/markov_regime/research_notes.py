from __future__ import annotations

import pandas as pd

from markov_regime.walkforward import WalkForwardResult


def build_research_notes(
    selected_result: WalkForwardResult,
    model_comparison: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []

    sharpe_spread = float(model_comparison["sharpe"].max() - model_comparison["sharpe"].min())
    if sharpe_spread > 0.35:
        notes.append(
            f"Out-of-sample Sharpe moves by {sharpe_spread:.2f} across 5-9 states, so the signal is materially sensitive to state-count selection."
        )

    stability = selected_result.state_stability["stability_score"].median() if not selected_result.state_stability.empty else 0.0
    if stability < 0.55:
        notes.append(
            f"Median state stability is {stability:.2f}; canonical regimes are being realigned across retrains, which means labels remain heuristic rather than fully persistent."
        )

    sharpe_ci = selected_result.bootstrap.loc[selected_result.bootstrap["metric"] == "sharpe"]
    if not sharpe_ci.empty and float(sharpe_ci.iloc[0]["lower"]) <= 0.0 <= float(sharpe_ci.iloc[0]["upper"]):
        notes.append(
            "The moving block bootstrap confidence interval for Sharpe still spans zero, so the observed edge may be partly sampling noise."
        )

    coverage = selected_result.metrics.get("confidence_coverage", 0.0)
    if coverage < 0.4:
        notes.append(
            f"Only {coverage:.0%} of bars pass the directional guardrails, which is good for caution but also means the live signal will often be inactive and statistically thin."
        )

    cost_row = selected_result.cost_stress.loc[selected_result.cost_stress["sharpe"] <= 0.0]
    if not cost_row.empty:
        first_break = float(cost_row.iloc[0]["cost_bps"])
        notes.append(
            f"Strategy Sharpe falls to zero or worse by roughly {first_break:.0f} bps in the stress test, so small execution frictions can erase the apparent edge."
        )

    notes.append(
        "Hidden Markov regimes are latent summaries of the chosen feature set, not observable market truths; changing features, windows, or the sample start date can move the segmentation."
    )
    notes.append(
        "Validation windows are short relative to market non-stationarity, so regime labels can overreact to recent noise even with walk-forward separation."
    )
    return notes

