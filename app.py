from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from markov_regime.config import DataConfig, ModelConfig, StrategyConfig, SweepConfig, WalkForwardConfig
from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.data import fetch_price_data
from markov_regime.features import FEATURE_COLUMNS, build_feature_frame
from markov_regime.reporting import export_signal_report
from markov_regime.research_notes import build_research_notes
from markov_regime.robustness import parse_symbol_list, run_multi_asset_robustness
from markov_regime.strategy import parameter_sweep
from markov_regime.ui import (
    plot_cost_stress,
    plot_equity_curve,
    plot_forward_return_heatmap,
    plot_guardrail_summary,
    plot_model_comparison,
    plot_regime_timeline,
    plot_robustness_results,
    plot_sensitivity,
    plot_state_stability,
)
from markov_regime.walkforward import compare_state_counts
from markov_regime.walkforward import suggest_walk_forward_config

st.set_page_config(page_title="Markov Regime Research", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap');
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 30%),
            linear-gradient(180deg, #f6f4ee 0%, #edf1eb 100%);
    }
    h1, h2, h3 {
        font-family: 'Instrument Serif', serif;
        letter-spacing: 0.02em;
    }
    [data-testid="stSidebar"] {
        background: rgba(248, 250, 249, 0.95);
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .note-card {
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.72);
        border-left: 5px solid #0f766e;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Markov Regime Research")
st.caption("BTC hourly preset with walk-forward HMM diagnostics, explicit guardrails, parameter sensitivity, and confidence intervals.")

with st.sidebar.form("controls"):
    st.subheader("Research Controls")
    st.caption("Default preset: BTC hourly with moderate rolling windows for local research.")
    symbol = st.text_input("Symbol", value="BTCUSD").upper()
    interval = st.selectbox("Interval", options=["1hour", "1day"], index=0)
    limit = st.number_input("Bars to fetch", min_value=600, max_value=10000, value=2000, step=100)
    selected_states = st.select_slider("Selected HMM states", options=[5, 6, 7, 8, 9], value=6)
    train_bars = st.number_input("Train bars", min_value=200, max_value=5000, value=720, step=24)
    purge_bars = st.number_input("Purge bars", min_value=0, max_value=240, value=6, step=1)
    validate_bars = st.number_input("Validate bars", min_value=50, max_value=1500, value=120, step=12)
    embargo_bars = st.number_input("Embargo bars", min_value=0, max_value=240, value=6, step=1)
    test_bars = st.number_input("Test bars", min_value=50, max_value=1500, value=120, step=12)
    refit_stride = st.number_input("Refit stride", min_value=25, max_value=1500, value=120, step=12)
    posterior_threshold = st.slider("Posterior threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.01)
    min_hold_bars = st.slider("Min hold bars", min_value=1, max_value=24, value=4)
    cooldown_bars = st.slider("Cooldown bars", min_value=0, max_value=24, value=4)
    required_confirmations = st.slider("Required confirmations", min_value=1, max_value=6, value=2)
    confidence_gap = st.slider("Top-two posterior gap", min_value=0.0, max_value=0.25, value=0.06, step=0.01)
    cost_bps = st.slider("Trading fee (bps)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
    spread_bps = st.slider("Spread estimate (bps)", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
    slippage_bps = st.slider("Slippage estimate (bps)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
    impact_bps = st.slider("Liquidity impact (bps)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    robustness_symbols = st.text_input("Robustness basket", value="BTCUSD,ETHUSD,SOLUSD")
    auto_adjust_windows = st.checkbox("Auto-size windows if data is shorter than requested", value=True)
    run_clicked = st.form_submit_button("Run Research")

if run_clicked:
    try:
        with st.spinner("Fetching data, retraining walk-forward folds, and compiling diagnostics..."):
            data_config = DataConfig(symbol=symbol, interval=interval, limit=int(limit))
            model_config = ModelConfig(n_states=selected_states)
            requested_walk_config = WalkForwardConfig(
                train_bars=int(train_bars),
                purge_bars=int(purge_bars),
                validate_bars=int(validate_bars),
                embargo_bars=int(embargo_bars),
                test_bars=int(test_bars),
                refit_stride_bars=int(refit_stride),
            )
            strategy_config = StrategyConfig(
                posterior_threshold=posterior_threshold,
                min_hold_bars=int(min_hold_bars),
                cooldown_bars=int(cooldown_bars),
                required_confirmations=int(required_confirmations),
                confidence_gap=confidence_gap,
                cost_bps=cost_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                impact_bps=impact_bps,
            )
            fetched = fetch_price_data(data_config)
            feature_frame = build_feature_frame(fetched.frame)
            walk_config, was_adjusted = (
                suggest_walk_forward_config(len(feature_frame), requested_walk_config)
                if auto_adjust_windows
                else (requested_walk_config, False)
            )
            comparison, results_by_state = compare_state_counts(
                feature_frame=feature_frame,
                feature_columns=FEATURE_COLUMNS,
                interval=data_config.interval,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=strategy_config,
            )
            selected_result = results_by_state[selected_states]
            sweep_results = parameter_sweep(
                predictions=selected_result.predictions,
                n_states=selected_states,
                base_config=strategy_config,
                sweep_config=SweepConfig(),
                interval=data_config.interval,
            )
            robustness = run_multi_asset_robustness(
                symbols=parse_symbol_list(robustness_symbols),
                interval=data_config.interval,
                limit=int(limit),
                feature_columns=FEATURE_COLUMNS,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=strategy_config,
                auto_adjust_windows=auto_adjust_windows,
            )
            notes = build_research_notes(selected_result, comparison)
            artifact = write_run_artifact_bundle(
                symbol=symbol,
                resolved_symbol=fetched.resolved_symbol,
                interval=interval,
                data_url=fetched.source_url,
                raw_frame=fetched.frame,
                feature_frame=feature_frame,
                data_config=data_config,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=strategy_config,
                selected_result=selected_result,
                comparison=comparison,
                sweep_results=sweep_results,
                notes=notes,
                robustness=robustness,
            )
            st.session_state["analysis"] = {
                "data_url": fetched.source_url,
                "comparison": comparison,
                "selected_result": selected_result,
                "sweep_results": sweep_results,
                "robustness": robustness,
                "notes": notes,
                "symbol": symbol,
                "resolved_symbol": fetched.resolved_symbol,
                "interval": interval,
                "data_config": data_config,
                "model_config": model_config,
                "strategy_config": strategy_config,
                "walk_config": walk_config,
                "walk_adjusted": was_adjusted,
                "available_rows": len(feature_frame),
                "raw_rows": len(fetched.frame),
                "feature_start": feature_frame["timestamp"].iloc[0],
                "feature_end": feature_frame["timestamp"].iloc[-1],
                "raw_start": fetched.frame["timestamp"].iloc[0],
                "raw_end": fetched.frame["timestamp"].iloc[-1],
                "latest_close": float(fetched.frame["close"].iloc[-1]),
                "artifact_run_id": artifact.run_id,
                "artifact_root": str(artifact.root),
                "artifact_manifest": str(artifact.manifest_path),
            }
    except ValueError as exc:
        st.session_state.pop("analysis", None)
        st.error(str(exc))
        st.info(
            "Try a larger history, the hourly interval, or enable automatic window sizing. "
            "For crypto, use `BTCUSD` rather than `BTC` so FMP returns the full series."
        )
        st.stop()

analysis = st.session_state.get("analysis")
if not analysis:
    st.info("Choose a symbol and run the research panel to generate walk-forward diagnostics.")
    st.stop()

comparison = analysis["comparison"]
selected_result = analysis["selected_result"]
sweep_results = analysis["sweep_results"]
robustness = analysis["robustness"]
notes = analysis["notes"]
latest_row = selected_result.predictions.iloc[-1]

signal_text = {1: "Long", 0: "Flat", -1: "Short"}.get(int(latest_row["signal_position"]), "Flat")
guardrail_text = latest_row["guardrail_reason"] or "Signal accepted"

metrics_columns = st.columns(5)
metric_items = [
    ("Current State", f"{int(latest_row['canonical_state'])}"),
    ("Signal", signal_text),
    ("Posterior", f"{latest_row['max_posterior']:.2f}"),
    ("Sharpe", f"{selected_result.metrics['sharpe']:.2f}"),
    ("Ann. Return", f"{selected_result.metrics['annualized_return']:.1%}"),
]
for column, (label, value) in zip(metrics_columns, metric_items, strict=True):
    column.markdown(f"<div class='metric-card'><strong>{label}</strong><br><span style='font-size:1.4rem'>{value}</span></div>", unsafe_allow_html=True)

st.caption(f"Latest guardrail decision: `{guardrail_text}` | Data source: `{analysis['data_url']}`")
if analysis.get("resolved_symbol") and analysis["resolved_symbol"] != analysis["symbol"]:
    st.info(
        f"Resolved symbol `{analysis['symbol']}` to `{analysis['resolved_symbol']}` for data fetch compatibility."
    )
if analysis.get("walk_adjusted"):
    adjusted = analysis["walk_config"]
    st.warning(
        "Requested walk-forward windows were reduced to fit the available sample: "
        f"train={adjusted.train_bars}, purge={adjusted.purge_bars}, validate={adjusted.validate_bars}, "
        f"embargo={adjusted.embargo_bars}, test={adjusted.test_bars}, stride={adjusted.refit_stride_bars} across "
        f"{analysis['available_rows']} usable rows."
    )
else:
    current_walk = analysis["walk_config"]
    st.caption(
        "Walk-forward layout: "
        f"train={current_walk.train_bars}, purge={current_walk.purge_bars}, validate={current_walk.validate_bars}, "
        f"embargo={current_walk.embargo_bars}, test={current_walk.test_bars}, stride={current_walk.refit_stride_bars}"
    )

data_columns = st.columns(5)
data_items = [
    ("Requested Symbol", analysis["symbol"]),
    ("Resolved Symbol", analysis["resolved_symbol"]),
    ("Fetched Rows", f"{analysis['raw_rows']}"),
    ("Usable Rows", f"{analysis['available_rows']}"),
    ("Latest Close", f"{analysis['latest_close']:,.2f}"),
]
for column, (label, value) in zip(data_columns, data_items, strict=True):
    column.markdown(f"<div class='metric-card'><strong>{label}</strong><br><span style='font-size:1.15rem'>{value}</span></div>", unsafe_allow_html=True)
st.caption(
    "Data window: "
    f"{pd.Timestamp(analysis['raw_start']).strftime('%Y-%m-%d %H:%M')} to "
    f"{pd.Timestamp(analysis['raw_end']).strftime('%Y-%m-%d %H:%M')} raw bars | "
    f"{pd.Timestamp(analysis['feature_start']).strftime('%Y-%m-%d %H:%M')} to "
    f"{pd.Timestamp(analysis['feature_end']).strftime('%Y-%m-%d %H:%M')} usable feature bars"
)

overview_tab, model_tab, stability_tab, sensitivity_tab, confidence_tab, robustness_tab, notes_tab, export_tab = st.tabs(
    ["Overview", "Model Comparison", "State Stability", "Sensitivity", "Confidence", "Robustness", "Research Notes", "Exports"]
)

with overview_tab:
    left, right = st.columns([1.4, 1.0])
    with left:
        st.plotly_chart(plot_equity_curve(selected_result.predictions), use_container_width=True)
        st.plotly_chart(plot_regime_timeline(selected_result.predictions), use_container_width=True)
    with right:
        metric_frame = pd.DataFrame(
            {
                "strategy": pd.Series(selected_result.metrics),
                "buy_and_hold": pd.Series(selected_result.benchmark_metrics),
            }
        )
        st.dataframe(metric_frame, use_container_width=True)
        st.plotly_chart(plot_guardrail_summary(selected_result.guardrail_summary), use_container_width=True)
        st.plotly_chart(plot_cost_stress(selected_result.cost_stress), use_container_width=True)

with model_tab:
    st.plotly_chart(plot_model_comparison(comparison), use_container_width=True)
    st.dataframe(comparison, use_container_width=True)

with stability_tab:
    st.plotly_chart(plot_state_stability(selected_result.state_stability), use_container_width=True)
    st.dataframe(selected_result.state_stability, use_container_width=True)
    st.plotly_chart(plot_forward_return_heatmap(selected_result.forward_returns), use_container_width=True)
    st.dataframe(selected_result.forward_returns, use_container_width=True)

with sensitivity_tab:
    metric_choice = st.selectbox(
        "Sensitivity metric",
        options=["sharpe", "annualized_return", "max_drawdown", "confidence_coverage"],
        index=0,
    )
    sensitivity_columns = st.columns(2)
    sensitivity_columns[0].plotly_chart(plot_sensitivity(sweep_results, "posterior_threshold", metric_choice), use_container_width=True)
    sensitivity_columns[1].plotly_chart(plot_sensitivity(sweep_results, "min_hold_bars", metric_choice), use_container_width=True)
    sensitivity_columns = st.columns(2)
    sensitivity_columns[0].plotly_chart(plot_sensitivity(sweep_results, "cooldown_bars", metric_choice), use_container_width=True)
    sensitivity_columns[1].plotly_chart(plot_sensitivity(sweep_results, "required_confirmations", metric_choice), use_container_width=True)
    st.dataframe(sweep_results.head(25), use_container_width=True)

with confidence_tab:
    st.subheader("Bootstrap Confidence Intervals")
    st.dataframe(selected_result.bootstrap, use_container_width=True)
    st.subheader("Fold Diagnostics")
    st.dataframe(selected_result.fold_diagnostics, use_container_width=True)

with robustness_tab:
    st.plotly_chart(plot_robustness_results(robustness), use_container_width=True)
    st.dataframe(robustness, use_container_width=True)

with notes_tab:
    st.subheader("Research Notes")
    for note in notes:
        st.markdown(f"<div class='note-card'>{note}</div>", unsafe_allow_html=True)

with export_tab:
    st.subheader("Run Artifacts")
    st.write(f"Snapshot run id: `{analysis['artifact_run_id']}`")
    st.write(f"Artifact folder: `{analysis['artifact_root']}`")
    st.write(f"Manifest: `{analysis['artifact_manifest']}`")
    st.write("Save the current signal history in both CSV and JSON formats.")
    if st.button("Export current signal report"):
        exported = export_signal_report(
            selected_result.predictions,
            symbol=analysis["symbol"],
            interval=analysis["interval"],
        )
        st.success(f"Saved CSV to {exported['csv']} and JSON to {exported['json']}")
    st.dataframe(selected_result.predictions.tail(50), use_container_width=True)
