from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from markov_regime.config import DataConfig, ModelConfig, StrategyConfig, SweepConfig, WalkForwardConfig, default_walk_forward_config
from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.data import fetch_price_data
from markov_regime.features import build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.interpretation import (
    CONTROL_HELP,
    build_control_interpretation_rows,
    build_metric_interpretation_rows,
    build_trust_snapshot,
    first_sentence,
)
from markov_regime.research import run_feature_pack_comparison, run_timeframe_comparison
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
    plot_feature_pack_comparison,
    plot_regime_timeline,
    plot_robustness_results,
    plot_sensitivity,
    plot_state_stability,
    plot_timeframe_comparison,
)
from markov_regime.walkforward import compare_state_counts, summarize_state_count_results
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
st.caption("BTC 4H preset with daily confirmation, walk-forward HMM diagnostics, explicit guardrails, and confidence intervals.")

with st.sidebar.form("controls"):
    st.subheader("Research Controls")
    st.caption("Default preset: BTC 4H research with daily confirmation and optional 1H baseline checks.")
    symbol = st.text_input("Symbol", value="BTCUSD").upper()
    interval = st.selectbox("Interval", options=["4hour", "1day", "1hour"], index=0, help=CONTROL_HELP["interval"])
    default_walk = default_walk_forward_config(interval)
    feature_pack = st.selectbox("Feature pack", options=list(list_feature_packs()), index=0, help=CONTROL_HELP["feature_pack"])
    limit = st.number_input("Bars to fetch", min_value=300, max_value=10000, value=5000, step=100, help=CONTROL_HELP["limit"])
    selected_states = st.select_slider("Selected HMM states", options=[5, 6, 7, 8, 9], value=6, help=CONTROL_HELP["states"])
    train_bars = st.number_input("Train bars", min_value=120, max_value=5000, value=default_walk.train_bars, step=12, help=CONTROL_HELP["train_bars"])
    purge_bars = st.number_input("Purge bars", min_value=0, max_value=240, value=default_walk.purge_bars, step=1, help=CONTROL_HELP["purge_bars"])
    validate_bars = st.number_input("Validate bars", min_value=24, max_value=1500, value=default_walk.validate_bars, step=12, help=CONTROL_HELP["validate_bars"])
    embargo_bars = st.number_input("Embargo bars", min_value=0, max_value=240, value=default_walk.embargo_bars, step=1, help=CONTROL_HELP["embargo_bars"])
    test_bars = st.number_input("Test bars", min_value=24, max_value=1500, value=default_walk.test_bars, step=12, help=CONTROL_HELP["test_bars"])
    refit_stride = st.number_input("Refit stride", min_value=24, max_value=1500, value=default_walk.refit_stride_bars, step=12, help=CONTROL_HELP["refit_stride"])
    posterior_threshold = st.slider("Posterior threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.01, help=CONTROL_HELP["posterior_threshold"])
    min_hold_bars = st.slider("Min hold bars", min_value=1, max_value=24, value=6, help=CONTROL_HELP["min_hold_bars"])
    cooldown_bars = st.slider("Cooldown bars", min_value=0, max_value=24, value=4, help=CONTROL_HELP["cooldown_bars"])
    required_confirmations = st.slider("Required confirmations", min_value=1, max_value=6, value=2, help=CONTROL_HELP["required_confirmations"])
    confidence_gap = st.slider("Top-two posterior gap", min_value=0.0, max_value=0.25, value=0.06, step=0.01, help=CONTROL_HELP["confidence_gap"])
    require_daily_confirmation = st.checkbox("Require daily confirmation for 4H trades", value=True, help=CONTROL_HELP["require_daily_confirmation"])
    cost_bps = st.slider("Trading fee (bps)", min_value=0.0, max_value=25.0, value=2.0, step=0.5, help=CONTROL_HELP["cost_bps"])
    spread_bps = st.slider("Spread estimate (bps)", min_value=0.0, max_value=30.0, value=4.0, step=0.5, help=CONTROL_HELP["spread_bps"])
    slippage_bps = st.slider("Slippage estimate (bps)", min_value=0.0, max_value=30.0, value=3.0, step=0.5, help=CONTROL_HELP["slippage_bps"])
    impact_bps = st.slider("Liquidity impact (bps)", min_value=0.0, max_value=20.0, value=2.0, step=0.5, help=CONTROL_HELP["impact_bps"])
    robustness_symbols = st.text_input("Robustness basket", value="BTCUSD,ETHUSD,SOLUSD")
    run_timeframe_check = st.checkbox("Run timeframe comparison (4H / 1D / 1H)", value=True)
    run_feature_pack_check = st.checkbox("Run feature-pack ablation", value=True)
    auto_adjust_windows = st.checkbox("Auto-size windows if data is shorter than requested", value=True)
    run_clicked = st.form_submit_button("Run Research")

if run_clicked:
    try:
        with st.spinner("Fetching data, retraining walk-forward folds, and compiling diagnostics..."):
            data_config = DataConfig(symbol=symbol, interval=interval, limit=int(limit))
            model_config = ModelConfig(n_states=selected_states)
            feature_columns = get_feature_columns(feature_pack)
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
                require_daily_confirmation=require_daily_confirmation,
                cost_bps=cost_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                impact_bps=impact_bps,
            )
            execution_strategy_config = strategy_config
            model_strategy_config = replace(strategy_config, require_daily_confirmation=False)
            fetched = fetch_price_data(data_config)
            feature_frame = build_feature_frame(fetched.frame, feature_columns=feature_columns)
            walk_config, was_adjusted = (
                suggest_walk_forward_config(len(feature_frame), requested_walk_config)
                if auto_adjust_windows
                else (requested_walk_config, False)
            )
            comparison, results_by_state = compare_state_counts(
                feature_frame=feature_frame,
                feature_columns=feature_columns,
                interval=data_config.interval,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=model_strategy_config,
            )
            confirmation_fetched = None
            confirmation_result = None
            if data_config.interval == "4hour" and strategy_config.require_daily_confirmation:
                confirmation_data_config = DataConfig(symbol=symbol, interval="1day", limit=int(limit))
                confirmation_fetched = fetch_price_data(confirmation_data_config)
                confirmation_feature_frame = build_feature_frame(confirmation_fetched.frame, feature_columns=feature_columns)
                confirmation_walk_config, _ = (
                    suggest_walk_forward_config(len(confirmation_feature_frame), default_walk_forward_config("1day"))
                    if auto_adjust_windows
                    else (default_walk_forward_config("1day"), False)
                )
                _, confirmation_results_by_state = compare_state_counts(
                    feature_frame=confirmation_feature_frame,
                    feature_columns=feature_columns,
                    interval="1day",
                    model_config=model_config,
                    walk_config=confirmation_walk_config,
                    strategy_config=model_strategy_config,
                )
                results_by_state = {
                    n_states: apply_higher_timeframe_confirmation(
                        result,
                        confirmation_results_by_state[n_states],
                        interval=data_config.interval,
                        strategy_config=execution_strategy_config,
                        confirmation_interval="1day",
                    )
                    for n_states, result in results_by_state.items()
                }
                comparison = summarize_state_count_results(results_by_state)
                confirmation_result = confirmation_results_by_state[selected_states]
            selected_result = results_by_state[selected_states]
            sweep_results = parameter_sweep(
                predictions=selected_result.predictions,
                n_states=selected_states,
                base_config=execution_strategy_config,
                sweep_config=SweepConfig(),
                interval=data_config.interval,
            )
            robustness = run_multi_asset_robustness(
                symbols=parse_symbol_list(robustness_symbols),
                interval=data_config.interval,
                limit=int(limit),
                feature_columns=feature_columns,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=execution_strategy_config,
                auto_adjust_windows=auto_adjust_windows,
            )
            timeframe_comparison = (
                run_timeframe_comparison(
                    symbol=symbol,
                    limit=int(limit),
                    model_config=model_config,
                    strategy_config=execution_strategy_config,
                    feature_pack=feature_pack,
                    feature_columns=feature_columns,
                    auto_adjust_windows=auto_adjust_windows,
                )
                if run_timeframe_check
                else pd.DataFrame()
            )
            feature_pack_comparison = (
                run_feature_pack_comparison(
                    price_frame=fetched.frame,
                    interval=data_config.interval,
                    model_config=model_config,
                    strategy_config=execution_strategy_config,
                    symbol=symbol,
                    limit=int(limit),
                    auto_adjust_windows=auto_adjust_windows,
                )
                if run_feature_pack_check
                else pd.DataFrame()
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
                feature_columns=feature_columns,
                timeframe_comparison=timeframe_comparison,
                feature_pack_comparison=feature_pack_comparison,
            )
            st.session_state["analysis"] = {
                "data_url": fetched.source_url,
                "comparison": comparison,
                "selected_result": selected_result,
                "sweep_results": sweep_results,
                "robustness": robustness,
                "timeframe_comparison": timeframe_comparison,
                "feature_pack_comparison": feature_pack_comparison,
                "confirmation_enabled": bool(data_config.interval == "4hour" and strategy_config.require_daily_confirmation),
                "confirmation_summary": selected_result.confirmation_summary,
                "confirmation_result": confirmation_result,
                "confirmation_data_url": confirmation_fetched.source_url if confirmation_fetched is not None else "",
                "notes": notes,
                "symbol": symbol,
                "resolved_symbol": fetched.resolved_symbol,
                "interval": interval,
                "data_config": data_config,
                "model_config": model_config,
                "feature_pack": feature_pack,
                "feature_columns": feature_columns,
                "strategy_config": execution_strategy_config,
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
            "Try a larger history, a higher timeframe like `4hour` or `1day`, or enable automatic window sizing. "
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
timeframe_comparison = analysis["timeframe_comparison"]
feature_pack_comparison = analysis["feature_pack_comparison"]
notes = analysis["notes"]
latest_row = selected_result.predictions.iloc[-1]
guardrail_text = latest_row["guardrail_reason"] or "accepted"
metric_interpretation = build_metric_interpretation_rows(
    latest_row=latest_row.to_dict(),
    metrics=selected_result.metrics,
    bootstrap=selected_result.bootstrap,
    state_stability=selected_result.state_stability,
    robustness=robustness,
    interval=analysis["interval"],
    available_rows=analysis["available_rows"],
    walk_adjusted=analysis["walk_adjusted"],
)
control_interpretation = build_control_interpretation_rows(
    interval=analysis["interval"],
    feature_pack=analysis["feature_pack"],
    walk_config=analysis["walk_config"],
    strategy_config=analysis["strategy_config"],
)
trust_snapshot = build_trust_snapshot(
    metrics=selected_result.metrics,
    bootstrap=selected_result.bootstrap,
    state_stability=selected_result.state_stability,
    robustness=robustness,
    interval=analysis["interval"],
    available_rows=analysis["available_rows"],
    walk_adjusted=analysis["walk_adjusted"],
)
metric_lookup = metric_interpretation.set_index("metric")

metric_items = [
    ("Current State", str(metric_lookup.loc["Current State", "value"]), first_sentence(str(metric_lookup.loc["Current State", "interpretation"]))),
    ("Held Position", str(metric_lookup.loc["Held Position", "value"]), first_sentence(str(metric_lookup.loc["Held Position", "interpretation"]))),
    ("Latest Candidate", str(metric_lookup.loc["Latest Candidate", "value"]), first_sentence(str(metric_lookup.loc["Latest Candidate", "interpretation"]))),
]
if "Daily Confirmation" in metric_lookup.index:
    metric_items.append(
        ("Daily Confirmation", str(metric_lookup.loc["Daily Confirmation", "value"]), first_sentence(str(metric_lookup.loc["Daily Confirmation", "interpretation"])))
    )
metric_items.extend(
    [
    ("Posterior", str(metric_lookup.loc["Posterior", "value"]), first_sentence(str(metric_lookup.loc["Posterior", "interpretation"]))),
    ("Sharpe", str(metric_lookup.loc["Sharpe", "value"]), first_sentence(str(metric_lookup.loc["Sharpe", "interpretation"]))),
    ("Ann. Return", str(metric_lookup.loc["Annualized Return", "value"]), first_sentence(str(metric_lookup.loc["Annualized Return", "interpretation"]))),
    ]
)
metrics_columns = st.columns(len(metric_items))
for column, (label, value, note) in zip(metrics_columns, metric_items, strict=True):
    column.markdown(
        (
            "<div class='metric-card'>"
            f"<strong>{label}</strong><br><span style='font-size:1.35rem'>{value}</span>"
            f"<br><span style='font-size:0.82rem;color:#475569'>{note}</span></div>"
        ),
        unsafe_allow_html=True,
    )

trust_message = trust_snapshot["summary"]
if trust_snapshot["severity"] == "success":
    st.success(trust_message)
elif trust_snapshot["severity"] == "error":
    st.error(trust_message)
else:
    st.warning(trust_message)

st.caption("Held Position shows the active book. Latest Candidate shows what the newest bar alone supports before hold and cooldown mechanics are applied.")
if analysis["confirmation_enabled"]:
    st.caption("Daily Confirmation shows whether the slower daily lane currently confirms, blocks, or stays neutral on the 4H trade.")
st.caption(f"Latest guardrail status: `{guardrail_text}` | Data source: `{analysis['data_url']}`")
if analysis["confirmation_enabled"] and analysis["confirmation_data_url"]:
    st.caption(f"Daily confirmation source: `{analysis['confirmation_data_url']}`")
st.caption(f"Feature pack: `{analysis['feature_pack']}`")
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

overview_tab, trades_tab, interpretation_tab, confirmation_tab, timeframe_tab, feature_tab, model_tab, stability_tab, sensitivity_tab, confidence_tab, robustness_tab, notes_tab, export_tab = st.tabs(
    ["Overview", "Trades", "Interpretation", "Confirmation", "Timeframes", "Feature Packs", "Model Comparison", "State Stability", "Sensitivity", "Confidence", "Robustness", "Research Notes", "Exports"]
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

with trades_tab:
    summary_col, log_col = st.columns([0.9, 1.5])
    with summary_col:
        st.subheader("Trade Summary")
        st.dataframe(selected_result.trade_summary, use_container_width=True, hide_index=True)
        st.caption("Trade-level metrics focus on realized trade outcomes, while `bar_win_rate` in the overview table tracks winning bars while exposure is active.")
    with log_col:
        st.subheader("Trade Log")
        if selected_result.trade_log.empty:
            st.info("No trades were opened on this run.")
        else:
            st.dataframe(selected_result.trade_log, use_container_width=True, hide_index=True)

with timeframe_tab:
    st.plotly_chart(plot_timeframe_comparison(timeframe_comparison), use_container_width=True)
    st.dataframe(timeframe_comparison, use_container_width=True)
    st.caption("The timeframe comparison uses the same strategy controls but interval-specific walk-forward window presets across `4hour`, `1day`, and `1hour`. Treat it as a relative sanity check, not a perfect apples-to-apples scorecard.")

with feature_tab:
    st.plotly_chart(plot_feature_pack_comparison(feature_pack_comparison), use_container_width=True)
    st.dataframe(feature_pack_comparison, use_container_width=True)
    st.caption("Feature-pack ablation holds the timeframe and strategy controls fixed while changing what the HMM sees. This is the cleanest way to tell whether signal improvements are coming from better market representation or just tighter filters.")

with interpretation_tab:
    st.subheader("Current Run Readout")
    st.dataframe(metric_interpretation, use_container_width=True, hide_index=True)
    st.caption("These explanations interpret the latest bar and the current backtest in plain English. They are heuristics for readability, not hard statistical proof.")
    st.subheader("Current Control Meanings")
    st.dataframe(control_interpretation, use_container_width=True, hide_index=True)
    st.caption("These rows explain what the current settings are encouraging the strategy to do, so you can tell whether results are coming from signal quality or just stricter filters.")

with confirmation_tab:
    if analysis["confirmation_enabled"]:
        st.subheader("4H + 1D Agreement")
        st.dataframe(analysis["confirmation_summary"], use_container_width=True, hide_index=True)
        confirmation_columns = [
            "timestamp",
            "base_signal_position",
            "signal_position",
            "base_candidate_action",
            "candidate_action",
            "confirmation_effective_direction",
            "confirmation_status",
            "guardrail_reason",
        ]
        available_columns = [column for column in confirmation_columns if column in selected_result.predictions.columns]
        st.dataframe(selected_result.predictions.loc[:, available_columns].tail(50), use_container_width=True)
        st.caption("`base_*` columns show the raw 4H proposal before the daily filter. The plain `signal_position` and `candidate_action` columns show the executed result after daily confirmation is applied.")
    else:
        st.info("Daily confirmation is currently off. Enable `Require daily confirmation for 4H trades` to make the daily lane confirm or veto the 4H execution.")

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
