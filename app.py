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
from markov_regime.consensus import apply_consensus_confirmation, compare_consensus_gate_modes, run_consensus_diagnostics
from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.data import fetch_live_quote, fetch_price_data
from markov_regime.features import build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.interpretation import (
    CONTROL_HELP,
    build_control_interpretation_rows,
    build_execution_plan,
    build_metric_interpretation_rows,
    build_promotion_gate_rows,
    build_trust_snapshot,
    first_sentence,
    recommend_strategy_engine,
    summarize_promotion_gates,
)
from markov_regime.research import (
    nested_holdout_evaluation,
    nested_holdout_summary_frame,
    run_candidate_search,
    run_feature_pack_comparison,
    run_timeframe_comparison,
    summarize_candidate_search,
)
from markov_regime.reporting import export_signal_report
from markov_regime.research_notes import build_research_notes
from markov_regime.robustness import parse_symbol_list, run_multi_asset_robustness
from markov_regime.strategy import parameter_sweep
from markov_regime.ui import (
    plot_baseline_comparison,
    plot_candidate_search,
    plot_cost_stress,
    plot_consensus_mode_comparison,
    plot_equity_curve,
    plot_forward_return_heatmap,
    plot_guardrail_summary,
    plot_model_comparison,
    plot_feature_pack_comparison,
    plot_regime_timeline,
    plot_robustness_results,
    plot_consensus_timeline,
    plot_sensitivity,
    plot_state_stability,
    plot_timeframe_comparison,
)
from markov_regime.walkforward import compare_state_counts, summarize_state_count_results
from markov_regime.walkforward import suggest_walk_forward_config

st.set_page_config(page_title="Markov Regime Research", layout="wide")


@st.cache_data(ttl=15, show_spinner=False)
def load_live_quote_cached(symbol: str):
    return fetch_live_quote(symbol)

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
st.caption("FMP live-quote workflow with auto-backed long-history crypto research, blind out-of-sample walk-forward diagnostics, explicit guardrails, and conservative research framing.")

with st.sidebar.form("controls"):
    st.subheader("Research Controls")
    st.caption("Default preset: BTC 4H `mean_reversion`, 8 states, auto history provider. The daily lane remains available as context, but it is not a hard veto by default.")
    feature_pack_options = list(list_feature_packs())
    symbol = st.text_input("Symbol", value="BTCUSD").upper()
    interval = st.selectbox("Interval", options=["4hour", "1day", "1hour"], index=0, help=CONTROL_HELP["interval"])
    history_provider = st.selectbox("Historical provider", options=["auto", "fmp", "coinbase", "yahoo"], index=0, help=CONTROL_HELP["provider"])
    default_walk = default_walk_forward_config(interval)
    feature_pack = st.selectbox(
        "Feature pack",
        options=feature_pack_options,
        index=feature_pack_options.index("mean_reversion") if "mean_reversion" in feature_pack_options else 0,
        help=CONTROL_HELP["feature_pack"],
    )
    limit = st.number_input("Bars to fetch", min_value=300, max_value=10000, value=5000, step=100, help=CONTROL_HELP["limit"])
    selected_states = st.select_slider("Selected HMM states", options=[5, 6, 7, 8, 9], value=8, help=CONTROL_HELP["states"])
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
    allow_short = st.checkbox("Allow short trades", value=False, help=CONTROL_HELP["allow_short"])
    require_daily_confirmation = st.checkbox("Require daily confirmation for 4H trades", value=False, help=CONTROL_HELP["require_daily_confirmation"])
    require_consensus_confirmation = st.checkbox("Require consensus confirmation", value=False, help=CONTROL_HELP["require_consensus_confirmation"])
    consensus_gate_mode = st.selectbox(
        "Consensus gate mode",
        options=["hard", "entry_only"],
        index=["hard", "entry_only"].index(StrategyConfig().consensus_gate_mode),
        help=CONTROL_HELP["consensus_gate_mode"],
    )
    consensus_min_share = st.slider("Consensus min share", min_value=0.5, max_value=1.0, value=0.67, step=0.01, help=CONTROL_HELP["consensus_min_share"])
    cost_bps = st.slider("Trading fee (bps)", min_value=0.0, max_value=25.0, value=10.0, step=0.5, help=CONTROL_HELP["cost_bps"])
    spread_bps = st.slider("Spread estimate (bps)", min_value=0.0, max_value=30.0, value=4.0, step=0.5, help=CONTROL_HELP["spread_bps"])
    slippage_bps = st.slider("Slippage estimate (bps)", min_value=0.0, max_value=30.0, value=3.0, step=0.5, help=CONTROL_HELP["slippage_bps"])
    impact_bps = st.slider("Liquidity impact (bps)", min_value=0.0, max_value=20.0, value=2.0, step=0.5, help=CONTROL_HELP["impact_bps"])
    robustness_symbols = st.text_input("Robustness basket", value="BTCUSD,ETHUSD,SOLUSD")
    run_timeframe_check = st.checkbox("Run timeframe comparison (4H / 1D / 1H)", value=True)
    run_feature_pack_check = st.checkbox("Run feature-pack ablation", value=True)
    run_consensus_check = st.checkbox("Run consensus diagnostics (nearby states + seeds)", value=True)
    run_candidate_search_check = st.checkbox("Run candidate search (feature pack / states / shorts / confirmation mode)", value=False)
    candidate_search_max = st.number_input("Candidate search max variants", min_value=4, max_value=80, value=4, step=4)
    auto_adjust_windows = st.checkbox("Auto-size windows if data is shorter than requested", value=True)
    run_clicked = st.form_submit_button("Run Research")

refresh_live_quote = st.sidebar.button("Refresh live quote")
if refresh_live_quote:
    load_live_quote_cached.clear()
    st.rerun()

if run_clicked:
    try:
        with st.spinner("Fetching data, retraining walk-forward folds, and compiling diagnostics..."):
            data_config = DataConfig(symbol=symbol, interval=interval, limit=int(limit), provider=history_provider)
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
                allow_short=allow_short,
                require_daily_confirmation=require_daily_confirmation,
                require_consensus_confirmation=require_consensus_confirmation,
                consensus_min_share=consensus_min_share,
                consensus_gate_mode=consensus_gate_mode,
                cost_bps=cost_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                impact_bps=impact_bps,
            )
            execution_strategy_config = strategy_config
            model_strategy_config = replace(strategy_config, require_daily_confirmation=False, require_consensus_confirmation=False)
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
                confirmation_data_config = DataConfig(symbol=symbol, interval="1day", limit=int(limit), provider=history_provider)
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
                history_provider=history_provider,
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
                    history_provider=history_provider,
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
                    history_provider=history_provider,
                    auto_adjust_windows=auto_adjust_windows,
                )
                if run_feature_pack_check
                else pd.DataFrame()
            )
            consensus_required = run_consensus_check or strategy_config.require_consensus_confirmation
            consensus = (
                run_consensus_diagnostics(
                    symbol=symbol,
                    interval=data_config.interval,
                    limit=int(limit),
                    history_provider=history_provider,
                    feature_columns=feature_columns,
                    model_config=model_config,
                    strategy_config=execution_strategy_config,
                    auto_adjust_windows=auto_adjust_windows,
                )
                if consensus_required
                else None
            )
            consensus_mode_comparison = (
                compare_consensus_gate_modes(
                    selected_result,
                    consensus,
                    interval=data_config.interval,
                    strategy_config=execution_strategy_config,
                )
                if consensus is not None
                else pd.DataFrame()
            )
            if consensus is not None and strategy_config.require_consensus_confirmation:
                selected_result = apply_consensus_confirmation(
                    selected_result,
                    consensus,
                    interval=data_config.interval,
                    strategy_config=execution_strategy_config,
                )
            nested_holdout = nested_holdout_evaluation(
                predictions=selected_result.predictions,
                n_states=selected_states,
                base_config=execution_strategy_config,
                interval=data_config.interval,
                outer_holdout_folds=1,
            )
            nested_holdout_table = nested_holdout_summary_frame(nested_holdout)
            candidate_search_results = (
                run_candidate_search(
                    symbol=symbol,
                    interval=data_config.interval,
                    limit=int(limit),
                    history_provider="coinbase",
                    base_model_config=model_config,
                    base_strategy_config=execution_strategy_config,
                    feature_packs=("mean_reversion", "trend", "baseline", "regime_mix", "atr_causal", "trend_context"),
                    state_counts=(5, 6, 7, 8, 9),
                    short_modes=(False, True),
                    confirmation_modes=("off", "daily", "consensus_entry", "daily_consensus_entry"),
                    robustness_symbols=tuple(parse_symbol_list(robustness_symbols)),
                    auto_adjust_windows=auto_adjust_windows,
                    max_candidates=int(candidate_search_max),
                )
                if run_candidate_search_check
                else pd.DataFrame()
            )
            candidate_search_summary = summarize_candidate_search(candidate_search_results)
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
                consensus_members=consensus.members if consensus is not None else None,
                consensus_timeline=consensus.timeline if consensus is not None else None,
                consensus_summary=consensus.summary if consensus is not None else None,
                consensus_mode_comparison=consensus_mode_comparison,
                nested_holdout_summary=pd.DataFrame([nested_holdout]),
                candidate_search_results=candidate_search_results,
            )
            st.session_state["analysis"] = {
                "data_url": fetched.source_url,
                "data_provider": fetched.provider,
                "data_provider_note": fetched.provider_note,
                "comparison": comparison,
                "selected_result": selected_result,
                "sweep_results": sweep_results,
                "robustness": robustness,
                "timeframe_comparison": timeframe_comparison,
                "feature_pack_comparison": feature_pack_comparison,
                "consensus": consensus,
                "consensus_mode_comparison": consensus_mode_comparison,
                "nested_holdout": nested_holdout,
                "nested_holdout_table": nested_holdout_table,
                "candidate_search_results": candidate_search_results,
                "candidate_search_summary": candidate_search_summary,
                "confirmation_enabled": bool(data_config.interval == "4hour" and strategy_config.require_daily_confirmation),
                "confirmation_summary": selected_result.confirmation_summary,
                "confirmation_result": confirmation_result,
                "confirmation_data_url": confirmation_fetched.source_url if confirmation_fetched is not None else "",
                "confirmation_data_provider": confirmation_fetched.provider if confirmation_fetched is not None else "",
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
            "For crypto, prefer the `auto` historical provider so the app can backfill beyond vendor intraday caps."
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
consensus_mode_comparison = analysis.get("consensus_mode_comparison", pd.DataFrame())
nested_holdout = analysis.get("nested_holdout", {})
nested_holdout_table = analysis.get("nested_holdout_table", pd.DataFrame())
notes = analysis["notes"]
latest_row = selected_result.predictions.iloc[-1]
guardrail_text = latest_row["guardrail_reason"] or "accepted"
live_quote = None
live_quote_error = ""
try:
    live_quote = load_live_quote_cached(analysis["resolved_symbol"])
except Exception as exc:  # pragma: no cover - depends on live vendor behavior
    live_quote_error = str(exc)
execution_plan = build_execution_plan(
    latest_row=latest_row.to_dict(),
    interval=analysis["interval"],
    live_price=float(live_quote.price) if live_quote is not None else None,
)
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
    history_provider=analysis["data_config"].provider,
)
promotion_gates = build_promotion_gate_rows(
    metrics=selected_result.metrics,
    bootstrap=selected_result.bootstrap,
    state_stability=selected_result.state_stability,
    robustness=robustness,
    baseline_comparison=selected_result.baseline_comparison,
    interval=analysis["interval"],
    available_rows=analysis["available_rows"],
    walk_adjusted=analysis["walk_adjusted"],
    fold_count=int(len(selected_result.fold_diagnostics)),
    nested_holdout=nested_holdout,
)
promotion_snapshot = summarize_promotion_gates(promotion_gates)
trust_snapshot = build_trust_snapshot(
    metrics=selected_result.metrics,
    bootstrap=selected_result.bootstrap,
    state_stability=selected_result.state_stability,
    robustness=robustness,
    interval=analysis["interval"],
    available_rows=analysis["available_rows"],
    walk_adjusted=analysis["walk_adjusted"],
)
engine_recommendation = recommend_strategy_engine(
    strategy_metrics=selected_result.metrics,
    baseline_comparison=selected_result.baseline_comparison,
    promotion_summary=promotion_snapshot,
)
candidate_search_results = analysis.get("candidate_search_results", pd.DataFrame())
candidate_search_summary = analysis.get("candidate_search_summary", {})
metric_lookup = metric_interpretation.set_index("metric")

metric_items = [
    ("Action Now", execution_plan["action"], first_sentence(execution_plan["summary"])),
    ("Current State", str(metric_lookup.loc["Current State", "value"]), first_sentence(str(metric_lookup.loc["Current State", "interpretation"]))),
    ("Held Position", str(metric_lookup.loc["Held Position", "value"]), first_sentence(str(metric_lookup.loc["Held Position", "interpretation"]))),
    ("Latest Candidate", str(metric_lookup.loc["Latest Candidate", "value"]), first_sentence(str(metric_lookup.loc["Latest Candidate", "interpretation"]))),
]
if "Daily Confirmation" in metric_lookup.index:
    metric_items.append(
        ("Daily Confirmation", str(metric_lookup.loc["Daily Confirmation", "value"]), first_sentence(str(metric_lookup.loc["Daily Confirmation", "interpretation"])))
    )
if "Consensus Filter" in metric_lookup.index:
    metric_items.append(
        ("Consensus", str(metric_lookup.loc["Consensus Filter", "value"]), first_sentence(str(metric_lookup.loc["Consensus Filter", "interpretation"])))
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

if engine_recommendation["severity"] == "success":
    st.success(f"{engine_recommendation['headline']}: {engine_recommendation['summary']}")
elif engine_recommendation["severity"] == "info":
    st.info(f"{engine_recommendation['headline']}: {engine_recommendation['summary']}")
else:
    st.warning(f"{engine_recommendation['headline']}: {engine_recommendation['summary']}")

if execution_plan["severity"] == "success":
    st.success(f"{execution_plan['action']}: {execution_plan['summary']}")
else:
    st.warning(f"{execution_plan['action']}: {execution_plan['summary']}")
st.caption(execution_plan["entry_guide"])
st.caption(execution_plan["timing_note"])

st.caption("Held Position shows the active book. Latest Candidate shows what the newest bar alone supports before hold and cooldown mechanics are applied.")
if analysis["confirmation_enabled"]:
    st.caption("Daily Confirmation shows whether the slower daily lane currently confirms, blocks, or stays neutral on the 4H trade.")
if analysis["strategy_config"].require_consensus_confirmation:
    st.caption("Consensus Filter shows whether nearby seeds and state counts agree strongly enough with the current direction to allow exposure.")
if analysis["strategy_config"].allow_short:
    st.caption("Shorts are enabled, so bearish validated regimes can surface as `Enter Short` or `Hold Short` instead of only flattening exposure.")
st.caption("Headline metrics are stitched only from blind test windows. Training and validation slices are excluded from performance totals.")
st.caption(
    f"Latest guardrail status: `{guardrail_text}` | Historical provider: `{analysis.get('data_provider', 'n/a')}` | Data request: `{analysis['data_url']}`"
)
if analysis.get("data_provider_note"):
    st.info(f"Historical provider note: {analysis['data_provider_note']}")
if analysis["confirmation_enabled"] and analysis["confirmation_data_url"]:
    st.caption(
        f"Daily confirmation source: `{analysis['confirmation_data_url']}` | Provider: `{analysis.get('confirmation_data_provider') or 'n/a'}`"
    )
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

data_columns = st.columns(6)
data_items = [
    ("Requested Symbol", analysis["symbol"]),
    ("Resolved Symbol", analysis["resolved_symbol"]),
    ("Fetched Rows", f"{analysis['raw_rows']}"),
    ("Usable Rows", f"{analysis['available_rows']}"),
    ("Last Completed Bar", f"{analysis['latest_close']:,.2f}"),
    ("Live Quote", f"{live_quote.price:,.2f}" if live_quote is not None else "unavailable"),
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
if live_quote is not None and live_quote.timestamp is not None:
    quote_time = live_quote.timestamp.tz_convert("America/New_York")
    quote_age_seconds = max((pd.Timestamp.now(tz="America/New_York") - quote_time).total_seconds(), 0.0)
    quote_age_label = f"{quote_age_seconds / 60:.1f} minutes" if quote_age_seconds >= 60 else f"{quote_age_seconds:.0f} seconds"
    st.caption(
        f"Live quote source: `{live_quote.source_url}` | Quote time: `{quote_time.strftime('%Y-%m-%d %H:%M:%S %Z')}` | "
        f"Quote age: `{quote_age_label}` | Exchange: `{live_quote.exchange or 'n/a'}`"
    )
    st.caption(
        f"Signal timing note: the live quote is a market reference, but the model itself only updates on completed `{analysis['interval']}` bars. "
        f"The latest completed model bar ended at `{pd.Timestamp(analysis['feature_end']).strftime('%Y-%m-%d %H:%M')}`."
    )
elif live_quote_error:
    st.warning(f"Live quote fetch was unavailable: {live_quote_error}")

overview_tab, trades_tab, baselines_tab, candidate_tab, interpretation_tab, methodology_tab, confirmation_tab, consensus_tab, timeframe_tab, feature_tab, model_tab, stability_tab, sensitivity_tab, confidence_tab, robustness_tab, notes_tab, export_tab = st.tabs(
    ["Overview", "Trades", "Baselines", "Candidate Search", "Interpretation", "Methodology", "Confirmation", "Consensus", "Timeframes", "Feature Packs", "Model Comparison", "State Stability", "Sensitivity", "Confidence", "Robustness", "Research Notes", "Exports"]
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

with baselines_tab:
    if not selected_result.baseline_comparison.empty:
        best_baseline_row = selected_result.baseline_comparison.iloc[0]
        best_baseline_name = str(best_baseline_row["baseline"])
        best_baseline_sharpe = float(best_baseline_row["sharpe"])
        strategy_sharpe = float(selected_result.metrics.get("sharpe", 0.0))
        if engine_recommendation["engine"] == "baseline":
            st.warning(
                f"Use baseline, not HMM for now. `{best_baseline_name}` is the stronger live reference while the HMM is still research-only."
            )
        elif strategy_sharpe > best_baseline_sharpe:
            st.success(
                f"HMM currently beats the best simple baseline, `{best_baseline_name}`, on Sharpe ({strategy_sharpe:.2f} vs {best_baseline_sharpe:.2f})."
            )
        else:
            st.warning(
                f"Best simple baseline right now is `{best_baseline_name}` with Sharpe {best_baseline_sharpe:.2f}. The HMM needs to beat that bar to justify its complexity."
            )
    st.plotly_chart(plot_baseline_comparison(selected_result.baseline_comparison), use_container_width=True)
    st.dataframe(selected_result.baseline_comparison, use_container_width=True, hide_index=True)
    st.caption("These are simpler reference systems on the same out-of-sample slices, including tougher ATR and daily-trend references. If the HMM cannot beat credible baselines, the added complexity is not earning its keep.")

with candidate_tab:
    if candidate_search_results.empty:
        st.info("Candidate search was not run for this session. Enable it in the sidebar to rank feature pack, state count, shorting mode, and confirmation mode on the deeper history source.")
    else:
        if candidate_search_summary.get("headline"):
            headline = str(candidate_search_summary["headline"])
            summary = str(candidate_search_summary.get("summary", ""))
            if str(candidate_search_summary.get("status", "")) == "keep":
                st.success(f"{headline}: {summary}")
            elif "baseline" in headline.lower():
                st.warning(f"{headline}: {summary}")
            else:
                st.info(f"{headline}: {summary}")
        st.plotly_chart(plot_candidate_search(candidate_search_results), use_container_width=True)
        st.dataframe(candidate_search_results, use_container_width=True, hide_index=True)
        st.caption("This search uses Coinbase-backed history so the ranking stays on one consistent deep intraday source. It includes promotion gates, outer holdout, and robustness, but it is still prioritized and capped rather than globally exhaustive.")

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

with methodology_tab:
    st.subheader("Methodology Summary")
    methodology_rows = pd.DataFrame(
        [
            {
                "component": "Historical Provider",
                "value": str(analysis.get("data_provider", "n/a")).upper(),
                "interpretation": (
                    str(analysis.get("data_provider_note"))
                    if analysis.get("data_provider_note")
                    else "Current run used the requested historical provider without auto-backfill."
                ),
            },
            {
                "component": "Performance Stitching",
                "value": "Blind test windows only",
                "interpretation": "Displayed equity and metrics are stitched only from test slices that were not used in fitting or state labeling.",
            },
            {
                "component": "Walk-Forward Layout",
                "value": (
                    f"train={analysis['walk_config'].train_bars}, purge={analysis['walk_config'].purge_bars}, "
                    f"validate={analysis['walk_config'].validate_bars}, embargo={analysis['walk_config'].embargo_bars}, "
                    f"test={analysis['walk_config'].test_bars}, stride={analysis['walk_config'].refit_stride_bars}"
                ),
                "interpretation": "Each fold retrains on the train slice, labels states on the validate slice, and only then scores the blind test slice.",
            },
            {
                "component": "Auto-Reduced Windows",
                "value": "Yes" if analysis["walk_adjusted"] else "No",
                "interpretation": "If this says `Yes`, the requested windows did not fit the sample and the run is less methodologically strict.",
            },
            {
                "component": "Cost Assumptions",
                "value": (
                    f"fee={analysis['strategy_config'].cost_bps:.1f}bps, spread={analysis['strategy_config'].spread_bps:.1f}bps, "
                    f"slippage={analysis['strategy_config'].slippage_bps:.1f}bps, impact={analysis['strategy_config'].impact_bps:.1f}bps"
                ),
                "interpretation": "Fees, spread, slippage, and liquidity impact are all charged before reporting strategy returns.",
            },
            {
                "component": "Nested Holdout Check",
                "value": str(nested_holdout.get("status", "not_run")).replace("_", " ").title(),
                "interpretation": "The inner folds choose settings from the diagnostic sweep, then the most recent untouched outer folds score those settings afterward.",
            },
            {
                "component": "Current Engine Recommendation",
                "value": engine_recommendation["headline"],
                "interpretation": engine_recommendation["summary"],
            },
        ]
    )
    st.dataframe(methodology_rows, use_container_width=True, hide_index=True)
    fold_schedule = (
        selected_result.predictions.loc[
            :,
            [
                "fold_id",
                "train_start_time",
                "train_end_time",
                "validate_start_time",
                "validate_end_time",
                "test_start_time",
                "test_end_time",
            ],
        ]
        .drop_duplicates(subset=["fold_id"])
        .sort_values("fold_id")
        .reset_index(drop=True)
    )
    st.subheader("Fold Schedule")
    st.dataframe(fold_schedule, use_container_width=True, hide_index=True)
    if promotion_snapshot["severity"] == "success":
        st.success(promotion_snapshot["summary"])
    elif promotion_snapshot["severity"] == "error":
        st.error(promotion_snapshot["summary"])
    else:
        st.warning(promotion_snapshot["summary"])
    st.subheader("Promotion Gates")
    st.dataframe(promotion_gates, use_container_width=True, hide_index=True)
    st.caption("These gates are intentionally stricter than a simple positive Sharpe. A run should clear them before it influences defaults or gets treated as a credible candidate.")
    st.subheader("Nested Holdout Check")
    if nested_holdout.get("status") == "ok":
        st.info(
            "Inner folds selected the sweep settings, and the outer holdout folds then judged them blind. "
            f"Outer holdout Sharpe: {float(nested_holdout.get('outer_holdout_sharpe', 0.0)):.2f}, "
            f"annualized return: {float(nested_holdout.get('outer_holdout_annualized_return', 0.0)):.1%}, "
            f"trades: {int(float(nested_holdout.get('outer_holdout_trades', 0.0)))}."
        )
    else:
        st.warning(
            "Nested holdout confirmation was not fully available for this run. "
            "That usually means there were too few folds to reserve a clean outer slice after inner selection."
        )
    st.dataframe(nested_holdout_table, use_container_width=True, hide_index=True)
    st.caption(
        "This is the closest thing in the app to a holdout-aware sweep check. "
        "Use it to separate 'the sweep found a nice row' from 'that row still worked on untouched later folds.'"
    )

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

with consensus_tab:
    consensus = analysis.get("consensus")
    if consensus is None:
        st.info("Consensus diagnostics were not run for this analysis.")
    else:
        top_left, top_right = st.columns([1.0, 1.6])
        with top_left:
            st.subheader("Consensus Summary")
            st.dataframe(consensus.summary, use_container_width=True, hide_index=True)
            st.caption("Consensus asks whether nearby state counts and random seeds tell the same story. Strong single-run performance with weak consensus is a fragility warning.")
            if not selected_result.consensus_summary.empty:
                st.subheader("Consensus Gate Outcomes")
                st.dataframe(selected_result.consensus_summary, use_container_width=True, hide_index=True)
        with top_right:
            st.plotly_chart(plot_consensus_timeline(consensus.timeline), use_container_width=True)
        st.subheader("Gate Mode Comparison")
        st.plotly_chart(plot_consensus_mode_comparison(consensus_mode_comparison), use_container_width=True)
        if not consensus_mode_comparison.empty:
            st.dataframe(consensus_mode_comparison, use_container_width=True, hide_index=True)
        st.caption("This panel replays the same run three ways: no consensus filter, a hard consensus gate, and an entry-only consensus gate. Use it to see whether consensus is improving trust or just crushing exposure.")
        st.subheader("Consensus Members")
        st.dataframe(consensus.members, use_container_width=True, hide_index=True)

with model_tab:
    st.plotly_chart(plot_model_comparison(comparison), use_container_width=True)
    st.dataframe(comparison, use_container_width=True)

with stability_tab:
    st.plotly_chart(plot_state_stability(selected_result.state_stability), use_container_width=True)
    st.dataframe(selected_result.state_stability, use_container_width=True)
    st.plotly_chart(plot_forward_return_heatmap(selected_result.forward_returns), use_container_width=True)
    st.dataframe(selected_result.forward_returns, use_container_width=True)

with sensitivity_tab:
    st.warning(
        "Parameter sweeps here are diagnostic only. They replay alternative thresholds on the already-observed out-of-sample path, so they help us understand sensitivity but do not qualify as blind model selection."
    )
    if nested_holdout.get("status") == "ok":
        st.info(
            "Nested holdout cross-check: "
            f"inner folds chose posterior {float(nested_holdout.get('selected_inner_posterior_threshold', 0.0)):.2f}, "
            f"min hold {int(float(nested_holdout.get('selected_inner_min_hold_bars', 0.0)))}, "
            f"cooldown {int(float(nested_holdout.get('selected_inner_cooldown_bars', 0.0)))}, "
            f"confirmations {int(float(nested_holdout.get('selected_inner_required_confirmations', 0.0)))}; "
            f"outer holdout Sharpe then came in at {float(nested_holdout.get('outer_holdout_sharpe', 0.0)):.2f}."
        )
    if not sweep_results.empty:
        top_row = sweep_results.iloc[0]
        st.info(
            "Top diagnostic sweep row: "
            f"Sharpe {float(top_row['sharpe']):.2f}, posterior {float(top_row['posterior_threshold']):.2f}, "
            f"min hold {int(top_row['min_hold_bars'])}, cooldown {int(top_row['cooldown_bars'])}, confirmations {int(top_row['required_confirmations'])}. "
            "Do not promote this row without separate holdout confirmation."
        )
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
