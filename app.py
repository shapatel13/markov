from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from markov_regime.config import (
    DataConfig,
    ModelConfig,
    StrategyConfig,
    SweepConfig,
    WalkForwardConfig,
    asset_class_label,
    describe_robustness_basket,
    default_asset_settings,
    default_walk_forward_config,
    infer_asset_class,
)
from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.baselines import (
    baseline_display_name,
    build_baseline_execution_plan,
    describe_live_baseline_universe,
    select_best_baseline_frame,
)
from markov_regime.consensus import apply_consensus_confirmation, compare_consensus_gate_modes, run_consensus_diagnostics
from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.data import fetch_live_quote, fetch_price_data
from markov_regime.features import build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.interpretation import (
    CONTROL_HELP,
    build_control_interpretation_rows,
    build_execution_plan,
    build_hmm_loss_breakdown,
    build_metric_interpretation_rows,
    build_promotion_gate_rows,
    build_trust_snapshot,
    first_sentence,
    recommend_strategy_engine,
    resolve_live_engine_mode,
    summarize_hmm_loss_breakdown,
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
from markov_regime.runtime import AnalysisPlan, resolve_analysis_plan
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


@st.cache_data(ttl=900, show_spinner=False)
def load_price_data_cached(data_config: DataConfig):
    return fetch_price_data(data_config)


@st.cache_data(ttl=900, show_spinner=False)
def build_feature_frame_cached(price_frame: pd.DataFrame, feature_columns: tuple[str, ...]):
    return build_feature_frame(price_frame, feature_columns=feature_columns)


@st.cache_data(ttl=900, show_spinner=False)
def compare_state_counts_cached(
    feature_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    interval: str,
    model_config: ModelConfig,
    walk_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
    state_values: tuple[int, ...],
):
    state_range = range(int(state_values[0]), int(state_values[-1]) + 1)
    return compare_state_counts(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        interval=interval,
        model_config=model_config,
        walk_config=walk_config,
        strategy_config=strategy_config,
        state_range=state_range,
    )


@st.cache_data(ttl=900, show_spinner=False)
def run_multi_asset_robustness_cached(
    symbols: tuple[str, ...],
    interval: str,
    limit: int,
    history_provider: str,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
    walk_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
    auto_adjust_windows: bool,
):
    return run_multi_asset_robustness(
        symbols=symbols,
        interval=interval,
        limit=limit,
        history_provider=history_provider,
        feature_columns=feature_columns,
        model_config=model_config,
        walk_config=walk_config,
        strategy_config=strategy_config,
        auto_adjust_windows=auto_adjust_windows,
    )


@st.cache_data(ttl=900, show_spinner=False)
def run_timeframe_comparison_cached(
    symbol: str,
    limit: int,
    history_provider: str,
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    feature_pack: str,
    feature_columns: tuple[str, ...],
    auto_adjust_windows: bool,
):
    return run_timeframe_comparison(
        symbol=symbol,
        limit=limit,
        history_provider=history_provider,
        model_config=model_config,
        strategy_config=strategy_config,
        feature_pack=feature_pack,
        feature_columns=feature_columns,
        auto_adjust_windows=auto_adjust_windows,
    )


@st.cache_data(ttl=900, show_spinner=False)
def run_feature_pack_comparison_cached(
    price_frame: pd.DataFrame,
    interval: str,
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    symbol: str | None,
    limit: int | None,
    history_provider: str,
    auto_adjust_windows: bool,
):
    return run_feature_pack_comparison(
        price_frame=price_frame,
        interval=interval,
        model_config=model_config,
        strategy_config=strategy_config,
        symbol=symbol,
        limit=limit,
        history_provider=history_provider,
        auto_adjust_windows=auto_adjust_windows,
    )


@st.cache_data(ttl=900, show_spinner=False)
def run_consensus_diagnostics_cached(
    symbol: str,
    interval: str,
    limit: int,
    history_provider: str,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    auto_adjust_windows: bool,
):
    return run_consensus_diagnostics(
        symbol=symbol,
        interval=interval,
        limit=limit,
        history_provider=history_provider,
        feature_columns=feature_columns,
        model_config=model_config,
        strategy_config=strategy_config,
        auto_adjust_windows=auto_adjust_windows,
    )


@st.cache_data(ttl=900, show_spinner=False)
def run_candidate_search_cached(
    symbol: str,
    interval: str,
    limit: int,
    history_provider: str,
    base_model_config: ModelConfig,
    base_strategy_config: StrategyConfig,
    feature_packs: tuple[str, ...],
    state_counts: tuple[int, ...],
    short_modes: tuple[bool, ...],
    confirmation_modes: tuple[str, ...],
    robustness_symbols: tuple[str, ...],
    auto_adjust_windows: bool,
    max_candidates: int,
    seed_robustness_top_k: int,
):
    return run_candidate_search(
        symbol=symbol,
        interval=interval,
        limit=limit,
        history_provider=history_provider,
        base_model_config=base_model_config,
        base_strategy_config=base_strategy_config,
        feature_packs=feature_packs,
        state_counts=state_counts,
        short_modes=short_modes,
        confirmation_modes=confirmation_modes,
        robustness_symbols=robustness_symbols,
        auto_adjust_windows=auto_adjust_windows,
        max_candidates=max_candidates,
        seed_robustness_top_k=seed_robustness_top_k,
    )

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
st.caption("FMP-first workflow with live quotes, blind out-of-sample walk-forward diagnostics, explicit guardrails, and automatic asset-aware defaults for crypto versus stocks and ETFs.")


def _provider_label(option: str) -> str:
    labels = {
        "auto": "auto (FMP primary + deep-history backfill)",
        "fmp": "fmp (Financial Modeling Prep only)",
        "coinbase": "coinbase (deep-history backfill only)",
        "yahoo": "yahoo (fallback only)",
    }
    return labels.get(option, option)


def _engine_mode_label(option: str) -> str:
    labels = {
        "auto": "auto (follow promoted engine)",
        "baseline": "baseline (force strongest simple reference)",
        "hmm_ensemble": "hmm_ensemble (force consensus-filtered HMM)",
        "hmm_research": "hmm_research (force HMM research output)",
    }
    return labels.get(option, option)


def _run_profile_label(option: str) -> str:
    labels = {
        "core_signal": "core signal (fast, recommended)",
        "full_research": "full research (slower diagnostics)",
    }
    return labels.get(option, option)


def _apply_asset_aware_sidebar_defaults(feature_pack_options: list[str]) -> str:
    current_symbol = str(st.session_state.get("symbol_input", "BTCUSD") or "BTCUSD").upper()
    defaults = default_asset_settings(current_symbol)
    marker = f"{current_symbol}|{defaults.asset_class}"

    if st.session_state.get("_asset_defaults_marker") != marker:
        st.session_state["interval_input"] = defaults.interval
        st.session_state["provider_input"] = defaults.provider
        st.session_state["feature_pack_input"] = defaults.feature_pack if defaults.feature_pack in feature_pack_options else feature_pack_options[0]
        st.session_state["limit_input"] = defaults.limit
        st.session_state["cost_bps_input"] = defaults.cost_bps
        st.session_state["spread_bps_input"] = defaults.spread_bps
        st.session_state["slippage_bps_input"] = defaults.slippage_bps
        st.session_state["impact_bps_input"] = defaults.impact_bps
        st.session_state["robustness_symbols_input"] = ",".join(defaults.robustness_symbols)
        st.session_state["require_daily_confirmation_input"] = defaults.require_daily_confirmation
        st.session_state["_asset_defaults_marker"] = marker

    current_interval = str(st.session_state.get("interval_input", defaults.interval))
    walk_marker = f"{marker}|{current_interval}"
    if st.session_state.get("_walk_defaults_marker") != walk_marker:
        walk_defaults = default_walk_forward_config(current_interval, defaults.asset_class)
        st.session_state["train_bars_input"] = walk_defaults.train_bars
        st.session_state["purge_bars_input"] = walk_defaults.purge_bars
        st.session_state["validate_bars_input"] = walk_defaults.validate_bars
        st.session_state["embargo_bars_input"] = walk_defaults.embargo_bars
        st.session_state["test_bars_input"] = walk_defaults.test_bars
        st.session_state["refit_stride_input"] = walk_defaults.refit_stride_bars
        st.session_state["_walk_defaults_marker"] = walk_marker

    st.session_state["_current_asset_class"] = defaults.asset_class
    return defaults.asset_class


engine_mode = st.sidebar.selectbox(
    "Live engine mode",
    options=["auto", "baseline", "hmm_ensemble", "hmm_research"],
    index=0,
    format_func=_engine_mode_label,
    help=CONTROL_HELP["live_engine_mode"],
)
run_profile = st.sidebar.selectbox(
    "Execution profile",
    options=["core_signal", "full_research"],
    index=0,
    format_func=_run_profile_label,
    help="Core signal mode runs only the selected state count and skips optional research diagnostics for speed. Full research mode honors the slower diagnostic toggles.",
)
auto_asset_defaults = st.sidebar.checkbox(
    "Auto asset-aware defaults",
    value=True,
    key="auto_asset_defaults",
    help="When enabled, changing the symbol automatically switches interval, feature pack, cost assumptions, walk-forward defaults, and robustness basket to a crypto-aware or equity-aware profile.",
)

if "symbol_input" not in st.session_state:
    st.session_state["symbol_input"] = "BTCUSD"

feature_pack_options = list(list_feature_packs())
current_asset_class = _apply_asset_aware_sidebar_defaults(feature_pack_options) if auto_asset_defaults else infer_asset_class(str(st.session_state.get("symbol_input", "BTCUSD")))
asset_defaults = default_asset_settings(str(st.session_state.get("symbol_input", "BTCUSD") or "BTCUSD"))
default_basket, default_basket_reason = describe_robustness_basket(str(st.session_state.get("symbol_input", "BTCUSD") or "BTCUSD"), current_asset_class)
st.sidebar.caption(
    f"Detected asset class: {asset_class_label(current_asset_class)}. "
    f"Auto profile favors `{asset_defaults.interval}` with `{asset_defaults.feature_pack}` and robustness basket `{', '.join(default_basket)}`."
)
st.sidebar.caption(default_basket_reason)

with st.sidebar.form("controls"):
    st.subheader("Research Controls")
    st.caption("The sidebar now auto-detects crypto versus equities and reseeds interval, feature pack, costs, and robustness defaults when the symbol changes.")
    st.caption(f"Execution profile: `{_run_profile_label(run_profile)}`")
    symbol = st.text_input("Symbol", key="symbol_input").upper()
    interval = st.selectbox("Interval", options=["4hour", "1day", "1hour"], help=CONTROL_HELP["interval"], key="interval_input")
    history_provider = st.selectbox(
        "Historical provider",
        options=["auto", "fmp", "coinbase", "yahoo"],
        format_func=_provider_label,
        help=CONTROL_HELP["provider"],
        key="provider_input",
    )
    feature_pack = st.selectbox(
        "Feature pack",
        options=feature_pack_options,
        help=CONTROL_HELP["feature_pack"],
        key="feature_pack_input",
    )
    limit = st.number_input("Bars to fetch", min_value=300, max_value=10000, step=100, help=CONTROL_HELP["limit"], key="limit_input")
    selected_states = st.select_slider("Selected HMM states", options=[5, 6, 7, 8, 9], value=8, help=CONTROL_HELP["states"])
    train_bars = st.number_input("Train bars", min_value=120, max_value=5000, step=12, help=CONTROL_HELP["train_bars"], key="train_bars_input")
    purge_bars = st.number_input("Purge bars", min_value=0, max_value=240, step=1, help=CONTROL_HELP["purge_bars"], key="purge_bars_input")
    validate_bars = st.number_input("Validate bars", min_value=24, max_value=1500, step=12, help=CONTROL_HELP["validate_bars"], key="validate_bars_input")
    embargo_bars = st.number_input("Embargo bars", min_value=0, max_value=240, step=1, help=CONTROL_HELP["embargo_bars"], key="embargo_bars_input")
    test_bars = st.number_input("Test bars", min_value=24, max_value=1500, step=12, help=CONTROL_HELP["test_bars"], key="test_bars_input")
    refit_stride = st.number_input("Refit stride", min_value=24, max_value=1500, step=12, help=CONTROL_HELP["refit_stride"], key="refit_stride_input")
    posterior_threshold = st.slider("Posterior threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.01, help=CONTROL_HELP["posterior_threshold"])
    min_hold_bars = st.slider("Min hold bars", min_value=1, max_value=24, value=6, help=CONTROL_HELP["min_hold_bars"])
    cooldown_bars = st.slider("Cooldown bars", min_value=0, max_value=24, value=4, help=CONTROL_HELP["cooldown_bars"])
    required_confirmations = st.slider("Required confirmations", min_value=1, max_value=6, value=2, help=CONTROL_HELP["required_confirmations"])
    confidence_gap = st.slider("Top-two posterior gap", min_value=0.0, max_value=0.25, value=0.06, step=0.01, help=CONTROL_HELP["confidence_gap"])
    allow_short = st.checkbox("Allow short trades", value=False, help=CONTROL_HELP["allow_short"])
    require_daily_confirmation = st.checkbox("Require daily confirmation for 4H trades", help=CONTROL_HELP["require_daily_confirmation"], key="require_daily_confirmation_input")
    require_consensus_confirmation = st.checkbox("Require consensus confirmation", value=False, help=CONTROL_HELP["require_consensus_confirmation"])
    consensus_gate_mode = st.selectbox(
        "Consensus gate mode",
        options=["hard", "entry_only"],
        index=["hard", "entry_only"].index(StrategyConfig().consensus_gate_mode),
        help=CONTROL_HELP["consensus_gate_mode"],
    )
    consensus_min_share = st.slider("Consensus min share", min_value=0.5, max_value=1.0, value=0.67, step=0.01, help=CONTROL_HELP["consensus_min_share"])
    cost_bps = st.slider("Trading fee (bps)", min_value=0.0, max_value=25.0, step=0.5, help=CONTROL_HELP["cost_bps"], key="cost_bps_input")
    spread_bps = st.slider("Spread estimate (bps)", min_value=0.0, max_value=30.0, step=0.5, help=CONTROL_HELP["spread_bps"], key="spread_bps_input")
    slippage_bps = st.slider("Slippage estimate (bps)", min_value=0.0, max_value=30.0, step=0.5, help=CONTROL_HELP["slippage_bps"], key="slippage_bps_input")
    impact_bps = st.slider("Liquidity impact (bps)", min_value=0.0, max_value=20.0, step=0.5, help=CONTROL_HELP["impact_bps"], key="impact_bps_input")
    robustness_symbols = st.text_input("Robustness basket", key="robustness_symbols_input")
    st.markdown("**Advanced Diagnostics**")
    st.caption("These are the slower desk-style extras. Core signal mode skips them automatically unless the chosen engine or guardrails require consensus.")
    run_model_comparison_check = st.checkbox("Run 5-9 state comparison", value=False)
    run_robustness_check = st.checkbox("Run robustness basket", value=False)
    run_timeframe_check = st.checkbox("Run timeframe comparison (4H / 1D / 1H)", value=False)
    run_feature_pack_check = st.checkbox("Run feature-pack ablation", value=False)
    run_consensus_check = st.checkbox("Run consensus diagnostics (nearby states + seeds)", value=False)
    run_candidate_search_check = st.checkbox("Run candidate search (feature pack / states / shorts / confirmation mode)", value=False)
    candidate_search_max = st.number_input("Candidate search max variants", min_value=4, max_value=80, value=4, step=4)
    auto_adjust_windows = st.checkbox("Auto-size windows if data is shorter than requested", value=True)
    run_clicked = st.form_submit_button("Run Research")

refresh_live_quote = st.sidebar.button("Refresh live quote")
if refresh_live_quote:
    load_live_quote_cached.clear()
    st.rerun()

if run_clicked:
    run_status = None
    try:
        with st.spinner("Fetching data, retraining walk-forward folds, and compiling diagnostics..."):
            data_config = DataConfig(symbol=symbol, interval=interval, limit=int(limit), provider=history_provider)
            current_asset_class = infer_asset_class(symbol)
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
            plan = resolve_analysis_plan(
                profile=run_profile,
                selected_states=int(selected_states),
                run_model_comparison=run_model_comparison_check,
                run_robustness=run_robustness_check,
                run_timeframe_comparison=run_timeframe_check,
                run_feature_pack_comparison=run_feature_pack_check,
                run_consensus_diagnostics=run_consensus_check,
                run_candidate_search=run_candidate_search_check,
                require_consensus_confirmation=strategy_config.require_consensus_confirmation,
                engine_mode=engine_mode,
            )
            effective_robustness_symbols = tuple(parse_symbol_list(robustness_symbols))
            default_run_basket, default_run_basket_reason = describe_robustness_basket(symbol, current_asset_class)
            robustness_basket_note = (
                default_run_basket_reason
                if effective_robustness_symbols == default_run_basket
                else "Custom robustness basket override is active for this run, so the app is using your manual symbol list instead of the default asset-aware peer map."
            )
            execution_strategy_config = strategy_config
            model_strategy_config = replace(strategy_config, require_daily_confirmation=False, require_consensus_confirmation=False)
            total_stages = 5
            total_stages += 1 if data_config.interval == "4hour" and strategy_config.require_daily_confirmation else 0
            total_stages += 1 if plan.run_robustness else 0
            total_stages += 1 if plan.run_timeframe_comparison else 0
            total_stages += 1 if plan.run_feature_pack_comparison else 0
            total_stages += 1 if plan.run_consensus_diagnostics else 0
            total_stages += 1 if plan.run_candidate_search else 0
            run_status = st.status("Running analysis...", expanded=True)
            progress_bar = st.progress(0.0, text="Preparing run...")
            stage_state = {"completed": 0}
            stage_timings: list[dict[str, float | str]] = []

            def begin_stage(label: str) -> float:
                progress_bar.progress(stage_state["completed"] / total_stages, text=label)
                run_status.write(f"Running: {label}")
                return perf_counter()

            def finish_stage(label: str, started_at: float) -> None:
                elapsed = perf_counter() - started_at
                stage_timings.append({"stage": label, "seconds": round(elapsed, 2)})
                stage_state["completed"] += 1
                progress_bar.progress(
                    min(stage_state["completed"] / total_stages, 1.0),
                    text=f"Completed: {label} ({elapsed:.1f}s)",
                )

            started = begin_stage("Fetch historical data")
            fetched = load_price_data_cached(data_config)
            feature_frame = build_feature_frame_cached(fetched.frame, feature_columns=feature_columns)
            finish_stage("Fetch historical data", started)
            walk_config, was_adjusted = (
                suggest_walk_forward_config(len(feature_frame), requested_walk_config)
                if auto_adjust_windows
                else (requested_walk_config, False)
            )
            started = begin_stage("Run primary HMM walk-forward")
            comparison, results_by_state = compare_state_counts_cached(
                feature_frame=feature_frame,
                feature_columns=feature_columns,
                interval=data_config.interval,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=model_strategy_config,
                state_values=plan.state_values,
            )
            finish_stage("Run primary HMM walk-forward", started)
            confirmation_fetched = None
            confirmation_result = None
            if data_config.interval == "4hour" and strategy_config.require_daily_confirmation:
                started = begin_stage("Apply daily confirmation overlay")
                confirmation_data_config = DataConfig(symbol=symbol, interval="1day", limit=int(limit), provider=history_provider)
                confirmation_fetched = load_price_data_cached(confirmation_data_config)
                confirmation_feature_frame = build_feature_frame_cached(confirmation_fetched.frame, feature_columns=feature_columns)
                confirmation_walk_config, _ = (
                    suggest_walk_forward_config(len(confirmation_feature_frame), default_walk_forward_config("1day", current_asset_class))
                    if auto_adjust_windows
                    else (default_walk_forward_config("1day", current_asset_class), False)
                )
                _, confirmation_results_by_state = compare_state_counts_cached(
                    feature_frame=confirmation_feature_frame,
                    feature_columns=feature_columns,
                    interval="1day",
                    model_config=model_config,
                    walk_config=confirmation_walk_config,
                    strategy_config=model_strategy_config,
                    state_values=plan.state_values,
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
                finish_stage("Apply daily confirmation overlay", started)
            raw_hmm_result = results_by_state[selected_states]
            selected_result = raw_hmm_result
            started = begin_stage("Replay parameter sweep")
            sweep_results = parameter_sweep(
                predictions=selected_result.predictions,
                n_states=selected_states,
                base_config=execution_strategy_config,
                sweep_config=SweepConfig(),
                interval=data_config.interval,
            )
            finish_stage("Replay parameter sweep", started)
            if plan.run_robustness:
                started = begin_stage("Run robustness basket")
                robustness = run_multi_asset_robustness_cached(
                    symbols=effective_robustness_symbols,
                    interval=data_config.interval,
                    limit=int(limit),
                    history_provider=history_provider,
                    feature_columns=feature_columns,
                    model_config=model_config,
                    walk_config=walk_config,
                    strategy_config=execution_strategy_config,
                    auto_adjust_windows=auto_adjust_windows,
                )
                finish_stage("Run robustness basket", started)
            else:
                robustness = pd.DataFrame()
            if plan.run_timeframe_comparison:
                started = begin_stage("Run timeframe comparison")
                timeframe_comparison = run_timeframe_comparison_cached(
                    symbol=symbol,
                    limit=int(limit),
                    history_provider=history_provider,
                    model_config=model_config,
                    strategy_config=execution_strategy_config,
                    feature_pack=feature_pack,
                    feature_columns=feature_columns,
                    auto_adjust_windows=auto_adjust_windows,
                )
                finish_stage("Run timeframe comparison", started)
            else:
                timeframe_comparison = pd.DataFrame()

            if plan.run_feature_pack_comparison:
                started = begin_stage("Run feature-pack ablation")
                feature_pack_comparison = run_feature_pack_comparison_cached(
                    price_frame=fetched.frame,
                    interval=data_config.interval,
                    model_config=model_config,
                    strategy_config=execution_strategy_config,
                    symbol=symbol,
                    limit=int(limit),
                    history_provider=history_provider,
                    auto_adjust_windows=auto_adjust_windows,
                )
                finish_stage("Run feature-pack ablation", started)
            else:
                feature_pack_comparison = pd.DataFrame()

            if plan.run_consensus_diagnostics:
                started = begin_stage("Run consensus diagnostics")
                consensus = run_consensus_diagnostics_cached(
                    symbol=symbol,
                    interval=data_config.interval,
                    limit=int(limit),
                    history_provider=history_provider,
                    feature_columns=feature_columns,
                    model_config=model_config,
                    strategy_config=execution_strategy_config,
                    auto_adjust_windows=auto_adjust_windows,
                )
                finish_stage("Run consensus diagnostics", started)
            else:
                consensus = None
            ensemble_result = None
            if consensus is not None:
                ensemble_strategy_config = replace(
                    execution_strategy_config,
                    require_consensus_confirmation=True,
                    consensus_gate_mode="entry_only",
                )
                ensemble_result = apply_consensus_confirmation(
                    raw_hmm_result,
                    consensus,
                    interval=data_config.interval,
                    strategy_config=ensemble_strategy_config,
                )
            consensus_mode_comparison = (
                compare_consensus_gate_modes(
                    raw_hmm_result,
                    consensus,
                    interval=data_config.interval,
                    strategy_config=execution_strategy_config,
                )
                if consensus is not None
                else pd.DataFrame()
            )
            if consensus is not None and strategy_config.require_consensus_confirmation:
                selected_result = apply_consensus_confirmation(
                    raw_hmm_result,
                    consensus,
                    interval=data_config.interval,
                    strategy_config=execution_strategy_config,
                )
            started = begin_stage("Evaluate nested holdout")
            nested_holdout = nested_holdout_evaluation(
                predictions=selected_result.predictions,
                n_states=selected_states,
                base_config=execution_strategy_config,
                interval=data_config.interval,
                outer_holdout_folds=1,
            )
            nested_holdout_table = nested_holdout_summary_frame(nested_holdout)
            finish_stage("Evaluate nested holdout", started)
            if plan.run_candidate_search:
                stage_started = begin_stage("Run candidate search")
                candidate_search_results = run_candidate_search_cached(
                    symbol=symbol,
                    interval=data_config.interval,
                    limit=int(limit),
                    history_provider="auto",
                    base_model_config=model_config,
                    base_strategy_config=execution_strategy_config,
                    feature_packs=("mean_reversion", "trend", "baseline", "regime_mix", "atr_causal", "trend_context"),
                    state_counts=(5, 6, 7, 8, 9),
                    short_modes=(False, True),
                    confirmation_modes=("off", "daily", "consensus_entry", "daily_consensus_entry"),
                    robustness_symbols=effective_robustness_symbols,
                    auto_adjust_windows=auto_adjust_windows,
                    max_candidates=int(candidate_search_max),
                    seed_robustness_top_k=min(2, int(candidate_search_max)),
                )
                finish_stage("Run candidate search", stage_started)
            else:
                candidate_search_results = pd.DataFrame()
            candidate_search_summary = summarize_candidate_search(candidate_search_results)
            notes = build_research_notes(selected_result, comparison)
            started = begin_stage("Write artifacts")
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
            finish_stage("Write artifacts", started)
            run_status.update(label="Analysis complete", state="complete")
            st.session_state["analysis"] = {
                "data_url": fetched.source_url,
                "data_provider": fetched.provider,
                "data_provider_note": fetched.provider_note,
                "comparison": comparison,
                "selected_result": selected_result,
                "raw_hmm_result": raw_hmm_result,
                "ensemble_result": ensemble_result,
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
                "robustness_symbols": effective_robustness_symbols,
                "robustness_basket_note": robustness_basket_note,
                "analysis_plan": plan,
                "stage_timings": pd.DataFrame(stage_timings),
                "notes": notes,
                "symbol": symbol,
                "resolved_symbol": fetched.resolved_symbol,
                "asset_class": current_asset_class,
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
        if run_status is not None:
            run_status.update(label="Analysis failed", state="error")
        st.session_state.pop("analysis", None)
        st.error(str(exc))
        st.info(
            "Try a larger history, a higher timeframe like `4hour` or `1day`, or enable automatic window sizing. "
            "For crypto, prefer the `auto` historical provider so the app keeps FMP primary and only backfills deeper history when needed."
        )
        st.stop()
    except Exception as exc:  # pragma: no cover - UI safety net
        if run_status is not None:
            run_status.update(label="Analysis failed", state="error")
        st.session_state.pop("analysis", None)
        st.error(f"Unexpected analysis failure: {exc}")
        st.info("Try `core signal` mode first. If it still fails, disable the slower diagnostics and rerun so we can isolate the step that is breaking.")
        st.stop()

analysis = st.session_state.get("analysis")
if not analysis:
    st.info("Choose a symbol and run the research panel to generate walk-forward diagnostics.")
    st.stop()

comparison = analysis["comparison"]
selected_result = analysis["selected_result"]
raw_hmm_result = analysis.get("raw_hmm_result", selected_result)
ensemble_result = analysis.get("ensemble_result")
sweep_results = analysis["sweep_results"]
robustness = analysis["robustness"]
timeframe_comparison = analysis["timeframe_comparison"]
feature_pack_comparison = analysis["feature_pack_comparison"]
consensus_mode_comparison = analysis.get("consensus_mode_comparison", pd.DataFrame())
nested_holdout = analysis.get("nested_holdout", {})
nested_holdout_table = analysis.get("nested_holdout_table", pd.DataFrame())
notes = analysis["notes"]
latest_row = selected_result.predictions.iloc[-1]
raw_hmm_latest_row = raw_hmm_result.predictions.iloc[-1]
ensemble_latest_row = ensemble_result.predictions.iloc[-1] if ensemble_result is not None and not ensemble_result.predictions.empty else None
guardrail_text = latest_row["guardrail_reason"] or "accepted"
live_quote = None
live_quote_error = ""
try:
    live_quote = load_live_quote_cached(analysis["resolved_symbol"])
except Exception as exc:  # pragma: no cover - depends on live vendor behavior
    live_quote_error = str(exc)
execution_plan = build_execution_plan(
    latest_row=raw_hmm_latest_row.to_dict(),
    interval=analysis["interval"],
    live_price=float(live_quote.price) if live_quote is not None else None,
)
ensemble_execution_plan = build_execution_plan(
    latest_row=ensemble_latest_row.to_dict() if ensemble_latest_row is not None else raw_hmm_latest_row.to_dict(),
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
    asset_class=analysis["asset_class"],
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
    asset_class=analysis["asset_class"],
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
    asset_class=analysis["asset_class"],
)
engine_recommendation = recommend_strategy_engine(
    strategy_metrics=selected_result.metrics,
    baseline_comparison=selected_result.baseline_comparison,
    promotion_summary=promotion_snapshot,
)
best_baseline_name, best_baseline_row, best_baseline_frame = select_best_baseline_frame(
    selected_result.predictions,
    analysis["interval"],
    analysis["strategy_config"],
    selected_result.baseline_comparison,
    asset_class=analysis["asset_class"],
)
live_engine = resolve_live_engine_mode(
    requested_mode=engine_mode,
    engine_recommendation=engine_recommendation,
    best_baseline=best_baseline_name,
    consensus_available=ensemble_result is not None,
)
baseline_execution_plan = (
    build_baseline_execution_plan(
        baseline_frame=best_baseline_frame,
        baseline_name=best_baseline_name,
        interval=analysis["interval"],
        live_price=float(live_quote.price) if live_quote is not None else None,
    )
    if best_baseline_name is not None
    else None
)
hmm_loss_breakdown = build_hmm_loss_breakdown(
    strategy_metrics=selected_result.metrics,
    ensemble_metrics=ensemble_result.metrics if ensemble_result is not None else None,
    baseline_row=best_baseline_row.to_dict() if not best_baseline_row.empty else None,
    promotion_gates=promotion_gates,
    nested_holdout=nested_holdout,
    robustness=robustness,
    bootstrap=selected_result.bootstrap,
)
hmm_loss_summary = summarize_hmm_loss_breakdown(hmm_loss_breakdown)

if live_engine["engine"] == "baseline" and baseline_execution_plan is not None:
    active_execution_plan = baseline_execution_plan
    active_engine_name = baseline_execution_plan["engine_label"]
    active_held_position = baseline_execution_plan["held_position"]
    active_sharpe = float(best_baseline_row.get("sharpe", 0.0))
    active_annualized_return = float(best_baseline_row.get("annualized_return", 0.0))
elif live_engine["engine"] == "hmm":
    active_execution_plan = execution_plan
    active_engine_name = f"HMM ({analysis['feature_pack']}, {analysis['model_config'].n_states} states)"
    active_held_position = {1: "Long", 0: "Flat", -1: "Short"}.get(int(raw_hmm_latest_row.get("signal_position", 0)), "Flat")
    active_sharpe = float(raw_hmm_result.metrics.get("sharpe", 0.0))
    active_annualized_return = float(raw_hmm_result.metrics.get("annualized_return", 0.0))
elif live_engine["engine"] == "hmm_ensemble" and ensemble_result is not None:
    active_execution_plan = ensemble_execution_plan
    active_engine_name = f"HMM Ensemble ({analysis['feature_pack']}, {analysis['model_config'].n_states} states)"
    active_held_position = {1: "Long", 0: "Flat", -1: "Short"}.get(int(ensemble_latest_row.get("signal_position", 0)), "Flat")
    active_sharpe = float(ensemble_result.metrics.get("sharpe", 0.0))
    active_annualized_return = float(ensemble_result.metrics.get("annualized_return", 0.0))
else:
    active_execution_plan = {
        "action": "No Active Deployment",
        "severity": "warning",
        "summary": "Neither the HMM nor the simple baselines are strong enough to justify a live recommendation right now.",
        "entry_guide": "Stay flat until either the promoted baseline or the HMM clears the current gates.",
        "timing_note": "The app is intentionally preferring capital preservation over a marginal signal.",
        "held_position": "Flat",
        "engine_label": "Cash / No Deployment",
    }
    active_engine_name = "Cash / No Deployment"
    active_held_position = "Flat"
    candidate_sharpes = [float(selected_result.metrics.get("sharpe", 0.0))]
    if not best_baseline_row.empty:
        candidate_sharpes.append(float(best_baseline_row.get("sharpe", 0.0)))
    active_sharpe = max(candidate_sharpes)
    active_annualized_return = 0.0

candidate_search_results = analysis.get("candidate_search_results", pd.DataFrame())
candidate_search_summary = analysis.get("candidate_search_summary", {})
live_baseline_universe_note = describe_live_baseline_universe(analysis["asset_class"], analysis["interval"])
analysis_plan = analysis.get("analysis_plan")
stage_timings = analysis.get("stage_timings", pd.DataFrame())
metric_lookup = metric_interpretation.set_index("metric")

live_metric_items = [
    ("Live Engine", active_engine_name, first_sentence(live_engine["summary"])),
    ("Action Now", active_execution_plan["action"], first_sentence(active_execution_plan["summary"])),
    ("Held Position", active_held_position, "This is the position the selected live engine is currently carrying."),
    ("Engine Sharpe", f"{active_sharpe:.2f}", "This comes from the currently selected live engine, not always the HMM."),
    ("Ann. Return", f"{active_annualized_return:.1%}", "Annualized return for the selected live engine on the stitched blind-OOS sample."),
]
live_metrics_columns = st.columns(len(live_metric_items))
for column, (label, value, note) in zip(live_metrics_columns, live_metric_items, strict=True):
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

st.info(f"{live_engine['headline']}: {live_engine['summary']}")

if active_execution_plan["severity"] == "success":
    st.success(f"{active_execution_plan['action']}: {active_execution_plan['summary']}")
else:
    st.warning(f"{active_execution_plan['action']}: {active_execution_plan['summary']}")
st.caption(active_execution_plan["entry_guide"])
st.caption(active_execution_plan["timing_note"])

if live_engine["engine"] == "baseline" and best_baseline_name is not None:
    st.caption(
        f"Live execution is currently following the promoted baseline `{best_baseline_name}`. "
        "The HMM cards below remain available as research diagnostics."
    )
elif live_engine["engine"] == "hmm":
    st.caption("Live execution is currently following the raw HMM research engine. Treat this as the most direct model read, not the most conservative HMM deployment path.")
elif live_engine["engine"] == "hmm_ensemble":
    st.caption("Live execution is currently following the consensus-filtered HMM ensemble. This path only acts when nearby seeds and state counts broadly agree with the trade direction.")
else:
    st.caption("Live execution is intentionally flat because no engine is strong enough to justify deployment on this run.")

engine_comparison_rows = [
    {
        "engine": "HMM Research",
        "active": live_engine["engine"] == "hmm",
        "action": execution_plan["action"],
        "held_position": {1: "Long", 0: "Flat", -1: "Short"}.get(int(raw_hmm_latest_row.get("signal_position", 0)), "Flat"),
        "sharpe": float(raw_hmm_result.metrics.get("sharpe", 0.0)),
        "annualized_return": float(raw_hmm_result.metrics.get("annualized_return", 0.0)),
        "trades": float(raw_hmm_result.metrics.get("trades", 0.0)),
        "latest_guardrail": str(raw_hmm_latest_row.get("guardrail_reason", "") or "accepted"),
        "notes": "Direct HMM output without consensus gating.",
    },
    {
        "engine": "HMM Ensemble",
        "active": live_engine["engine"] == "hmm_ensemble",
        "action": ensemble_execution_plan["action"] if ensemble_result is not None else "Unavailable",
        "held_position": (
            {1: "Long", 0: "Flat", -1: "Short"}.get(int(ensemble_latest_row.get("signal_position", 0)), "Flat")
            if ensemble_latest_row is not None
            else "n/a"
        ),
        "sharpe": float(ensemble_result.metrics.get("sharpe", 0.0)) if ensemble_result is not None else float("nan"),
        "annualized_return": float(ensemble_result.metrics.get("annualized_return", 0.0)) if ensemble_result is not None else float("nan"),
        "trades": float(ensemble_result.metrics.get("trades", 0.0)) if ensemble_result is not None else float("nan"),
        "latest_guardrail": str(ensemble_latest_row.get("guardrail_reason", "") or "accepted") if ensemble_latest_row is not None else "consensus_not_available",
        "notes": "Consensus-filtered HMM that only acts when nearby seeds and state counts broadly agree." if ensemble_result is not None else "Run consensus diagnostics to evaluate the ensemble path.",
    },
]
if best_baseline_name is not None and baseline_execution_plan is not None:
    engine_comparison_rows.append(
        {
            "engine": f"Baseline ({baseline_display_name(best_baseline_name)})",
            "active": live_engine["engine"] == "baseline",
            "action": baseline_execution_plan["action"],
            "held_position": baseline_execution_plan["held_position"],
            "sharpe": float(best_baseline_row.get("sharpe", 0.0)),
            "annualized_return": float(best_baseline_row.get("annualized_return", 0.0)),
            "trades": float(best_baseline_row.get("trades", 0.0)),
            "latest_guardrail": "baseline_rule",
            "notes": "Strongest simple reference inside the asset-aware live baseline set on the same blind-OOS slices.",
        }
    )
engine_comparison = pd.DataFrame(engine_comparison_rows)
st.subheader("Engine Comparison")
st.dataframe(engine_comparison, use_container_width=True, hide_index=True)
st.caption("This panel keeps the live choice honest by showing the raw HMM, the consensus-filtered HMM ensemble, and the strongest simple baseline side by side on the same stitched blind-OOS run.")
if engine_recommendation["engine"] != "hmm":
    if hmm_loss_summary["severity"] == "success":
        st.success(f"{hmm_loss_summary['headline']}: {hmm_loss_summary['summary']}")
    elif hmm_loss_summary["severity"] == "warning":
        st.warning(f"{hmm_loss_summary['headline']}: {hmm_loss_summary['summary']}")
    else:
        st.info(f"{hmm_loss_summary['headline']}: {hmm_loss_summary['summary']}")
    st.dataframe(hmm_loss_breakdown, use_container_width=True, hide_index=True)
    st.caption("This breakdown explains why the app routed the live seat away from the raw HMM on this run.")

research_metric_items = [
    ("HMM State", str(metric_lookup.loc["Current State", "value"]), first_sentence(str(metric_lookup.loc["Current State", "interpretation"]))),
    ("HMM Held", str(metric_lookup.loc["Held Position", "value"]), first_sentence(str(metric_lookup.loc["Held Position", "interpretation"]))),
    ("HMM Candidate", str(metric_lookup.loc["Latest Candidate", "value"]), first_sentence(str(metric_lookup.loc["Latest Candidate", "interpretation"]))),
]
if "Daily Confirmation" in metric_lookup.index:
    research_metric_items.append(
        ("Daily Confirmation", str(metric_lookup.loc["Daily Confirmation", "value"]), first_sentence(str(metric_lookup.loc["Daily Confirmation", "interpretation"])))
    )
if "Consensus Filter" in metric_lookup.index:
    research_metric_items.append(
        ("Consensus", str(metric_lookup.loc["Consensus Filter", "value"]), first_sentence(str(metric_lookup.loc["Consensus Filter", "interpretation"])))
    )
research_metric_items.extend(
    [
        ("HMM Posterior", str(metric_lookup.loc["Posterior", "value"]), first_sentence(str(metric_lookup.loc["Posterior", "interpretation"]))),
        ("HMM Sharpe", str(metric_lookup.loc["Sharpe", "value"]), first_sentence(str(metric_lookup.loc["Sharpe", "interpretation"]))),
        ("HMM Ann. Return", str(metric_lookup.loc["Annualized Return", "value"]), first_sentence(str(metric_lookup.loc["Annualized Return", "interpretation"]))),
    ]
)
st.caption("HMM research diagnostics below show the selected HMM research path for this run, even when the selected live engine is a simpler promoted baseline or the ensemble-filtered HMM.")
research_columns = st.columns(len(research_metric_items))
for column, (label, value, note) in zip(research_columns, research_metric_items, strict=True):
    column.markdown(
        (
            "<div class='metric-card'>"
            f"<strong>{label}</strong><br><span style='font-size:1.35rem'>{value}</span>"
            f"<br><span style='font-size:0.82rem;color:#475569'>{note}</span></div>"
        ),
        unsafe_allow_html=True,
    )

st.caption("Headline metrics are stitched only from blind test windows. Training and validation slices are excluded from performance totals.")
st.caption(
    f"Latest guardrail status: `{guardrail_text}` | Historical provider: `{_provider_label(str(analysis.get('data_provider', 'n/a')))}` | Data request: `{analysis['data_url']}`"
)
st.caption(
    f"Detected asset class: `{asset_class_label(str(analysis.get('asset_class', 'crypto')) if analysis.get('asset_class') else 'crypto')}` "
    f"| Annualization basis: `{int(float(selected_result.metrics.get('annualization_bars_per_year', 0.0)))} bars/year`"
)
if analysis.get("data_provider_note"):
    st.info(f"Historical provider note: {analysis['data_provider_note']}")
if selected_result.converged_ratio < 1.0:
    st.warning(
        f"HMM convergence quality was {selected_result.converged_ratio:.0%} across walk-forward folds. "
        "A non-converged fold does not automatically invalidate the run, but it does make the regime fit less trustworthy."
    )
optimizer_warning_count = 0
optimizer_warning_folds = 0
optimizer_warning_excerpt = ""
if "optimizer_warning_count" in selected_result.fold_diagnostics.columns:
    optimizer_warning_count = int(selected_result.fold_diagnostics["optimizer_warning_count"].fillna(0).sum())
    optimizer_warning_folds = int((selected_result.fold_diagnostics["optimizer_warning_count"].fillna(0) > 0).sum())
    warning_texts = (
        selected_result.fold_diagnostics.get("optimizer_warning_text", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
    )
    non_empty_warning_texts = [text for text in warning_texts.tolist() if text.strip()]
    optimizer_warning_excerpt = non_empty_warning_texts[0] if non_empty_warning_texts else ""
if optimizer_warning_count > 0:
    st.warning(
        f"HMM optimizer warnings appeared in {optimizer_warning_folds} fold(s) ({optimizer_warning_count} total warning lines). "
        "The warnings are now captured into fold diagnostics instead of printing raw optimizer spam into the terminal."
    )
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
        metric_sources: dict[str, pd.Series] = {
            "hmm_strategy": pd.Series(selected_result.metrics),
            "buy_and_hold": pd.Series(selected_result.benchmark_metrics),
        }
        if not best_baseline_row.empty:
            metric_sources[f"best_baseline ({best_baseline_name})"] = pd.Series(best_baseline_row.to_dict())
        metric_frame = pd.DataFrame(metric_sources)
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
    st.info(live_baseline_universe_note)
    if not selected_result.baseline_comparison.empty:
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
        if baseline_execution_plan is not None:
            baseline_cards = st.columns(4)
            baseline_card_items = [
                ("Baseline Engine", baseline_display_name(best_baseline_name), "This is the strongest simple reference inside the asset-aware live baseline set for this symbol."),
                ("Baseline Action", baseline_execution_plan["action"], first_sentence(baseline_execution_plan["summary"])),
                ("Baseline Held", baseline_execution_plan["held_position"], "This is the position the baseline itself is currently carrying."),
                ("Baseline Sharpe", f"{best_baseline_sharpe:.2f}", "Current Sharpe for the strongest simple baseline on the same out-of-sample slices."),
            ]
            for column, (label, value, note) in zip(baseline_cards, baseline_card_items, strict=True):
                column.markdown(
                    (
                        "<div class='metric-card'>"
                        f"<strong>{label}</strong><br><span style='font-size:1.2rem'>{value}</span>"
                        f"<br><span style='font-size:0.82rem;color:#475569'>{note}</span></div>"
                    ),
                    unsafe_allow_html=True,
                )
            st.caption(baseline_execution_plan["entry_guide"])
            st.caption(baseline_execution_plan["timing_note"])
    st.plotly_chart(plot_baseline_comparison(selected_result.baseline_comparison), use_container_width=True)
    st.dataframe(selected_result.baseline_comparison, use_container_width=True, hide_index=True)
    st.caption(
        "These are simpler reference systems on the same out-of-sample slices. `live_preferred` marks the asset-appropriate baseline universe used for live routing, while the extra rows stay visible for research context and stress-testing."
    )

with candidate_tab:
    if candidate_search_results.empty:
        st.info("Candidate search was not run for this session. Switch to `full research` and enable it in the sidebar when you want the slower leaderboard pass.")
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
        st.caption("This search uses `auto` historical sourcing: FMP stays primary, Coinbase is used only as deep-history backfill when FMP intraday crypto history is too short, and Yahoo remains a last-resort fallback. It includes promotion gates, outer holdout, cross-asset robustness, and staged multi-seed HMM checks on the top variants, but it is still prioritized and capped rather than globally exhaustive.")

with timeframe_tab:
    if timeframe_comparison.empty:
        st.info("Timeframe comparison was skipped for speed on this run. Switch to `full research` and enable it when you want the slower cross-interval check.")
    else:
        st.plotly_chart(plot_timeframe_comparison(timeframe_comparison), use_container_width=True)
        st.dataframe(timeframe_comparison, use_container_width=True)
        st.caption("The timeframe comparison uses the same strategy controls but interval-specific walk-forward window presets across `4hour`, `1day`, and `1hour`. Treat it as a relative sanity check, not a perfect apples-to-apples scorecard.")

with feature_tab:
    if feature_pack_comparison.empty:
        st.info("Feature-pack ablation was skipped for speed on this run. Switch to `full research` and enable it when you want the slower feature comparison.")
    else:
        st.plotly_chart(plot_feature_pack_comparison(feature_pack_comparison), use_container_width=True)
        st.dataframe(feature_pack_comparison, use_container_width=True)
        st.caption("Feature-pack ablation holds the timeframe and strategy controls fixed while changing what the HMM sees. This is the cleanest way to tell whether signal improvements are coming from better market representation or just tighter filters.")

with interpretation_tab:
    st.subheader("Current Run Readout")
    st.dataframe(metric_interpretation, use_container_width=True, hide_index=True)
    st.caption("These explanations interpret the latest bar and the current backtest in plain English. They are heuristics for readability, not hard statistical proof.")
    if engine_recommendation["engine"] != "hmm":
        st.subheader("Why HMM Lost")
        st.dataframe(hmm_loss_breakdown, use_container_width=True, hide_index=True)
        st.caption("When the live engine routes to a baseline or to cash, this table shows the main blockers: baseline competition, holdout weakness, bootstrap fragility, robustness, trade count, and failed promotion gates.")
    st.subheader("Current Control Meanings")
    st.dataframe(control_interpretation, use_container_width=True, hide_index=True)
    st.caption("These rows explain what the current settings are encouraging the strategy to do, so you can tell whether results are coming from signal quality or just stricter filters.")

with methodology_tab:
    st.subheader("Methodology Summary")
    methodology_rows = pd.DataFrame(
        [
            {
                "component": "Detected Asset Class",
                "value": asset_class_label(str(analysis.get("asset_class", "crypto"))),
                "interpretation": "The app now auto-detects crypto versus equities and adjusts annualization, walk-forward defaults, cost assumptions, and robustness suggestions accordingly.",
            },
            {
                "component": "Execution Profile",
                "value": _run_profile_label(getattr(analysis_plan, "profile", "core_signal")),
                "interpretation": getattr(analysis_plan, "note", "This run used the default execution profile."),
            },
            {
                "component": "Annualization Basis",
                "value": f"{int(float(selected_result.metrics.get('annualization_bars_per_year', 0.0)))} bars/year",
                "interpretation": "Annualized return and volatility use a crypto 24/7 calendar for crypto symbols and a market-hours approximation for stocks and ETFs.",
            },
            {
                "component": "Historical Provider",
                "value": _provider_label(str(analysis.get("data_provider", "n/a"))),
                "interpretation": (
                    str(analysis.get("data_provider_note"))
                    if analysis.get("data_provider_note")
                    else "Current run used the requested historical provider without auto-backfill."
                ),
            },
            {
                "component": "Robustness Basket",
                "value": ", ".join(analysis.get("robustness_symbols", ())),
                "interpretation": str(analysis.get("robustness_basket_note", "This run used the selected cross-asset basket for robustness checks.")),
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
                "component": "Model Convergence",
                "value": f"{selected_result.converged_ratio:.0%}",
                "interpretation": "This is the share of walk-forward folds whose HMM optimizer reported convergence. Lower values do not guarantee the signal is wrong, but they do make the fitted regimes less trustworthy.",
            },
            {
                "component": "Optimizer Warnings",
                "value": (
                    f"{optimizer_warning_folds} folds / {optimizer_warning_count} lines"
                    if optimizer_warning_count > 0
                    else "None captured"
                ),
                "interpretation": (
                    f"Example warning: {optimizer_warning_excerpt}"
                    if optimizer_warning_excerpt
                    else "Optimizer stderr is captured into fold diagnostics so convergence issues are inspectable without flooding the terminal."
                ),
            },
            {
                "component": "Current Engine Recommendation",
                "value": engine_recommendation["headline"],
                "interpretation": engine_recommendation["summary"],
            },
            {
                "component": "Live Baseline Universe",
                "value": "asset-aware preferred set",
                "interpretation": live_baseline_universe_note,
            },
            {
                "component": "Live Engine Mode",
                "value": _engine_mode_label(engine_mode),
                "interpretation": live_engine["summary"],
            },
            {
                "component": "Active Live Engine",
                "value": active_engine_name,
                "interpretation": active_execution_plan["summary"],
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
    st.subheader("Run Stages")
    if stage_timings.empty:
        st.info("No stage timing breakdown was recorded for this run.")
    else:
        st.dataframe(stage_timings, use_container_width=True, hide_index=True)
        st.caption("This timing table shows where the app spent time, so it is easier to tell the difference between a healthy heavy run and a genuine hang.")
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
    if not getattr(analysis_plan, "run_model_comparison", False):
        st.info("This run evaluated only the selected HMM state count for speed. Switch to `full research` and enable `Run 5-9 state comparison` to compare neighboring state counts.")
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
    if robustness.empty:
        st.info("Robustness basket was skipped for speed on this run. Switch to `full research` and enable `Run robustness basket` to compute the cross-asset check.")
    else:
        st.plotly_chart(plot_robustness_results(robustness), use_container_width=True)
        st.dataframe(robustness, use_container_width=True)
        st.caption(str(analysis.get("robustness_basket_note", "Robustness checks rerun the same methodology across the selected basket.")))

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
