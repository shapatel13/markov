from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.baselines import (
    build_baseline_execution_plan,
    build_daily_trend_filter_baseline,
    select_best_baseline_frame,
    summarize_baselines,
)
from markov_regime.bootstrap import block_bootstrap_confidence_intervals
from markov_regime.consensus import (
    ConsensusDiagnostics,
    apply_consensus_overlay,
    build_consensus_timeline,
    compare_consensus_gate_modes,
    summarize_consensus,
)
from markov_regime.confirmation import align_confirmation_predictions, apply_confirmation_overlay
from markov_regime.config import DataConfig, ModelConfig, StrategyConfig, SweepConfig, WalkForwardConfig, bars_per_year, default_walk_forward_config
from markov_regime.data import DataFetchResult, fetch_live_quote, fetch_price_data, normalize_symbol
from markov_regime.data import _redact_api_key, _resample_ohlcv
from markov_regime.features import FEATURE_COLUMNS, FORWARD_HORIZONS, build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.interpretation import (
    build_control_interpretation_rows,
    build_execution_plan,
    build_metric_interpretation_rows,
    build_promotion_gate_rows,
    build_trust_snapshot,
    recommend_strategy_engine,
    resolve_live_engine_mode,
    summarize_promotion_gates,
)
from markov_regime.research import (
    ResearchProgram,
    _candidate_grid,
    _fold_consistency_metrics,
    ensure_results_tsv,
    load_research_program,
    nested_holdout_evaluation,
    nested_holdout_summary_frame,
    run_candidate_search,
    run_feature_pack_comparison,
    summarize_candidate_search,
    write_research_program,
)
from markov_regime.readiness import (
    PrimetimeAuditResult,
    build_platform_gate_rows,
    summarize_platform_gates,
    summarize_primetime_report,
    write_primetime_audit_report,
)
from markov_regime.reporting import export_signal_report
from markov_regime.robustness import parse_symbol_list
from markov_regime.strategy import (
    apply_trading_rules,
    attach_state_action_columns,
    build_trade_table,
    compute_metrics,
    derive_state_actions,
    estimate_execution_cost_bps,
    parameter_sweep,
    replay_strategy,
    stress_test_transaction_costs,
    summarize_trade_table,
)
from markov_regime.walkforward import WalkForwardResult, generate_walk_forward_windows, run_walk_forward, suggest_walk_forward_config


def _signal_input_frame(max_posteriors: list[float]) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(max_posteriors), freq="h"),
            "open": [100.0] * len(max_posteriors),
            "high": [101.0] * len(max_posteriors),
            "low": [99.0] * len(max_posteriors),
            "close": [100.0 + index for index in range(len(max_posteriors))],
            "volume": [1_000_000] * len(max_posteriors),
            "bar_return": [0.001] * len(max_posteriors),
            "canonical_state": [0] * len(max_posteriors),
            "max_posterior": max_posteriors,
            "confidence_gap": [0.2] * len(max_posteriors),
        }
    )
    return base


class _FakeResponse:
    def __init__(self, url: str, payload: object, status_code: int = 200) -> None:
        self.url = url
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse]) -> None:
        self.responses = responses

    def get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        timeout: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        _ = timeout
        _ = headers
        if url in self.responses:
            return self.responses[url]
        rendered = f"{url}?{urlencode(params or {})}"
        return _FakeResponse(rendered, [], status_code=404)


def _fmp_hourly_payload(periods: int, start: str = "2025-01-01 00:00:00") -> list[dict[str, object]]:
    timestamps = pd.date_range(start, periods=periods, freq="h")
    payload: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        base = 100.0 + index
        payload.append(
            {
                "date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "volume": 1_000.0 + index,
            }
        )
    return payload


def _yahoo_chart_payload(periods: int, start: str = "2025-01-01 00:00:00", *, include_partial: bool = False) -> dict[str, object]:
    timestamps = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    if include_partial:
        timestamps = timestamps.append(pd.DatetimeIndex([timestamps[-1] + pd.Timedelta(minutes=16)]))

    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    for index, _timestamp in enumerate(timestamps):
        base = 200.0 + index
        opens.append(base)
        highs.append(base + 1.0)
        lows.append(base - 1.0)
        closes.append(base + 0.5)
        volumes.append(2_000.0 + index)

    return {
        "chart": {
            "result": [
                {
                    "timestamp": [int(item.timestamp()) for item in timestamps],
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": volumes,
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def _coinbase_payload(periods: int, start: str = "2025-01-01 00:00:00") -> list[list[float]]:
    timestamps = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    payload: list[list[float]] = []
    for index, timestamp in enumerate(timestamps):
        base = 300.0 + index
        payload.append(
            [
                float(int(timestamp.timestamp())),
                base - 1.0,
                base + 1.0,
                base,
                base + 0.5,
                3_000.0 + index,
            ]
        )
    return payload


def _consensus_comparison_fixture() -> tuple[WalkForwardResult, ConsensusDiagnostics]:
    frame = _signal_input_frame([0.9, 0.9, 0.9, 0.9])
    frame["fold_id"] = [0, 0, 0, 0]
    frame["signal_position"] = [0, 1, 1, 1]
    frame["candidate_action"] = [1, 1, 1, 1]
    frame["guardrail_reason"] = ["", "", "", ""]
    frame["turnover"] = [0.0, 1.0, 0.0, 0.0]
    frame["gross_strategy_return"] = [0.0, 0.001, 0.001, -0.001]
    frame["execution_cost_bps"] = [0.0, 0.0, 0.0, 0.0]
    frame["transaction_cost"] = [0.0, 0.0, 0.0, 0.0]
    frame["net_strategy_return"] = frame["gross_strategy_return"]
    frame["asset_wealth"] = (1.0 + frame["bar_return"]).cumprod()
    frame["strategy_wealth"] = (1.0 + frame["net_strategy_return"]).cumprod()

    config = StrategyConfig(require_consensus_confirmation=True, consensus_min_share=0.67, consensus_gate_mode="entry_only")
    trade_log = build_trade_table(frame)
    result = WalkForwardResult(
        n_states=1,
        predictions=frame,
        fold_diagnostics=pd.DataFrame(),
        state_stability=pd.DataFrame(),
        metrics=compute_metrics(frame, "4hour"),
        benchmark_metrics=compute_metrics(frame.assign(signal_position=1, candidate_action=1, turnover=0.0), "4hour"),
        cost_stress=stress_test_transaction_costs(frame, (0.0,), "4hour", config),
        bootstrap=block_bootstrap_confidence_intervals(frame["net_strategy_return"], interval="4hour", block_length=2, samples=20),
        forward_returns=pd.DataFrame(),
        guardrail_summary=pd.DataFrame([{"guardrail_reason": "accepted", "bars": len(frame), "share": 1.0}]),
        converged_ratio=1.0,
        trade_log=trade_log,
        trade_summary=summarize_trade_table(trade_log),
        baseline_comparison=summarize_baselines(frame, "4hour", config),
    )
    diagnostics = ConsensusDiagnostics(
        members=pd.DataFrame(),
        timeline=pd.DataFrame(
            {
                "timestamp": frame["timestamp"],
                "close": frame["close"],
                "position_consensus": [1, 1, 1, 1],
                "position_consensus_label": ["Long"] * 4,
                "position_consensus_share": [0.8, 0.8, 0.55, 0.55],
                "candidate_consensus": [1, 1, 1, 1],
                "candidate_consensus_label": ["Long"] * 4,
                "candidate_consensus_share": [0.8, 0.8, 0.55, 0.55],
                "long_votes": [2, 2, 2, 2],
                "flat_votes": [1, 1, 1, 1],
                "short_votes": [0, 0, 0, 0],
                "candidate_long_votes": [2, 2, 2, 2],
                "candidate_flat_votes": [1, 1, 1, 1],
                "candidate_short_votes": [0, 0, 0, 0],
                "active_member_share": [2 / 3] * 4,
                "active_candidate_share": [2 / 3] * 4,
                "member_count": [3] * 4,
            }
        ),
        summary=pd.DataFrame(),
    )
    return result, diagnostics


def test_feature_frame_contains_forward_horizons(synthetic_feature_frame: pd.DataFrame) -> None:
    for horizon in FORWARD_HORIZONS:
        assert f"forward_return_{horizon}" in synthetic_feature_frame.columns
    assert synthetic_feature_frame.loc[:, list(FEATURE_COLUMNS)].isna().sum().sum() == 0


def test_feature_packs_are_available_and_buildable(synthetic_prices: pd.DataFrame) -> None:
    assert {
        "baseline",
        "trend",
        "volatility",
        "regime_mix",
        "trend_strength",
        "mean_reversion",
        "vol_surface",
        "regime_mix_v2",
        "trend_context",
        "regime_context",
        "atr_causal",
    }.issubset(set(list_feature_packs()))

    trend_columns = get_feature_columns("trend")
    trend_frame = build_feature_frame(synthetic_prices, feature_columns=trend_columns)
    trend_strength_columns = get_feature_columns("trend_strength")
    trend_strength_frame = build_feature_frame(synthetic_prices, feature_columns=trend_strength_columns)
    mean_reversion_columns = get_feature_columns("mean_reversion")
    mean_reversion_frame = build_feature_frame(synthetic_prices, feature_columns=mean_reversion_columns)
    vol_surface_columns = get_feature_columns("vol_surface")
    vol_surface_frame = build_feature_frame(synthetic_prices, feature_columns=vol_surface_columns)
    trend_context_columns = get_feature_columns("trend_context")
    trend_context_frame = build_feature_frame(synthetic_prices, feature_columns=trend_context_columns)
    regime_context_columns = get_feature_columns("regime_context")
    regime_context_frame = build_feature_frame(synthetic_prices, feature_columns=regime_context_columns)
    atr_causal_columns = get_feature_columns("atr_causal")
    atr_causal_frame = build_feature_frame(synthetic_prices, feature_columns=atr_causal_columns)

    assert "ema_gap_24" in trend_columns
    assert "adx_14" in trend_strength_columns
    assert "rsi_14" in mean_reversion_columns
    assert "parkinson_vol_24" in vol_surface_columns
    assert "daily_trend_20" in trend_context_columns
    assert "daily_adx_14" in regime_context_columns
    assert "atr_momentum_24" in atr_causal_columns
    assert trend_frame.loc[:, list(trend_columns)].isna().sum().sum() == 0
    assert trend_strength_frame.loc[:, list(trend_strength_columns)].isna().sum().sum() == 0
    assert mean_reversion_frame.loc[:, list(mean_reversion_columns)].isna().sum().sum() == 0
    assert vol_surface_frame.loc[:, list(vol_surface_columns)].isna().sum().sum() == 0
    assert trend_context_frame.loc[:, list(trend_context_columns)].isna().sum().sum() == 0
    assert regime_context_frame.loc[:, list(regime_context_columns)].isna().sum().sum() == 0
    assert atr_causal_frame.loc[:, list(atr_causal_columns)].isna().sum().sum() == 0


def test_advanced_feature_columns_are_present_and_finite(synthetic_prices: pd.DataFrame) -> None:
    advanced_columns = (
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "rsi_14",
        "stoch_k_14",
        "bollinger_z_20",
        "bollinger_bandwidth_20",
        "donchian_position_20",
        "breakout_distance_20",
        "distance_to_vwap_24",
        "realized_skew_24",
        "realized_kurt_24",
        "parkinson_vol_24",
        "garman_klass_vol_24",
        "daily_trend_5",
        "daily_trend_20",
        "daily_ema_gap_20",
        "daily_rsi_14",
        "daily_bollinger_z_20",
        "daily_adx_14",
        "daily_di_spread_14",
        "daily_range_ratio_10",
        "atr_momentum_24",
        "causal_gap_24",
        "causal_slope_24",
    )
    frame = build_feature_frame(synthetic_prices, feature_columns=advanced_columns)

    assert set(advanced_columns).issubset(frame.columns)
    assert np.isfinite(frame.loc[:, list(advanced_columns)].to_numpy()).all()


def test_normalize_symbol_maps_common_crypto_aliases() -> None:
    assert normalize_symbol("BTC") == "BTCUSD"
    assert normalize_symbol("btc-usd") == "BTCUSD"
    assert normalize_symbol("SPY") == "SPY"


def test_redact_api_key_hides_secret_in_source_url() -> None:
    redacted = _redact_api_key("https://example.com/path?apikey=supersecret&symbol=BTCUSD")
    assert "supersecret" not in redacted
    assert "apikey=%2A%2A%2A" in redacted


def test_fetch_price_data_auto_falls_back_to_yahoo_for_thin_intraday_crypto() -> None:
    fmp_response = _FakeResponse(
        "https://financialmodelingprep.com/stable/historical-chart/1hour?symbol=BTCUSD&apikey=%2A%2A%2A",
        _fmp_hourly_payload(600),
    )
    yahoo_response = _FakeResponse(
        "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?interval=1h&range=730d",
        _yahoo_chart_payload(1200),
    )
    session = _FakeSession(
        {
            "https://financialmodelingprep.com/stable/historical-chart/1hour": fmp_response,
            "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD": yahoo_response,
        }
    )

    result = fetch_price_data(
        DataConfig(symbol="BTCUSD", interval="4hour", limit=200, provider="auto"),
        api_key="test-key",
        session=session,
    )

    assert result.provider == "yahoo"
    assert len(result.frame) == 200
    assert result.provider_note is not None
    assert "too thin" in result.provider_note
    assert "query1.finance.yahoo.com" in result.source_url


def test_fetch_price_data_auto_prefers_coinbase_backfill_before_yahoo() -> None:
    fmp_response = _FakeResponse(
        "https://financialmodelingprep.com/stable/historical-chart/1hour?symbol=BTCUSD&apikey=%2A%2A%2A",
        _fmp_hourly_payload(600),
    )
    coinbase_response = _FakeResponse(
        "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600&start=2025-01-01T00%3A00%3A00%2B00%3A00&end=2025-01-13T12%3A00%3A00%2B00%3A00",
        _coinbase_payload(1300),
    )
    yahoo_response = _FakeResponse(
        "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?interval=1h&range=730d",
        _yahoo_chart_payload(1500),
    )
    session = _FakeSession(
        {
            "https://financialmodelingprep.com/stable/historical-chart/1hour": fmp_response,
            "https://api.exchange.coinbase.com/products/BTC-USD/candles": coinbase_response,
            "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD": yahoo_response,
        }
    )

    result = fetch_price_data(
        DataConfig(symbol="BTCUSD", interval="1hour", limit=1000, provider="auto"),
        api_key="test-key",
        session=session,
    )

    assert result.provider == "coinbase"
    assert result.provider_note is not None
    assert "primary source" in result.provider_note


def test_fetch_price_data_respects_explicit_fmp_provider() -> None:
    fmp_response = _FakeResponse(
        "https://financialmodelingprep.com/stable/historical-chart/1hour?symbol=BTCUSD&apikey=%2A%2A%2A",
        _fmp_hourly_payload(600),
    )
    session = _FakeSession({"https://financialmodelingprep.com/stable/historical-chart/1hour": fmp_response})

    result = fetch_price_data(
        DataConfig(symbol="BTCUSD", interval="4hour", limit=200, provider="fmp"),
        api_key="test-key",
        session=session,
    )

    assert result.provider == "fmp"
    assert len(result.frame) == 149
    assert result.provider_note is None


def test_fetch_price_data_yahoo_drops_partial_intraday_bar() -> None:
    yahoo_response = _FakeResponse(
        "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?interval=1h&range=730d",
        _yahoo_chart_payload(12, include_partial=True),
    )
    session = _FakeSession({"https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD": yahoo_response})

    result = fetch_price_data(
        DataConfig(symbol="BTCUSD", interval="1hour", limit=20, provider="yahoo"),
        session=session,
    )

    assert result.provider == "yahoo"
    assert len(result.frame) == 12
    assert bool(result.frame["timestamp"].dt.minute.eq(0).all())


def test_fetch_price_data_coinbase_provider_parses_hourly_crypto_payload() -> None:
    coinbase_response = _FakeResponse(
        "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600",
        _coinbase_payload(10),
    )
    session = _FakeSession({"https://api.exchange.coinbase.com/products/BTC-USD/candles": coinbase_response})

    result = fetch_price_data(
        DataConfig(
            symbol="BTCUSD",
            interval="1hour",
            limit=10,
            provider="coinbase",
            start="2025-01-01 00:00:00",
            end="2025-01-01 09:00:00",
        ),
        session=session,
    )

    assert result.provider == "coinbase"
    assert len(result.frame) == 10
    assert float(result.frame.iloc[0]["open"]) == 300.0
    assert "api.exchange.coinbase.com" in result.source_url


def test_resample_ohlcv_builds_complete_4hour_bars_and_drops_partial() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01 01:00:00", periods=9, freq="h"),
            "open": [100.0 + index for index in range(9)],
            "high": [101.0 + index for index in range(9)],
            "low": [99.0 + index for index in range(9)],
            "close": [100.5 + index for index in range(9)],
            "volume": [10.0] * 9,
        }
    )

    resampled = _resample_ohlcv(frame, "4hour")

    assert len(resampled) == 2
    assert resampled["timestamp"].dt.hour.tolist() == [4, 8]
    assert float(resampled.iloc[0]["open"]) == 100.0
    assert float(resampled.iloc[0]["close"]) == 103.5
    assert float(resampled.iloc[0]["volume"]) == 40.0


def test_generate_walk_forward_windows_respects_roll_stride() -> None:
    windows = generate_walk_forward_windows(
        total_rows=600,
        config=WalkForwardConfig(train_bars=300, purge_bars=0, validate_bars=100, embargo_bars=0, test_bars=100, refit_stride_bars=50),
    )
    assert len(windows) == 3
    assert windows[0].train_end == 300
    assert windows[0].validate_start == 300
    assert windows[0].test_start == 400
    assert windows[1].train_start == 50
    assert windows[-1].test_end == 600


def test_generate_walk_forward_windows_respects_purge_and_embargo() -> None:
    windows = generate_walk_forward_windows(
        total_rows=700,
        config=WalkForwardConfig(train_bars=300, purge_bars=12, validate_bars=100, embargo_bars=24, test_bars=100, refit_stride_bars=100),
    )
    assert windows[0].validate_start == 312
    assert windows[0].test_start == 436
    assert windows[0].test_end == 536


def test_suggest_walk_forward_config_reduces_windows_to_fit_sample() -> None:
    adjusted, was_adjusted = suggest_walk_forward_config(
        total_rows=401,
        requested=WalkForwardConfig(train_bars=750, purge_bars=6, validate_bars=180, embargo_bars=6, test_bars=180, refit_stride_bars=180),
    )
    assert was_adjusted is True
    assert adjusted.train_bars + adjusted.purge_bars + adjusted.validate_bars + adjusted.embargo_bars + adjusted.test_bars <= 401
    assert adjusted.refit_stride_bars <= adjusted.test_bars


def test_guardrail_keeps_strategy_flat_when_posterior_is_low() -> None:
    frame = _signal_input_frame([0.51, 0.52, 0.54, 0.53])
    state_actions = pd.DataFrame(
        [{"canonical_state": 0, "action": 1, "label": "risk_on", "validation_edge": 0.02, "samples": 50, "avg_confidence": 0.8}]
    )
    config = StrategyConfig(posterior_threshold=0.65, required_confirmations=1)
    result = apply_trading_rules(frame, state_actions, config)

    assert result["signal_position"].eq(0).all()
    assert set(result["guardrail_reason"]) == {"posterior_below_threshold"}


def test_multi_horizon_state_scoring_requires_consistency() -> None:
    validate_frame = pd.DataFrame(
        {
            "canonical_state": [0] * 30 + [1] * 30,
            "max_posterior": [0.9] * 60,
            "forward_return_6": [0.02] * 30 + [0.02] * 30,
            "forward_return_12": [-0.02] * 30 + [0.015] * 30,
            "forward_return_24": [0.0] * 30 + [0.01] * 30,
        }
    )

    state_actions = derive_state_actions(
        validate_frame,
        n_states=2,
        config=StrategyConfig(
            min_validation_samples=10,
            scoring_horizons=(6, 12, 24),
            validation_shrinkage=0.0,
            min_consistent_horizons=2,
            required_confirmations=1,
        ),
    ).set_index("canonical_state")

    assert int(state_actions.loc[0, "action"]) == 0
    assert state_actions.loc[0, "label"] == "inconsistent_across_horizons"
    assert int(state_actions.loc[1, "action"]) == 1
    assert state_actions.loc[1, "label"] == "risk_on"
    assert int(state_actions.loc[1, "consistent_horizons"]) >= 2


def test_daily_confirmation_overlay_blocks_opposing_exposure() -> None:
    frame = _signal_input_frame([0.8, 0.8, 0.8])
    frame["signal_position"] = [0, 1, 1]
    frame["candidate_action"] = [1, 1, 1]
    frame["guardrail_reason"] = ["", "", ""]
    frame["turnover"] = [0.0, 1.0, 0.0]
    frame["gross_strategy_return"] = [0.0, 0.001, 0.001]
    frame["execution_cost_bps"] = [0.0, 0.0, 0.0]
    frame["transaction_cost"] = [0.0, 0.0, 0.0]
    frame["net_strategy_return"] = [0.0, 0.001, 0.001]
    frame["asset_wealth"] = [1.0, 1.001, 1.002001]
    frame["strategy_wealth"] = [1.0, 1.001, 1.002001]
    frame["confirmation_source_timestamp"] = pd.date_range("2024-12-31", periods=3, freq="D")
    frame["confirmation_effective_direction"] = [-1, -1, -1]
    frame["confirmation_signal_position"] = [-1, -1, -1]
    frame["confirmation_candidate_action"] = [-1, -1, -1]
    frame["confirmation_guardrail_reason"] = ["accepted", "accepted", "accepted"]

    filtered, summary = apply_confirmation_overlay(frame, StrategyConfig(require_daily_confirmation=True))

    assert filtered["signal_position"].eq(0).all()
    assert filtered["candidate_action"].eq(0).all()
    assert set(filtered["guardrail_reason"]) == {"daily_confirmation_opposes"}
    assert "blocked" in set(summary["confirmation_status"])


def test_daily_confirmation_overlay_allows_neutral_exposure() -> None:
    frame = _signal_input_frame([0.8, 0.8, 0.8])
    frame["signal_position"] = [0, 1, 1]
    frame["candidate_action"] = [1, 1, 1]
    frame["guardrail_reason"] = ["", "", ""]
    frame["turnover"] = [0.0, 1.0, 0.0]
    frame["gross_strategy_return"] = [0.0, 0.001, 0.001]
    frame["execution_cost_bps"] = [0.0, 0.0, 0.0]
    frame["transaction_cost"] = [0.0, 0.0, 0.0]
    frame["net_strategy_return"] = [0.0, 0.001, 0.001]
    frame["asset_wealth"] = [1.0, 1.001, 1.002001]
    frame["strategy_wealth"] = [1.0, 1.001, 1.002001]
    frame["confirmation_source_timestamp"] = pd.date_range("2024-12-31", periods=3, freq="D")
    frame["confirmation_effective_direction"] = [0, 0, 0]
    frame["confirmation_signal_position"] = [0, 0, 0]
    frame["confirmation_candidate_action"] = [0, 0, 0]
    frame["confirmation_guardrail_reason"] = ["accepted", "accepted", "accepted"]

    filtered, _ = apply_confirmation_overlay(frame, StrategyConfig(require_daily_confirmation=True))

    assert filtered["signal_position"].tolist() == [0, 1, 1]
    assert filtered["candidate_action"].tolist() == [1, 1, 1]


def test_align_confirmation_predictions_uses_latest_daily_bar() -> None:
    primary = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-02 04:00:00", "2025-01-02 08:00:00"]),
            "signal_position": [1, 1],
            "candidate_action": [1, 1],
            "guardrail_reason": ["", ""],
            "max_posterior": [0.8, 0.8],
            "confidence_gap": [0.2, 0.2],
        }
    )
    confirmation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-02 00:00:00"]),
            "signal_position": [0, 1],
            "candidate_action": [0, 1],
            "guardrail_reason": ["", ""],
            "max_posterior": [0.6, 0.9],
            "confidence_gap": [0.1, 0.2],
        }
    )

    aligned = align_confirmation_predictions(primary, confirmation, "1day")

    assert aligned["confirmation_interval"].eq("1day").all()
    assert aligned["confirmation_effective_direction"].tolist() == [1, 1]


def test_confirmations_and_min_hold_delay_entry_and_exit() -> None:
    frame = _signal_input_frame([0.8, 0.8, 0.8, 0.4, 0.4, 0.4])
    state_actions = pd.DataFrame(
        [{"canonical_state": 0, "action": 1, "label": "risk_on", "validation_edge": 0.02, "samples": 50, "avg_confidence": 0.8}]
    )
    config = StrategyConfig(
        posterior_threshold=0.65,
        required_confirmations=2,
        min_hold_bars=4,
        confidence_gap=0.05,
    )
    result = apply_trading_rules(frame, state_actions, config)

    assert result["signal_position"].tolist() == [0, 1, 1, 1, 1, 0]
    assert result.loc[3, "guardrail_reason"] == "posterior_below_threshold"


def test_trade_table_extracts_closed_and_open_trades() -> None:
    frame = _signal_input_frame([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    frame["signal_position"] = [0, 1, 1, 0, 1, 1]
    frame["candidate_action"] = [0, 1, 1, 0, 1, 1]
    frame["guardrail_reason"] = ["", "", "", "flat_exit", "", ""]
    frame["turnover"] = frame["signal_position"].diff().abs().fillna(0.0)
    frame["gross_strategy_return"] = [0.0, 0.0, 0.01, -0.02, 0.0, 0.03]
    frame["execution_cost_bps"] = [0.0] * len(frame)
    frame["transaction_cost"] = [0.0] * len(frame)
    frame["net_strategy_return"] = frame["gross_strategy_return"]
    frame["asset_wealth"] = (1.0 + frame["bar_return"]).cumprod()
    frame["strategy_wealth"] = (1.0 + frame["net_strategy_return"]).cumprod()

    trade_table = build_trade_table(frame)

    assert len(trade_table) == 2
    assert trade_table["status"].tolist() == ["closed", "open"]
    assert trade_table["bars_held"].tolist() == [2, 1]
    assert float(trade_table.iloc[0]["net_return"]) < 0.0
    assert float(trade_table.iloc[1]["net_return"]) > 0.0


def test_compute_metrics_includes_trade_level_fields() -> None:
    frame = _signal_input_frame([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    frame["signal_position"] = [0, 1, 1, 0, 1, 1]
    frame["candidate_action"] = [0, 1, 1, 0, 1, 1]
    frame["guardrail_reason"] = ["", "", "", "flat_exit", "", ""]
    frame["turnover"] = frame["signal_position"].diff().abs().fillna(0.0)
    frame["gross_strategy_return"] = [0.0, 0.0, 0.02, -0.005, 0.0, 0.01]
    frame["execution_cost_bps"] = [0.0] * len(frame)
    frame["transaction_cost"] = [0.0] * len(frame)
    frame["net_strategy_return"] = frame["gross_strategy_return"]
    frame["asset_wealth"] = (1.0 + frame["bar_return"]).cumprod()
    frame["strategy_wealth"] = (1.0 + frame["net_strategy_return"]).cumprod()

    metrics = compute_metrics(frame, "1hour")

    assert {"bar_win_rate", "trade_win_rate", "expectancy", "profit_factor", "avg_mfe", "avg_mae", "open_trade_count"}.issubset(metrics)
    assert metrics["win_rate"] == metrics["trade_win_rate"]
    assert metrics["trades"] == 2.0
    assert metrics["closed_trade_count"] == 1.0
    assert metrics["open_trade_count"] == 1.0


def test_parameter_sweep_produces_all_requested_combinations() -> None:
    frame = _signal_input_frame([0.8, 0.8, 0.7, 0.4, 0.8, 0.8, 0.7, 0.4])
    frame["fold_id"] = 0
    state_actions = pd.DataFrame(
        [{"canonical_state": 0, "action": 1, "label": "risk_on", "validation_edge": 0.02, "samples": 50, "avg_confidence": 0.8}]
    )
    base = attach_state_action_columns(apply_trading_rules(frame, state_actions, StrategyConfig(required_confirmations=1)), state_actions, 1)
    sweep = parameter_sweep(
        predictions=base,
        n_states=1,
        base_config=StrategyConfig(required_confirmations=1),
        sweep_config=SweepConfig(
            posterior_thresholds=(0.6, 0.7),
            min_hold_bars=(1, 3),
            cooldown_bars=(0,),
            required_confirmations=(1, 2),
        ),
        interval="1hour",
    )

    assert len(sweep) == 8
    assert {"posterior_threshold", "min_hold_bars", "cooldown_bars", "required_confirmations", "sharpe"}.issubset(sweep.columns)


def test_execution_cost_model_penalizes_wider_ranges_and_thinner_volume() -> None:
    frame = _signal_input_frame([0.8, 0.8])
    frame["range_ratio"] = [0.001, 0.02]
    frame["volume"] = [5_000_000, 50_000]
    costs = estimate_execution_cost_bps(frame, StrategyConfig())
    assert float(costs.iloc[1]) > float(costs.iloc[0])


def test_baseline_summary_includes_expected_references(synthetic_feature_frame: pd.DataFrame) -> None:
    baseline_comparison = summarize_baselines(synthetic_feature_frame, "1hour", StrategyConfig())

    assert {
        "buy_and_hold",
        "ema_trend",
        "vol_filtered_trend",
        "breakout",
        "atr_trend",
        "atr_breakout_stop",
        "daily_trend_filter",
        "atr_causal_trend",
        "daily_breakout_filter",
    }.issubset(set(baseline_comparison["baseline"]))
    assert {"sharpe", "annualized_return", "trades", "expectancy"}.issubset(baseline_comparison.columns)


def test_daily_trend_filter_baseline_uses_daily_context_when_available(synthetic_feature_frame: pd.DataFrame) -> None:
    frame = synthetic_feature_frame.copy()
    frame["daily_trend_20"] = 0.1
    frame["daily_ema_gap_20"] = 0.02
    frame["daily_adx_14"] = 0.2

    baseline = build_daily_trend_filter_baseline(frame, StrategyConfig())

    assert baseline["signal_position"].max() == 1
    assert baseline["guardrail_reason"].eq("daily_trend_filter").all()


def test_select_best_baseline_frame_uses_supplied_leaderboard(synthetic_feature_frame: pd.DataFrame) -> None:
    comparison = pd.DataFrame(
        [
            {"baseline": "ema_trend", "sharpe": 0.1},
            {"baseline": "daily_breakout_filter", "sharpe": 0.4},
        ]
    )

    baseline_name, best_row, baseline_frame = select_best_baseline_frame(
        synthetic_feature_frame,
        "4hour",
        StrategyConfig(),
        comparison,
    )

    assert baseline_name == "daily_breakout_filter"
    assert best_row["baseline"] == "daily_breakout_filter"
    assert not baseline_frame.empty
    assert baseline_frame["baseline_name"].eq("daily_breakout_filter").all()


def test_block_bootstrap_returns_major_metric_intervals() -> None:
    intervals = block_bootstrap_confidence_intervals(
        returns=[0.01, -0.005, 0.002, 0.003, -0.001] * 20,
        interval="1hour",
        block_length=5,
        samples=50,
    )
    assert set(intervals["metric"]) == {"total_return", "annualized_return", "annualized_volatility", "sharpe", "max_drawdown"}
    assert (intervals["upper"] >= intervals["lower"]).all()


def test_export_signal_report_writes_csv_and_json(tmp_path: Path) -> None:
    frame = _signal_input_frame([0.8, 0.8, 0.8])
    frame["signal_position"] = [0, 1, 1]
    frame["candidate_action"] = [1, 1, 1]
    frame["guardrail_reason"] = ["", "", ""]
    frame["gross_strategy_return"] = [0.0, 0.001, 0.001]
    frame["transaction_cost"] = [0.0, 0.0, 0.0]
    frame["net_strategy_return"] = [0.0, 0.001, 0.001]
    frame["strategy_wealth"] = [1.0, 1.001, 1.002001]
    frame["fold_id"] = 0

    exported = export_signal_report(frame, symbol="SPY", interval="1hour", export_dir=tmp_path)

    assert exported["csv"].exists()
    assert exported["json"].exists()
    assert exported["csv"].suffix == ".csv"
    assert exported["json"].suffix == ".json"


def test_bars_per_year_supports_4hour_crypto_calendar() -> None:
    assert bars_per_year("1hour") == 24 * 365
    assert bars_per_year("4hour") == 6 * 365
    assert bars_per_year("1day") == 365


def test_default_walk_forward_config_prefers_higher_timeframe_windows() -> None:
    four_hour = default_walk_forward_config("4hour")
    one_day = default_walk_forward_config("1day")
    one_hour = default_walk_forward_config("1hour")

    assert four_hour.train_bars == 365 * 6
    assert four_hour.validate_bars == 90 * 6
    assert four_hour.test_bars == 90 * 6
    assert one_day.train_bars == 365
    assert one_day.validate_bars == 90
    assert one_day.test_bars == 90
    assert one_hour.train_bars == 720


def test_research_program_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "research_program.md"
    write_research_program(
        path,
        ResearchProgram(
            symbol="ETHUSD",
            intervals=("4hour", "1day", "1hour"),
            feature_packs=("trend", "regime_mix"),
            max_candidates=8,
            artifact_top_k=2,
        ),
    )
    loaded = load_research_program(path)

    assert loaded.symbol == "ETHUSD"
    assert loaded.intervals == ("4hour", "1day", "1hour")
    assert loaded.feature_packs == ("trend", "regime_mix")
    assert loaded.max_candidates == 8
    assert loaded.artifact_top_k == 2


def test_candidate_grid_prioritizes_4hour_then_1day_then_1hour() -> None:
    grid = _candidate_grid(
        ResearchProgram(
            intervals=("1hour", "1day", "4hour"),
            feature_packs=("baseline",),
            state_counts=(6,),
            posterior_thresholds=(0.65,),
            min_hold_bars=(6,),
            cooldown_bars=(4,),
            required_confirmations=(2,),
            max_candidates=3,
        )
    )

    assert [candidate[0] for candidate in grid] == ["4hour", "1day", "1hour"]


def test_trust_snapshot_flags_thin_positive_run_as_fragile() -> None:
    snapshot = build_trust_snapshot(
        metrics={"sharpe": 2.5, "trades": 3.0},
        bootstrap=pd.DataFrame([{"metric": "sharpe", "lower": -1.2, "upper": 4.7}]),
        state_stability=pd.DataFrame([{"canonical_state": 0, "stability_score": 0.9}]),
        robustness=pd.DataFrame([{"symbol": "BTCUSD", "status": "ok", "sharpe": -0.4}]),
        interval="4hour",
        available_rows=513,
        walk_adjusted=True,
    )

    assert snapshot["verdict"] == "Promising but fragile"
    assert "bootstrap still crosses zero" in snapshot["summary"]


def test_metric_interpretation_explains_held_position_vs_candidate() -> None:
    rows = build_metric_interpretation_rows(
        latest_row={
            "signal_position": 1,
            "candidate_action": 0,
            "canonical_state": 3,
            "guardrail_reason": "no_directional_edge",
            "max_posterior": 0.99,
            "confidence_gap": 0.18,
        },
        metrics={"sharpe": 1.4, "annualized_return": 0.42, "confidence_coverage": 0.53, "trades": 3.0},
        bootstrap=pd.DataFrame([{"metric": "sharpe", "lower": -3.0, "upper": 4.0}]),
        state_stability=pd.DataFrame([{"canonical_state": 0, "stability_score": 1.0}]),
        robustness=pd.DataFrame([{"symbol": "BTCUSD", "status": "ok", "sharpe": -0.2}]),
        interval="4hour",
        available_rows=513,
        walk_adjusted=True,
    )
    lookup = rows.set_index("metric")

    assert lookup.loc["Held Position", "value"] == "Long"
    assert lookup.loc["Latest Candidate", "value"] == "Flat"
    assert "older position may still be held" in lookup.loc["Guardrail Status", "interpretation"]


def test_metric_interpretation_humanizes_daily_confirmation_status() -> None:
    rows = build_metric_interpretation_rows(
        latest_row={
            "signal_position": 0,
            "candidate_action": 0,
            "canonical_state": 3,
            "guardrail_reason": "no_directional_edge",
            "max_posterior": 0.78,
            "confidence_gap": 0.06,
            "confirmation_status": "no_primary_signal",
            "confirmation_effective_direction": 0,
            "confirmation_interval": "1day",
        },
        metrics={"sharpe": 0.14, "annualized_return": -0.002, "confidence_coverage": 0.1, "trades": 3.0},
        bootstrap=pd.DataFrame([{"metric": "sharpe", "lower": -1.0, "upper": 1.5}]),
        state_stability=pd.DataFrame([{"canonical_state": 0, "stability_score": 0.7}]),
        robustness=pd.DataFrame([{"symbol": "BTCUSD", "status": "ok", "sharpe": -0.2}]),
        interval="4hour",
        available_rows=512,
        walk_adjusted=True,
    )
    lookup = rows.set_index("metric")

    assert lookup.loc["Daily Confirmation", "value"] == "No Primary Signal (Flat)"


def test_build_execution_plan_flags_no_entry_when_guardrail_blocks() -> None:
    plan = build_execution_plan(
        latest_row={
            "timestamp": pd.Timestamp("2025-01-01 04:00:00"),
            "signal_position": 0,
            "candidate_action": 0,
            "guardrail_reason": "no_directional_edge",
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
        },
        interval="4hour",
        live_price=100.5,
    )

    assert plan["action"] == "No Entry"
    assert "no valid live entry level" in plan["entry_guide"].lower()


def test_build_execution_plan_surfaces_enter_long_levels() -> None:
    plan = build_execution_plan(
        latest_row={
            "timestamp": pd.Timestamp("2025-01-01 04:00:00"),
            "signal_position": 0,
            "candidate_action": 1,
            "guardrail_reason": "accepted",
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
        },
        interval="4hour",
        live_price=100.5,
    )

    assert plan["action"] == "Enter Long"
    assert "101.00" in plan["entry_guide"]
    assert "99.00" in plan["entry_guide"]


def test_build_execution_plan_surfaces_enter_short_levels() -> None:
    plan = build_execution_plan(
        latest_row={
            "timestamp": pd.Timestamp("2025-01-01 04:00:00"),
            "signal_position": 0,
            "candidate_action": -1,
            "guardrail_reason": "accepted",
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
        },
        interval="4hour",
        live_price=99.5,
    )

    assert plan["action"] == "Enter Short"
    assert "99.00" in plan["entry_guide"]
    assert "101.00" in plan["entry_guide"]


def test_build_execution_plan_supports_hold_short() -> None:
    plan = build_execution_plan(
        latest_row={
            "timestamp": pd.Timestamp("2025-01-01 04:00:00"),
            "signal_position": -1,
            "candidate_action": -1,
            "guardrail_reason": "accepted",
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
        },
        interval="4hour",
        live_price=99.5,
    )

    assert plan["action"] == "Hold Short"
    assert "already carrying short exposure" in plan["entry_guide"]


def test_build_baseline_execution_plan_surfaces_long_entry_levels() -> None:
    baseline_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="4h"),
            "close": [100.0, 102.0],
            "high": [101.0, 103.0],
            "low": [99.0, 100.5],
            "signal_position": [0, 1],
            "entry_trigger": [100.5, 101.5],
            "stop_level": [99.0, 98.5],
        }
    )

    plan = build_baseline_execution_plan(
        baseline_frame=baseline_frame,
        baseline_name="daily_breakout_filter",
        interval="4hour",
        live_price=102.5,
    )

    assert plan["action"] == "Enter Long"
    assert "101.50" in plan["entry_guide"]
    assert "98.50" in plan["entry_guide"]
    assert plan["engine_label"] == "Daily Breakout Filter"


def test_build_baseline_execution_plan_respects_flat_wait_state() -> None:
    baseline_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="4h"),
            "close": [100.0, 100.5],
            "high": [101.0, 101.0],
            "low": [99.0, 99.5],
            "signal_position": [0, 0],
            "entry_trigger": [101.0, 101.2],
            "stop_level": [99.0, 99.0],
        }
    )

    plan = build_baseline_execution_plan(
        baseline_frame=baseline_frame,
        baseline_name="daily_breakout_filter",
        interval="4hour",
        live_price=100.7,
    )

    assert plan["action"] == "No Entry"
    assert "101.20" in plan["entry_guide"]
    assert "prefer flat" in plan["timing_note"].lower()


def test_control_interpretation_describes_cautious_filters() -> None:
    rows = build_control_interpretation_rows(
        interval="4hour",
        feature_pack="regime_mix",
        walk_config=WalkForwardConfig(train_bars=420, purge_bars=2, validate_bars=84, embargo_bars=2, test_bars=84, refit_stride_bars=84),
        strategy_config=StrategyConfig(
            posterior_threshold=0.7,
            min_hold_bars=12,
            cooldown_bars=4,
            required_confirmations=3,
            confidence_gap=0.08,
            cost_bps=2.0,
            spread_bps=4.0,
            slippage_bps=3.0,
            impact_bps=2.0,
        ),
    )
    lookup = rows.set_index("control")

    assert "balanced" in lookup.loc["Posterior Threshold", "interpretation"].lower()
    assert "sticky" in lookup.loc["Min Hold", "interpretation"].lower()
    assert "anti-whipsaw" in lookup.loc["Cooldown", "interpretation"].lower()
    assert lookup.loc["Allow Shorts", "value"] == "off"


def test_control_interpretation_describes_short_enablement() -> None:
    rows = build_control_interpretation_rows(
        interval="4hour",
        feature_pack="trend",
        walk_config=WalkForwardConfig(train_bars=420, purge_bars=2, validate_bars=84, embargo_bars=2, test_bars=84, refit_stride_bars=84),
        strategy_config=StrategyConfig(allow_short=True),
    )
    lookup = rows.set_index("control")

    assert lookup.loc["Allow Shorts", "value"] == "on"
    assert "bearish regimes can become actual short trades" in lookup.loc["Allow Shorts", "interpretation"]


def test_promotion_gates_require_more_than_positive_sharpe() -> None:
    gates = build_promotion_gate_rows(
        metrics={"sharpe": 1.2, "trades": 3.0},
        bootstrap=pd.DataFrame([{"metric": "sharpe", "lower": -0.4, "upper": 2.5}]),
        state_stability=pd.DataFrame([{"canonical_state": 0, "stability_score": 0.42}]),
        robustness=pd.DataFrame([{"symbol": "BTCUSD", "status": "ok", "sharpe": -0.1}]),
        baseline_comparison=pd.DataFrame([{"baseline": "ema_trend", "sharpe": 0.4}]),
        interval="4hour",
        available_rows=900,
        walk_adjusted=True,
        fold_count=2,
        nested_holdout={"status": "insufficient_folds"},
    )
    snapshot = summarize_promotion_gates(gates)

    assert "pass" in set(gates["status"])
    assert "fail" in set(gates["status"])
    assert snapshot["verdict"] == "Not Ready"
    assert "Enough Walk-Forward Folds" in set(gates["gate"])
    assert "Nested Holdout Available" in set(gates["gate"])


def test_default_strategy_config_prefers_entry_only_consensus_mode() -> None:
    assert StrategyConfig().consensus_gate_mode == "entry_only"


def test_recommend_strategy_engine_prefers_positive_baseline_when_hmm_not_promoted() -> None:
    recommendation = recommend_strategy_engine(
        strategy_metrics={"sharpe": 0.42},
        baseline_comparison=pd.DataFrame([{"baseline": "atr_trend", "sharpe": 0.68}]),
        promotion_summary={"verdict": "Not Ready"},
    )

    assert recommendation["engine"] == "baseline"
    assert recommendation["headline"] == "Use baseline, not HMM"
    assert "atr_trend" in recommendation["summary"]


def test_resolve_live_engine_mode_routes_auto_to_baseline() -> None:
    resolved = resolve_live_engine_mode(
        requested_mode="auto",
        engine_recommendation={"engine": "baseline", "best_baseline": "daily_breakout_filter"},
        best_baseline="daily_breakout_filter",
    )

    assert resolved["engine"] == "baseline"
    assert "Auto Mode" in resolved["headline"]
    assert "daily_breakout_filter" in resolved["summary"]


def test_resolve_live_engine_mode_forces_hmm_research() -> None:
    resolved = resolve_live_engine_mode(
        requested_mode="hmm_research",
        engine_recommendation={"engine": "baseline", "best_baseline": "daily_breakout_filter"},
        best_baseline="daily_breakout_filter",
    )

    assert resolved["engine"] == "hmm"
    assert resolved["mode"] == "hmm_research"


def test_platform_gates_fail_on_stale_quote() -> None:
    gates = build_platform_gate_rows(
        tests_passed=True,
        compile_passed=True,
        historical_fetch_ok=True,
        live_quote_ok=True,
        live_quote_age_seconds=301.0,
        freshness_threshold_seconds=120.0,
        export_smoke_ok=True,
        artifact_smoke_ok=True,
        blind_oos_only=True,
    )
    lookup = gates.set_index("gate")

    assert lookup.loc["Live Quote Freshness", "status"] == "fail"


def test_primetime_summary_distinguishes_platform_and_strategy() -> None:
    summary = summarize_primetime_report(
        platform_summary={"verdict": "Operationally Ready", "severity": "success", "summary": "ok"},
        strategy_summary={"verdict": "Not Ready", "severity": "error", "summary": "not ok"},
        action_plan={"action": "No Entry"},
    )

    assert summary["verdict"] == "Platform Ready, Strategy Not Promoted"


def test_write_primetime_audit_report_writes_json_and_markdown(tmp_path: Path) -> None:
    audit = PrimetimeAuditResult(
        created_at_utc="2026-04-05T00:00:00+00:00",
        symbol="BTCUSD",
        resolved_symbol="BTCUSD",
        interval="4hour",
        feature_pack="trend",
        historical_provider="yahoo",
        historical_provider_note="Auto-selected Yahoo Finance long-history bars because the FMP intraday sample was too thin.",
        raw_rows=600,
        usable_rows=512,
        walk_adjusted=True,
        fold_count=2,
        live_quote_price=67000.0,
        live_quote_age_seconds=30.0,
        action_plan={"action": "No Entry", "summary": "flat", "entry_guide": "none", "timing_note": "wait"},
        platform_gates=pd.DataFrame([{"gate": "Pytest Suite", "status": "pass", "detail": "ok"}]),
        strategy_gates=pd.DataFrame([{"gate": "Positive OOS Sharpe", "status": "fail", "detail": "bad"}]),
        platform_summary={"verdict": "Operationally Ready", "severity": "success", "summary": "ok"},
        strategy_summary={"verdict": "Not Ready", "severity": "error", "summary": "not ok"},
        report_summary={"verdict": "Platform Ready, Strategy Not Promoted", "severity": "warning", "summary": "mixed"},
        nested_holdout={"status": "insufficient_folds"},
        pytest_output="49 passed",
        compile_output="compiled",
    )

    report_dir = write_primetime_audit_report(audit, output_dir=tmp_path)

    assert (report_dir / "primetime_audit.json").exists()
    assert (report_dir / "primetime_audit.md").exists()


def test_fetch_live_quote_parses_quote_payload() -> None:
    response = _FakeResponse(
        "https://financialmodelingprep.com/stable/quote?symbol=BTCUSD&apikey=%2A%2A%2A",
        [
            {
                "symbol": "BTCUSD",
                "price": 67166.33,
                "change": -946.02,
                "changePercentage": -1.38891,
                "volume": 460780205,
                "dayLow": 65696.96,
                "dayHigh": 68652.0,
                "marketCap": 1341485570894,
                "open": 68112.34,
                "previousClose": 68112.35,
                "exchange": "CRYPTO",
                "timestamp": 1775148953,
            }
        ],
    )
    session = _FakeSession({"https://financialmodelingprep.com/stable/quote": response})

    quote = fetch_live_quote("BTC", api_key="test-key", session=session)

    assert quote.symbol == "BTCUSD"
    assert quote.price == 67166.33
    assert quote.exchange == "CRYPTO"
    assert quote.timestamp is not None


def test_ensure_results_tsv_writes_header(tmp_path: Path) -> None:
    path = ensure_results_tsv(tmp_path / "results.tsv")
    header = path.read_text(encoding="utf-8").strip().split("\t")

    assert path.exists()
    assert header[0] == "created_at_utc"
    assert header[-1] == "notes"


def test_fold_consistency_metrics_split_early_and_late_folds() -> None:
    result = type(
        "DummyResult",
        (),
        {
            "fold_diagnostics": pd.DataFrame(
                [
                    {"fold_id": 0, "sharpe": 1.0},
                    {"fold_id": 1, "sharpe": 0.5},
                    {"fold_id": 2, "sharpe": -0.25},
                    {"fold_id": 3, "sharpe": -0.75},
                ]
            )
        },
    )()

    metrics = _fold_consistency_metrics(result)

    assert metrics["selection_sharpe"] == 0.75
    assert metrics["confirmation_sharpe"] == -0.5
    assert metrics["fold_consistency_gap"] == 1.25


def test_consensus_timeline_computes_majority_votes() -> None:
    member_predictions = {
        "a": pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="4h"),
                "close": [100.0, 101.0, 102.0],
                "signal_position": [1, 1, 0],
                "candidate_action": [1, 0, 0],
            }
        ),
        "b": pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="4h"),
                "close": [100.0, 101.0, 102.0],
                "signal_position": [1, 0, 0],
                "candidate_action": [1, -1, 0],
            }
        ),
        "c": pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="4h"),
                "close": [100.0, 101.0, 102.0],
                "signal_position": [0, 0, 0],
                "candidate_action": [0, 0, 0],
            }
        ),
    }

    timeline = build_consensus_timeline(member_predictions)
    summary = summarize_consensus(pd.DataFrame([{"sharpe": 1.0, "stability_score": 0.8}, {"sharpe": 0.5, "stability_score": 0.6}, {"sharpe": -0.2, "stability_score": 0.4}]), timeline)

    assert timeline["position_consensus"].tolist() == [1, 0, 0]
    assert timeline["candidate_consensus"].tolist() == [1, 0, 0]
    assert timeline["position_consensus_share"].tolist()[0] == 2 / 3
    assert not summary.empty
    assert "Latest Held Consensus" in set(summary["metric"])


def test_consensus_overlay_blocks_when_share_is_too_low() -> None:
    frame = _signal_input_frame([0.9, 0.9, 0.9])
    frame["signal_position"] = [0, 1, 1]
    frame["candidate_action"] = [1, 1, 1]
    frame["guardrail_reason"] = ["", "", ""]
    frame["turnover"] = [0.0, 1.0, 0.0]
    frame["gross_strategy_return"] = [0.0, 0.001, 0.001]
    frame["execution_cost_bps"] = [0.0, 0.0, 0.0]
    frame["transaction_cost"] = [0.0, 0.0, 0.0]
    frame["net_strategy_return"] = [0.0, 0.001, 0.001]
    frame["asset_wealth"] = [1.0, 1.001, 1.002001]
    frame["strategy_wealth"] = [1.0, 1.001, 1.002001]
    frame["consensus_timestamp"] = pd.date_range("2024-12-31", periods=3, freq="4h")
    frame["consensus_position"] = [1, 1, 1]
    frame["consensus_position_share"] = [0.55, 0.55, 0.55]
    frame["consensus_candidate"] = [1, 1, 1]
    frame["consensus_candidate_share"] = [0.55, 0.55, 0.55]

    filtered, summary = apply_consensus_overlay(
        frame,
        StrategyConfig(require_consensus_confirmation=True, consensus_min_share=0.67, consensus_gate_mode="hard"),
    )

    assert filtered["signal_position"].eq(0).all()
    assert set(filtered["guardrail_reason"]) == {"consensus_weak_share"}
    assert "weak_share" in set(summary["consensus_status"])


def test_consensus_entry_only_mode_preserves_existing_hold() -> None:
    frame = _signal_input_frame([0.9, 0.9, 0.9])
    frame["signal_position"] = [0, 1, 1]
    frame["candidate_action"] = [1, 1, 1]
    frame["guardrail_reason"] = ["", "", ""]
    frame["turnover"] = [0.0, 1.0, 0.0]
    frame["gross_strategy_return"] = [0.0, 0.001, 0.001]
    frame["execution_cost_bps"] = [0.0, 0.0, 0.0]
    frame["transaction_cost"] = [0.0, 0.0, 0.0]
    frame["net_strategy_return"] = [0.0, 0.001, 0.001]
    frame["asset_wealth"] = [1.0, 1.001, 1.002001]
    frame["strategy_wealth"] = [1.0, 1.001, 1.002001]
    frame["consensus_timestamp"] = pd.date_range("2024-12-31", periods=3, freq="4h")
    frame["consensus_position"] = [1, 1, 1]
    frame["consensus_position_share"] = [0.8, 0.8, 0.55]
    frame["consensus_candidate"] = [1, 1, 1]
    frame["consensus_candidate_share"] = [0.8, 0.8, 0.55]

    filtered, summary = apply_consensus_overlay(
        frame,
        StrategyConfig(require_consensus_confirmation=True, consensus_min_share=0.67, consensus_gate_mode="entry_only"),
    )

    assert filtered["signal_position"].tolist() == [0, 1, 1]
    assert filtered.loc[2, "guardrail_reason"] == "consensus_hold_weak_share"
    assert "weak_share" in set(summary["consensus_status"])


def test_compare_consensus_gate_modes_surfaces_off_hard_and_entry_only() -> None:
    result, diagnostics = _consensus_comparison_fixture()

    comparison = compare_consensus_gate_modes(
        result,
        diagnostics,
        interval="4hour",
        strategy_config=StrategyConfig(require_consensus_confirmation=True, consensus_min_share=0.67, consensus_gate_mode="entry_only"),
    )

    assert comparison["mode"].tolist() == ["off", "hard", "entry_only"]
    assert comparison.loc[comparison["mode"] == "entry_only", "selected"].item() is True
    assert comparison.loc[comparison["mode"] == "off", "selected"].item() is False
    assert comparison.loc[comparison["mode"] == "hard", "latest_guardrail"].item() == "consensus_weak_share"
    assert comparison.loc[comparison["mode"] == "entry_only", "latest_guardrail"].item() == "consensus_hold_weak_share"
    assert comparison.loc[comparison["mode"] == "hard", "blocked_requested_share"].item() > 0.0
    assert comparison.loc[comparison["mode"] == "off", "blocked_requested_share"].item() == 0.0


def test_nested_holdout_evaluation_returns_outer_metrics() -> None:
    frame = _signal_input_frame([0.9] * 15)
    frame["fold_id"] = [0] * 5 + [1] * 5 + [2] * 5
    state_actions = pd.DataFrame(
        [
            {
                "canonical_state": 0,
                "action": 1,
                "label": "risk_on",
                "validation_edge": 0.02,
                "score_lower": 0.015,
                "score_upper": 0.03,
                "consistent_horizons": 3,
                "samples": 50,
                "avg_confidence": 0.8,
            }
        ]
    )
    predictions = attach_state_action_columns(
        apply_trading_rules(frame, state_actions, StrategyConfig(required_confirmations=1)),
        state_actions,
        1,
    )

    nested = nested_holdout_evaluation(
        predictions=predictions,
        n_states=1,
        base_config=StrategyConfig(required_confirmations=1),
        interval="1hour",
        outer_holdout_folds=1,
    )

    assert nested["status"] == "ok"
    assert nested["outer_holdout_folds"] == 1.0
    assert {"outer_holdout_sharpe", "selected_inner_posterior_threshold", "selected_inner_required_confirmations"}.issubset(nested)


def test_nested_holdout_summary_frame_formats_status_and_selection() -> None:
    rows = nested_holdout_summary_frame(
        {
            "status": "ok",
            "outer_holdout_folds": 1.0,
            "outer_holdout_sharpe": 0.84,
            "outer_holdout_annualized_return": 0.12,
            "outer_holdout_trades": 4.0,
            "selected_inner_posterior_threshold": 0.7,
            "selected_inner_min_hold_bars": 6.0,
            "selected_inner_cooldown_bars": 4.0,
            "selected_inner_required_confirmations": 2.0,
            "selection_score": 1.25,
        }
    )
    lookup = rows.set_index("component")

    assert lookup.loc["Nested Holdout Status", "value"] == "Ok"
    assert lookup.loc["Outer Holdout Sharpe", "value"] == "0.84"
    assert lookup.loc["Selected Confirmations", "value"] == 2
    assert "trustworthy than the best row" in lookup.loc["Outer Holdout Sharpe", "interpretation"]


def test_replay_strategy_applies_consensus_overlay_when_requested() -> None:
    frame = _signal_input_frame([0.9, 0.9, 0.9, 0.9])
    frame["fold_id"] = [0, 0, 0, 0]
    frame["state_action_0"] = [1, 1, 1, 1]
    frame["validation_edge_0"] = [0.02, 0.02, 0.02, 0.02]
    frame["score_lower_0"] = [0.01, 0.01, 0.01, 0.01]
    frame["score_upper_0"] = [0.03, 0.03, 0.03, 0.03]
    frame["consistent_horizons_0"] = [3, 3, 3, 3]
    frame["consensus_timestamp"] = frame["timestamp"]
    frame["consensus_position"] = [1, 1, 1, 1]
    frame["consensus_candidate"] = [1, 1, 1, 1]
    frame["consensus_position_share"] = [0.8, 0.8, 0.55, 0.55]
    frame["consensus_candidate_share"] = [0.8, 0.8, 0.55, 0.55]

    replayed, _ = replay_strategy(
        frame,
        n_states=1,
        config=StrategyConfig(
            required_confirmations=1,
            require_consensus_confirmation=True,
            consensus_min_share=0.67,
            consensus_gate_mode="hard",
        ),
        interval="4hour",
    )

    assert replayed["signal_position"].tolist() == [1, 1, 0, 0]
    assert replayed["guardrail_reason"].tolist()[-1] == "consensus_weak_share"


def test_feature_pack_comparison_runs_on_synthetic_prices(synthetic_prices: pd.DataFrame) -> None:
    comparison = run_feature_pack_comparison(
        price_frame=synthetic_prices,
        interval="1hour",
        model_config=ModelConfig(n_states=5, random_state=11, n_iter=150),
        strategy_config=StrategyConfig(required_confirmations=1, min_validation_samples=10),
        feature_packs=("baseline", "regime_mix"),
        auto_adjust_windows=True,
    )

    assert set(comparison["feature_pack"]) == {"baseline", "regime_mix"}
    assert (comparison["status"] == "ok").all()
    assert {"selection_sharpe", "confirmation_sharpe", "fold_consistency_gap"}.issubset(comparison.columns)


def test_run_candidate_search_returns_ranked_leaderboard(monkeypatch, synthetic_prices: pd.DataFrame) -> None:
    def _fake_fetch(data_config: DataConfig, *args, **kwargs) -> DataFetchResult:
        _ = args
        _ = kwargs
        return DataFetchResult(
            frame=synthetic_prices.copy(),
            source_url="https://example.com/history",
            requested_symbol=data_config.symbol,
            resolved_symbol=data_config.symbol,
            provider="coinbase",
            provider_note=None,
        )

    monkeypatch.setattr("markov_regime.research.fetch_price_data", _fake_fetch)
    monkeypatch.setattr(
        "markov_regime.research.run_multi_asset_robustness",
        lambda **kwargs: pd.DataFrame(
            [
                {"symbol": "BTCUSD", "status": "ok", "sharpe": 0.35},
                {"symbol": "ETHUSD", "status": "ok", "sharpe": 0.22},
                {"symbol": "SOLUSD", "status": "ok", "sharpe": 0.11},
            ]
        ),
    )

    leaderboard = run_candidate_search(
        symbol="BTCUSD",
        interval="1hour",
        limit=1200,
        history_provider="auto",
        base_model_config=ModelConfig(n_states=5, random_state=11, n_iter=150),
        base_strategy_config=StrategyConfig(required_confirmations=1, min_validation_samples=10),
        feature_packs=("baseline", "trend"),
        state_counts=(5,),
        short_modes=(False,),
        confirmation_modes=("off",),
        robustness_symbols=("BTCUSD", "ETHUSD", "SOLUSD"),
        auto_adjust_windows=True,
        max_candidates=4,
    )

    assert not leaderboard.empty
    assert leaderboard.iloc[0]["rank"] == 1
    assert {"engine_recommendation", "promotion_verdict", "candidate_score", "recommendation_detail"}.issubset(leaderboard.columns)

    summary = summarize_candidate_search(leaderboard)
    assert summary["status"] in {"keep", "candidate", "discard"}
    assert "Top ranked variant" in summary["summary"]


def test_walk_forward_pipeline_runs_end_to_end(synthetic_feature_frame: pd.DataFrame) -> None:
    result = run_walk_forward(
        feature_frame=synthetic_feature_frame,
        feature_columns=FEATURE_COLUMNS,
        interval="1hour",
        model_config=ModelConfig(n_states=5, random_state=11, n_iter=150),
        walk_config=WalkForwardConfig(train_bars=300, validate_bars=100, test_bars=100, refit_stride_bars=100),
        strategy_config=StrategyConfig(required_confirmations=1, min_validation_samples=10),
    )

    assert not result.predictions.empty
    assert not result.fold_diagnostics.empty
    assert not result.state_stability.empty
    assert not result.forward_returns.empty
    assert not result.trade_summary.empty
    assert {"metric", "value"}.issubset(result.trade_summary.columns)
    assert "sharpe" in result.metrics
    assert "trade_win_rate" in result.metrics
    assert "sharpe" in result.benchmark_metrics
    assert result.predictions["is_blind_oos"].all()
    assert set(result.predictions["oos_segment"]) == {"blind_test"}


def test_parse_symbol_list_handles_commas_and_spacing() -> None:
    assert parse_symbol_list(" BTCUSD, ethusd ,SOLUSD ") == ["BTCUSD", "ETHUSD", "SOLUSD"]


def test_artifact_bundle_writes_manifest_and_reports(tmp_path: Path, synthetic_feature_frame: pd.DataFrame) -> None:
    result = run_walk_forward(
        feature_frame=synthetic_feature_frame,
        feature_columns=FEATURE_COLUMNS,
        interval="1hour",
        model_config=ModelConfig(n_states=5, random_state=11, n_iter=150),
        walk_config=WalkForwardConfig(train_bars=300, purge_bars=6, validate_bars=100, embargo_bars=6, test_bars=100, refit_stride_bars=100),
        strategy_config=StrategyConfig(required_confirmations=1, min_validation_samples=10),
    )
    bundle = write_run_artifact_bundle(
        symbol="BTC",
        resolved_symbol="BTCUSD",
        interval="1hour",
        data_url="https://example.com?apikey=%2A%2A%2A&symbol=BTCUSD",
        raw_frame=synthetic_feature_frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy(),
        feature_frame=synthetic_feature_frame,
        data_config=DataConfig(symbol="BTCUSD", interval="1hour", limit=500),
        model_config=ModelConfig(n_states=5),
        walk_config=WalkForwardConfig(train_bars=300, purge_bars=6, validate_bars=100, embargo_bars=6, test_bars=100, refit_stride_bars=100),
        strategy_config=StrategyConfig(),
        selected_result=result,
        comparison=pd.DataFrame([{"n_states": 5, "sharpe": result.metrics["sharpe"]}]),
        sweep_results=pd.DataFrame([{"posterior_threshold": 0.7, "sharpe": result.metrics["sharpe"]}]),
        notes=["artifact test"],
        robustness=pd.DataFrame([{"symbol": "BTCUSD", "status": "ok", "sharpe": result.metrics["sharpe"]}]),
        feature_columns=FEATURE_COLUMNS,
        metadata={"feature_pack": "baseline"},
        timeframe_comparison=pd.DataFrame([{"interval": "4hour", "status": "ok", "sharpe": result.metrics["sharpe"]}]),
        consensus_mode_comparison=pd.DataFrame([{"mode": "off", "label": "No Consensus", "selected": True, "sharpe": result.metrics["sharpe"]}]),
        candidate_search_results=pd.DataFrame(
            [
                {
                    "rank": 1,
                    "feature_pack": "baseline",
                    "n_states": 5,
                    "engine_recommendation": "Use baseline, not HMM",
                }
            ]
        ),
        nested_holdout_summary=pd.DataFrame(
            [
                {
                    "status": "ok",
                    "outer_holdout_folds": 1.0,
                    "outer_holdout_sharpe": result.metrics["sharpe"],
                    "outer_holdout_annualized_return": result.metrics["annualized_return"],
                    "outer_holdout_trades": result.metrics["trades"],
                }
            ]
        ),
        export_dir=tmp_path,
    )
    assert bundle.manifest_path.exists()
    assert bundle.files["signal_report_csv"].exists()
    assert bundle.files["trade_summary_csv"].exists()
    assert bundle.files["timeframe_comparison_csv"].exists()
    assert bundle.files["consensus_mode_comparison_csv"].exists()
    assert bundle.files["candidate_search_results_csv"].exists()
    assert bundle.files["nested_holdout_summary_csv"].exists()
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS)
    assert manifest["metadata"]["feature_pack"] == "baseline"
    assert manifest["schema_version"] == 6
    assert manifest["methodology"]["performance_stitching"] == "blind_test_windows_only"
    assert manifest["methodology"]["nested_holdout"]["status"] == "ok"
