from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.baselines import summarize_baselines
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
from markov_regime.data import normalize_symbol
from markov_regime.data import _redact_api_key, _resample_ohlcv
from markov_regime.features import FEATURE_COLUMNS, FORWARD_HORIZONS, build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.interpretation import (
    build_control_interpretation_rows,
    build_metric_interpretation_rows,
    build_trust_snapshot,
)
from markov_regime.research import (
    ResearchProgram,
    _candidate_grid,
    _fold_consistency_metrics,
    ensure_results_tsv,
    load_research_program,
    nested_holdout_evaluation,
    run_feature_pack_comparison,
    write_research_program,
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

    assert "ema_gap_24" in trend_columns
    assert "adx_14" in trend_strength_columns
    assert "rsi_14" in mean_reversion_columns
    assert "parkinson_vol_24" in vol_surface_columns
    assert "daily_trend_20" in trend_context_columns
    assert "daily_adx_14" in regime_context_columns
    assert trend_frame.loc[:, list(trend_columns)].isna().sum().sum() == 0
    assert trend_strength_frame.loc[:, list(trend_strength_columns)].isna().sum().sum() == 0
    assert mean_reversion_frame.loc[:, list(mean_reversion_columns)].isna().sum().sum() == 0
    assert vol_surface_frame.loc[:, list(vol_surface_columns)].isna().sum().sum() == 0
    assert trend_context_frame.loc[:, list(trend_context_columns)].isna().sum().sum() == 0
    assert regime_context_frame.loc[:, list(regime_context_columns)].isna().sum().sum() == 0


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

    assert {"buy_and_hold", "ema_trend", "vol_filtered_trend", "breakout"}.issubset(set(baseline_comparison["baseline"]))
    assert {"sharpe", "annualized_return", "trades", "expectancy"}.issubset(baseline_comparison.columns)


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
    one_hour = default_walk_forward_config("1hour")

    assert four_hour.train_bars < one_hour.train_bars
    assert four_hour.validate_bars < one_hour.validate_bars
    assert four_hour.test_bars < one_hour.test_bars


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

    filtered, summary = apply_consensus_overlay(frame, StrategyConfig(require_consensus_confirmation=True, consensus_min_share=0.67))

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
        export_dir=tmp_path,
    )
    assert bundle.manifest_path.exists()
    assert bundle.files["signal_report_csv"].exists()
    assert bundle.files["trade_summary_csv"].exists()
    assert bundle.files["timeframe_comparison_csv"].exists()
    assert bundle.files["consensus_mode_comparison_csv"].exists()
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS)
    assert manifest["metadata"]["feature_pack"] == "baseline"
    assert manifest["schema_version"] == 4
