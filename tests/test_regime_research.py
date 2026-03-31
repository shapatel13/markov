from __future__ import annotations

from pathlib import Path

import pandas as pd

from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.bootstrap import block_bootstrap_confidence_intervals
from markov_regime.config import DataConfig, ModelConfig, StrategyConfig, SweepConfig, WalkForwardConfig
from markov_regime.data import normalize_symbol
from markov_regime.data import _redact_api_key
from markov_regime.features import FEATURE_COLUMNS, FORWARD_HORIZONS
from markov_regime.reporting import export_signal_report
from markov_regime.robustness import parse_symbol_list
from markov_regime.strategy import (
    apply_trading_rules,
    attach_state_action_columns,
    derive_state_actions,
    estimate_execution_cost_bps,
    parameter_sweep,
)
from markov_regime.walkforward import generate_walk_forward_windows, run_walk_forward, suggest_walk_forward_config


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


def test_feature_frame_contains_forward_horizons(synthetic_feature_frame: pd.DataFrame) -> None:
    for horizon in FORWARD_HORIZONS:
        assert f"forward_return_{horizon}" in synthetic_feature_frame.columns
    assert synthetic_feature_frame.loc[:, list(FEATURE_COLUMNS)].isna().sum().sum() == 0


def test_normalize_symbol_maps_common_crypto_aliases() -> None:
    assert normalize_symbol("BTC") == "BTCUSD"
    assert normalize_symbol("btc-usd") == "BTCUSD"
    assert normalize_symbol("SPY") == "SPY"


def test_redact_api_key_hides_secret_in_source_url() -> None:
    redacted = _redact_api_key("https://example.com/path?apikey=supersecret&symbol=BTCUSD")
    assert "supersecret" not in redacted
    assert "apikey=%2A%2A%2A" in redacted


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
    assert "sharpe" in result.metrics
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
        export_dir=tmp_path,
    )
    assert bundle.manifest_path.exists()
    assert bundle.files["signal_report_csv"].exists()
