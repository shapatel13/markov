from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.config import (
    DataConfig,
    HistoricalProvider,
    Interval,
    ModelConfig,
    StrategyConfig,
    SweepConfig,
    WalkForwardConfig,
    default_robustness_basket,
    default_walk_forward_config,
    infer_asset_class,
)
from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.consensus import apply_consensus_confirmation, compare_consensus_gate_modes, run_consensus_diagnostics
from markov_regime.data import LiveQuote, fetch_live_quote, fetch_price_data
from markov_regime.features import build_feature_frame, get_feature_columns
from markov_regime.interpretation import build_execution_plan, build_promotion_gate_rows, summarize_promotion_gates
from markov_regime.reporting import export_signal_report
from markov_regime.research import nested_holdout_evaluation
from markov_regime.research_notes import build_research_notes
from markov_regime.robustness import parse_symbol_list, run_multi_asset_robustness
from markov_regime.strategy import parameter_sweep
from markov_regime.walkforward import compare_state_counts, summarize_state_count_results, suggest_walk_forward_config


DEFAULT_AUDIT_STRATEGY = StrategyConfig(
    posterior_threshold=0.7,
    min_hold_bars=6,
    cooldown_bars=4,
    required_confirmations=2,
    confidence_gap=0.06,
    require_daily_confirmation=False,
    require_consensus_confirmation=False,
    consensus_min_share=0.67,
    consensus_gate_mode="entry_only",
    allow_short=False,
    cost_bps=10.0,
    spread_bps=4.0,
    slippage_bps=3.0,
    impact_bps=2.0,
)


@dataclass(frozen=True)
class PrimetimeAuditResult:
    created_at_utc: str
    symbol: str
    resolved_symbol: str
    interval: Interval
    feature_pack: str
    historical_provider: HistoricalProvider
    historical_provider_note: str | None
    raw_rows: int
    usable_rows: int
    walk_adjusted: bool
    fold_count: int
    live_quote_price: float | None
    live_quote_age_seconds: float | None
    action_plan: dict[str, str]
    platform_gates: pd.DataFrame
    strategy_gates: pd.DataFrame
    platform_summary: dict[str, str]
    strategy_summary: dict[str, str]
    report_summary: dict[str, str]
    nested_holdout: dict[str, object]
    pytest_output: str
    compile_output: str


def _run_command(command: list[str], cwd: str | Path) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
    return result.returncode == 0, output


def build_platform_gate_rows(
    *,
    tests_passed: bool,
    compile_passed: bool,
    historical_fetch_ok: bool,
    live_quote_ok: bool,
    live_quote_age_seconds: float | None,
    freshness_threshold_seconds: float,
    export_smoke_ok: bool,
    artifact_smoke_ok: bool,
    blind_oos_only: bool,
) -> pd.DataFrame:
    rows = [
        {
            "gate": "Pytest Suite",
            "status": "pass" if tests_passed else "fail",
            "detail": "The full automated test suite completed successfully." if tests_passed else "The automated test suite reported at least one failure.",
        },
        {
            "gate": "Compile / Import Check",
            "status": "pass" if compile_passed else "fail",
            "detail": "Core application modules compiled successfully." if compile_passed else "At least one core module failed to compile.",
        },
        {
            "gate": "Historical Data Fetch",
            "status": "pass" if historical_fetch_ok else "fail",
            "detail": "Historical bars were fetched successfully from the configured provider." if historical_fetch_ok else "Historical data fetch failed.",
        },
        {
            "gate": "Live Quote Fetch",
            "status": "pass" if live_quote_ok else "fail",
            "detail": "A live quote was fetched successfully from the configured provider." if live_quote_ok else "Live quote fetch failed.",
        },
        {
            "gate": "Live Quote Freshness",
            "status": "pass" if live_quote_age_seconds is not None and live_quote_age_seconds <= freshness_threshold_seconds else "fail",
            "detail": (
                f"Live quote age is {live_quote_age_seconds:.1f} seconds."
                if live_quote_age_seconds is not None
                else "Live quote age is unavailable because the quote fetch failed."
            ),
        },
        {
            "gate": "Signal Export Smoke",
            "status": "pass" if export_smoke_ok else "fail",
            "detail": "CSV and JSON signal exports completed successfully." if export_smoke_ok else "Signal export smoke test failed.",
        },
        {
            "gate": "Artifact Bundle Smoke",
            "status": "pass" if artifact_smoke_ok else "fail",
            "detail": "Artifact bundle creation completed successfully." if artifact_smoke_ok else "Artifact bundle creation failed.",
        },
        {
            "gate": "Blind OOS Integrity",
            "status": "pass" if blind_oos_only else "fail",
            "detail": "Predictions are stitched only from blind out-of-sample test windows." if blind_oos_only else "The predictions include rows that are not flagged as blind OOS.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_platform_gates(gates: pd.DataFrame) -> dict[str, str]:
    if gates.empty:
        return {"verdict": "Unavailable", "severity": "warning", "summary": "Platform readiness gates are unavailable."}
    passed = int((gates["status"] == "pass").sum())
    total = int(len(gates))
    if passed == total:
        return {
            "verdict": "Operationally Ready",
            "severity": "success",
            "summary": f"The platform-level checks all passed. Passed {passed} of {total} operational gates.",
        }
    return {
        "verdict": "Operational Risk",
        "severity": "error",
        "summary": f"At least one platform-level check failed. Passed {passed} of {total} operational gates.",
    }


def summarize_primetime_report(
    *,
    platform_summary: Mapping[str, str],
    strategy_summary: Mapping[str, str],
    action_plan: Mapping[str, str],
) -> dict[str, str]:
    if platform_summary.get("severity") == "error":
        return {
            "verdict": "Not Primetime Ready",
            "severity": "error",
            "summary": "The platform itself still has operational failures. Fix those before considering any live usage.",
        }
    if strategy_summary.get("severity") == "success":
        return {
            "verdict": "Primetime Candidate",
            "severity": "success",
            "summary": f"The platform checks passed and the current strategy run cleared the promotion gates. Current action: {action_plan.get('action', 'n/a')}.",
        }
    return {
        "verdict": "Platform Ready, Strategy Not Promoted",
        "severity": "warning",
        "summary": (
            "The software platform looks operationally healthy, but the current strategy instance does not clear the promotion gates. "
            f"Current action: {action_plan.get('action', 'n/a')}."
        ),
    }


def _app_default_strategy_config() -> StrategyConfig:
    return DEFAULT_AUDIT_STRATEGY


def run_primetime_audit(
    *,
    repo_root: str | Path,
    symbol: str = "BTCUSD",
    interval: Interval = "4hour",
    feature_pack: str = "trend",
    states: int = 6,
    limit: int = 5000,
    history_provider: HistoricalProvider = "auto",
    strategy_config: StrategyConfig | None = None,
    walk_config: WalkForwardConfig | None = None,
    strict_windows: bool = False,
    robustness_symbols: tuple[str, ...] = (),
    freshness_threshold_seconds: float = 120.0,
) -> PrimetimeAuditResult:
    repo_path = Path(repo_root)
    created_at = pd.Timestamp.utcnow()
    active_strategy = strategy_config or _app_default_strategy_config()
    asset_class = infer_asset_class(symbol)
    requested_walk = walk_config or default_walk_forward_config(interval, asset_class)
    effective_robustness_symbols = robustness_symbols or default_robustness_basket(symbol, asset_class)

    tests_passed, pytest_output = _run_command([sys.executable, "-m", "pytest", "-q"], repo_path)
    compile_passed, compile_output = _run_command([sys.executable, "-m", "compileall", "src", "app.py"], repo_path)

    historical_fetch_ok = False
    live_quote_ok = False
    export_smoke_ok = False
    artifact_smoke_ok = False

    data_config = DataConfig(symbol=symbol, interval=interval, limit=limit, provider=history_provider)
    model_config = ModelConfig(n_states=states)
    feature_columns = get_feature_columns(feature_pack)
    fetched = fetch_price_data(data_config)
    historical_fetch_ok = True
    feature_frame = build_feature_frame(fetched.frame, feature_columns=feature_columns)
    effective_walk_config, was_adjusted = (
        (requested_walk, False)
        if strict_windows
        else suggest_walk_forward_config(len(feature_frame), requested_walk)
    )
    model_strategy = replace(active_strategy, require_daily_confirmation=False, require_consensus_confirmation=False)
    comparison, results_by_state = compare_state_counts(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        interval=interval,
        model_config=model_config,
        walk_config=effective_walk_config,
        strategy_config=model_strategy,
    )
    confirmation_fetched = None
    confirmation_result = None
    if interval == "4hour" and active_strategy.require_daily_confirmation:
        confirmation_data_config = DataConfig(symbol=symbol, interval="1day", limit=limit, provider=history_provider)
        confirmation_fetched = fetch_price_data(confirmation_data_config)
        confirmation_feature_frame = build_feature_frame(confirmation_fetched.frame, feature_columns=feature_columns)
        confirmation_walk_config, _ = (
            (default_walk_forward_config("1day", infer_asset_class(symbol)), False)
            if strict_windows
            else suggest_walk_forward_config(len(confirmation_feature_frame), default_walk_forward_config("1day", infer_asset_class(symbol)))
        )
        _, confirmation_results_by_state = compare_state_counts(
            feature_frame=confirmation_feature_frame,
            feature_columns=feature_columns,
            interval="1day",
            model_config=model_config,
            walk_config=confirmation_walk_config,
            strategy_config=model_strategy,
        )
        results_by_state = {
            n_states: apply_higher_timeframe_confirmation(
                result,
                confirmation_results_by_state[n_states],
                interval=interval,
                strategy_config=active_strategy,
                confirmation_interval="1day",
            )
            for n_states, result in results_by_state.items()
        }
        comparison = summarize_state_count_results(results_by_state)
        confirmation_result = confirmation_results_by_state[states]

    selected_result = results_by_state[states]
    consensus = None
    consensus_mode_comparison = pd.DataFrame()
    if active_strategy.require_consensus_confirmation:
        consensus = run_consensus_diagnostics(
            symbol=symbol,
            interval=interval,
            limit=limit,
            history_provider=history_provider,
            feature_columns=feature_columns,
            model_config=model_config,
            strategy_config=replace(active_strategy, require_consensus_confirmation=False),
            auto_adjust_windows=not strict_windows,
        )
        consensus_mode_comparison = compare_consensus_gate_modes(
            selected_result,
            consensus,
            interval=interval,
            strategy_config=active_strategy,
        )
        selected_result = apply_consensus_confirmation(
            selected_result,
            consensus,
            interval=interval,
            strategy_config=active_strategy,
        )

    sweep_results = parameter_sweep(
        predictions=selected_result.predictions,
        n_states=states,
        base_config=active_strategy,
        sweep_config=SweepConfig(),
        interval=interval,
    )
    robustness = run_multi_asset_robustness(
        symbols=parse_symbol_list(effective_robustness_symbols),
        interval=interval,
        limit=limit,
        history_provider=history_provider,
        feature_columns=feature_columns,
        model_config=model_config,
        walk_config=effective_walk_config,
        strategy_config=active_strategy,
        auto_adjust_windows=not strict_windows,
    )
    nested_holdout = nested_holdout_evaluation(
        predictions=selected_result.predictions,
        n_states=states,
        base_config=active_strategy,
        interval=interval,
        outer_holdout_folds=1,
    )
    quote = fetch_live_quote(fetched.resolved_symbol)
    live_quote_ok = True
    now_utc = pd.Timestamp.now(tz="UTC")
    live_quote_age_seconds = (
        float((now_utc - quote.timestamp).total_seconds())
        if quote.timestamp is not None
        else None
    )
    action_plan = build_execution_plan(
        latest_row=selected_result.predictions.iloc[-1].to_dict(),
        interval=interval,
        live_price=quote.price,
    )
    strategy_gates = build_promotion_gate_rows(
        metrics=selected_result.metrics,
        bootstrap=selected_result.bootstrap,
        state_stability=selected_result.state_stability,
        robustness=robustness,
        baseline_comparison=selected_result.baseline_comparison,
        interval=interval,
        available_rows=len(feature_frame),
        walk_adjusted=was_adjusted,
        fold_count=int(len(selected_result.fold_diagnostics)),
        nested_holdout=nested_holdout,
    )
    strategy_summary = summarize_promotion_gates(strategy_gates)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        exported = export_signal_report(
            selected_result.predictions,
            symbol=symbol,
            interval=interval,
            export_dir=tmp_path / "exports",
        )
        export_smoke_ok = exported["csv"].exists() and exported["json"].exists()
        notes = build_research_notes(selected_result, comparison)
        bundle = write_run_artifact_bundle(
            symbol=symbol,
            resolved_symbol=fetched.resolved_symbol,
            interval=interval,
            data_url=fetched.source_url,
            raw_frame=fetched.frame,
            feature_frame=feature_frame,
            data_config=data_config,
            model_config=model_config,
            walk_config=effective_walk_config,
            strategy_config=active_strategy,
            selected_result=selected_result,
            comparison=comparison,
            sweep_results=sweep_results,
            notes=notes,
            robustness=robustness,
            feature_columns=feature_columns,
            consensus_members=consensus.members if consensus is not None else None,
            consensus_timeline=consensus.timeline if consensus is not None else None,
            consensus_summary=consensus.summary if consensus is not None else None,
            consensus_mode_comparison=consensus_mode_comparison,
            nested_holdout_summary=pd.DataFrame([nested_holdout]),
            export_dir=tmp_path / "artifacts",
        )
        artifact_smoke_ok = bundle.manifest_path.exists()

    blind_oos_only = bool(selected_result.predictions.get("is_blind_oos", pd.Series(dtype=bool)).fillna(False).all())
    platform_gates = build_platform_gate_rows(
        tests_passed=tests_passed,
        compile_passed=compile_passed,
        historical_fetch_ok=historical_fetch_ok,
        live_quote_ok=live_quote_ok,
        live_quote_age_seconds=live_quote_age_seconds,
        freshness_threshold_seconds=freshness_threshold_seconds,
        export_smoke_ok=export_smoke_ok,
        artifact_smoke_ok=artifact_smoke_ok,
        blind_oos_only=blind_oos_only,
    )
    platform_summary = summarize_platform_gates(platform_gates)
    report_summary = summarize_primetime_report(
        platform_summary=platform_summary,
        strategy_summary=strategy_summary,
        action_plan=action_plan,
    )

    return PrimetimeAuditResult(
        created_at_utc=created_at.isoformat(),
        symbol=symbol,
        resolved_symbol=fetched.resolved_symbol,
        interval=interval,
        feature_pack=feature_pack,
        historical_provider=fetched.provider,
        historical_provider_note=fetched.provider_note,
        raw_rows=len(fetched.frame),
        usable_rows=len(feature_frame),
        walk_adjusted=was_adjusted,
        fold_count=int(len(selected_result.fold_diagnostics)),
        live_quote_price=quote.price if live_quote_ok else None,
        live_quote_age_seconds=live_quote_age_seconds,
        action_plan=action_plan,
        platform_gates=platform_gates,
        strategy_gates=strategy_gates,
        platform_summary=platform_summary,
        strategy_summary=strategy_summary,
        report_summary=report_summary,
        nested_holdout=nested_holdout,
        pytest_output=pytest_output,
        compile_output=compile_output,
    )


def write_primetime_audit_report(
    audit: PrimetimeAuditResult,
    *,
    output_dir: str | Path = "artifacts/primetime",
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + audit.resolved_symbol.lower()
    report_dir = root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at_utc": audit.created_at_utc,
        "symbol": audit.symbol,
        "resolved_symbol": audit.resolved_symbol,
        "interval": audit.interval,
        "feature_pack": audit.feature_pack,
        "historical_provider": audit.historical_provider,
        "historical_provider_note": audit.historical_provider_note,
        "raw_rows": audit.raw_rows,
        "usable_rows": audit.usable_rows,
        "walk_adjusted": audit.walk_adjusted,
        "fold_count": audit.fold_count,
        "live_quote_price": audit.live_quote_price,
        "live_quote_age_seconds": audit.live_quote_age_seconds,
        "action_plan": audit.action_plan,
        "platform_summary": dict(audit.platform_summary),
        "strategy_summary": dict(audit.strategy_summary),
        "report_summary": dict(audit.report_summary),
        "nested_holdout": dict(audit.nested_holdout),
        "platform_gates": audit.platform_gates.to_dict(orient="records"),
        "strategy_gates": audit.strategy_gates.to_dict(orient="records"),
        "pytest_output": audit.pytest_output,
        "compile_output": audit.compile_output,
    }
    json_path = report_dir / "primetime_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Primetime Audit",
        "",
        f"- Verdict: **{audit.report_summary['verdict']}**",
        f"- Platform: **{audit.platform_summary['verdict']}**",
        f"- Strategy: **{audit.strategy_summary['verdict']}**",
        f"- Current action: **{audit.action_plan['action']}**",
        f"- Symbol / interval: `{audit.resolved_symbol}` / `{audit.interval}`",
        f"- Feature pack: `{audit.feature_pack}`",
        f"- Historical provider: `{audit.historical_provider}`",
        f"- Historical provider note: `{audit.historical_provider_note}`",
        f"- Raw rows: `{audit.raw_rows}`",
        f"- Usable rows: `{audit.usable_rows}`",
        f"- Walk adjusted: `{audit.walk_adjusted}`",
        f"- Fold count: `{audit.fold_count}`",
        f"- Live quote price: `{audit.live_quote_price}`",
        f"- Live quote age seconds: `{audit.live_quote_age_seconds}`",
        "",
        "## Report Summary",
        "",
        audit.report_summary["summary"],
        "",
        "## Platform Gates",
        "",
        "```text",
        audit.platform_gates.to_string(index=False),
        "```",
        "",
        "## Strategy Gates",
        "",
        "```text",
        audit.strategy_gates.to_string(index=False),
        "```",
        "",
        "## Action Plan",
        "",
        f"- Summary: {audit.action_plan['summary']}",
        f"- Entry guide: {audit.action_plan['entry_guide']}",
        f"- Timing note: {audit.action_plan['timing_note']}",
        "",
    ]
    markdown_path = report_dir / "primetime_audit.md"
    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")
    return report_dir
