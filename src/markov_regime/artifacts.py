from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from markov_regime.reporting import build_signal_report
from markov_regime.walkforward import WalkForwardResult


@dataclass(frozen=True)
class ArtifactBundle:
    run_id: str
    root: Path
    manifest_path: Path
    files: dict[str, Path]


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest_frame = frame.loc[:, [column for column in ["timestamp", "open", "high", "low", "close", "volume"] if column in frame.columns]].copy()
    if "timestamp" in digest_frame.columns:
        digest_frame["timestamp"] = pd.to_datetime(digest_frame["timestamp"]).astype(str)
    payload = digest_frame.to_json(orient="records")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def write_run_artifact_bundle(
    *,
    symbol: str,
    resolved_symbol: str,
    interval: str,
    data_url: str,
    raw_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    data_config: Any,
    model_config: Any,
    walk_config: Any,
    strategy_config: Any,
    selected_result: WalkForwardResult,
    comparison: pd.DataFrame,
    sweep_results: pd.DataFrame,
    notes: list[str],
    robustness: pd.DataFrame,
    feature_columns: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
    timeframe_comparison: pd.DataFrame | None = None,
    feature_pack_comparison: pd.DataFrame | None = None,
    consensus_members: pd.DataFrame | None = None,
    consensus_timeline: pd.DataFrame | None = None,
    consensus_summary: pd.DataFrame | None = None,
    export_dir: str | Path = "artifacts",
) -> ArtifactBundle:
    created_at = pd.Timestamp.utcnow()
    run_id = created_at.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.sha256(
        f"{resolved_symbol}|{interval}|{len(feature_frame)}|{selected_result.n_states}|{created_at.isoformat()}".encode("utf-8")
    ).hexdigest()[:8]
    root = Path(export_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    files["signal_report_csv"] = _write_table(build_signal_report(selected_result.predictions), root / "signal_report.csv")
    files["model_comparison_csv"] = _write_table(comparison, root / "model_comparison.csv")
    files["fold_diagnostics_csv"] = _write_table(selected_result.fold_diagnostics, root / "fold_diagnostics.csv")
    files["state_stability_csv"] = _write_table(selected_result.state_stability, root / "state_stability.csv")
    files["cost_stress_csv"] = _write_table(selected_result.cost_stress, root / "cost_stress.csv")
    files["bootstrap_csv"] = _write_table(selected_result.bootstrap, root / "bootstrap.csv")
    files["forward_returns_csv"] = _write_table(selected_result.forward_returns, root / "forward_returns.csv")
    files["guardrails_csv"] = _write_table(selected_result.guardrail_summary, root / "guardrails.csv")
    if selected_result.trade_log is not None and not selected_result.trade_log.empty:
        files["trade_log_csv"] = _write_table(selected_result.trade_log, root / "trade_log.csv")
    if selected_result.trade_summary is not None and not selected_result.trade_summary.empty:
        files["trade_summary_csv"] = _write_table(selected_result.trade_summary, root / "trade_summary.csv")
    if selected_result.consensus_summary is not None and not selected_result.consensus_summary.empty:
        files["consensus_gate_summary_csv"] = _write_table(selected_result.consensus_summary, root / "consensus_gate_summary.csv")
    if selected_result.confirmation_summary is not None and not selected_result.confirmation_summary.empty:
        files["confirmation_summary_csv"] = _write_table(selected_result.confirmation_summary, root / "confirmation_summary.csv")
    files["sweep_results_csv"] = _write_table(sweep_results, root / "sweep_results.csv")
    files["robustness_csv"] = _write_table(robustness, root / "robustness.csv")
    if timeframe_comparison is not None and not timeframe_comparison.empty:
        files["timeframe_comparison_csv"] = _write_table(timeframe_comparison, root / "timeframe_comparison.csv")
    if feature_pack_comparison is not None and not feature_pack_comparison.empty:
        files["feature_pack_comparison_csv"] = _write_table(feature_pack_comparison, root / "feature_pack_comparison.csv")
    if consensus_members is not None and not consensus_members.empty:
        files["consensus_members_csv"] = _write_table(consensus_members, root / "consensus_members.csv")
    if consensus_timeline is not None and not consensus_timeline.empty:
        files["consensus_timeline_csv"] = _write_table(consensus_timeline, root / "consensus_timeline.csv")
    if consensus_summary is not None and not consensus_summary.empty:
        files["consensus_summary_csv"] = _write_table(consensus_summary, root / "consensus_summary.csv")

    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "symbol": symbol,
        "resolved_symbol": resolved_symbol,
        "interval": interval,
        "data_url": data_url,
        "git_head": _safe_git_head(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "data_config": asdict(data_config),
        "model_config": asdict(model_config),
        "walk_config": asdict(walk_config),
        "strategy_config": asdict(strategy_config),
        "feature_columns": list(feature_columns) if feature_columns is not None else None,
        "data_summary": {
            "raw_rows": int(len(raw_frame)),
            "feature_rows": int(len(feature_frame)),
            "raw_start": str(pd.to_datetime(raw_frame["timestamp"].iloc[0])),
            "raw_end": str(pd.to_datetime(raw_frame["timestamp"].iloc[-1])),
            "feature_start": str(pd.to_datetime(feature_frame["timestamp"].iloc[0])),
            "feature_end": str(pd.to_datetime(feature_frame["timestamp"].iloc[-1])),
            "raw_fingerprint": _frame_fingerprint(raw_frame),
            "feature_fingerprint": _frame_fingerprint(feature_frame),
        },
        "selected_result": {
            "n_states": selected_result.n_states,
            "metrics": selected_result.metrics,
            "benchmark_metrics": selected_result.benchmark_metrics,
            "converged_ratio": selected_result.converged_ratio,
        },
        "notes": notes,
        "files": {name: str(path) for name, path in files.items()},
    }
    if metadata:
        manifest["metadata"] = metadata
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ArtifactBundle(run_id=run_id, root=root, manifest_path=manifest_path, files=files)
