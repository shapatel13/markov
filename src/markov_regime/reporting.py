from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_signal_report(predictions: pd.DataFrame) -> pd.DataFrame:
    report = predictions.copy()
    report["signal_label"] = report["signal_position"].map({1: "long", 0: "flat", -1: "short"}).fillna("flat")
    report["timestamp"] = pd.to_datetime(report["timestamp"])
    preferred_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "canonical_state",
        "max_posterior",
        "confidence_gap",
        "candidate_action",
        "signal_position",
        "signal_label",
        "guardrail_reason",
        "bar_return",
        "gross_strategy_return",
        "transaction_cost",
        "net_strategy_return",
        "strategy_wealth",
        "fold_id",
    ]
    existing = [column for column in preferred_columns if column in report.columns]
    remaining = [column for column in report.columns if column not in existing]
    return report.loc[:, existing + remaining]


def export_signal_report(
    predictions: pd.DataFrame,
    symbol: str,
    interval: str,
    export_dir: str | Path = "exports",
) -> dict[str, Path]:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    report = build_signal_report(predictions)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"{symbol.lower()}_{interval}_{timestamp}_signal_report"
    csv_path = export_path / f"{stem}.csv"
    json_path = export_path / f"{stem}.json"

    report.to_csv(csv_path, index=False)
    json_ready = report.assign(timestamp=lambda frame: frame["timestamp"].astype(str))
    json_ready.to_json(json_path, orient="records", indent=2)

    return {"csv": csv_path, "json": json_path}

