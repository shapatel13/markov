from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from markov_regime.config import (
    DataConfig,
    Interval,
    ModelConfig,
    StrategyConfig,
    SweepConfig,
    WalkForwardConfig,
    default_walk_forward_config,
)
from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.data import DataFetchResult, fetch_price_data
from markov_regime.features import FEATURE_COLUMNS, build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.research_notes import build_research_notes
from markov_regime.robustness import parse_symbol_list
from markov_regime.strategy import parameter_sweep
from markov_regime.walkforward import run_walk_forward, suggest_walk_forward_config


RESULT_COLUMNS: tuple[str, ...] = (
    "created_at_utc",
    "experiment_id",
    "symbol",
    "interval",
    "feature_pack",
    "n_states",
    "posterior_threshold",
    "min_hold_bars",
    "cooldown_bars",
    "required_confirmations",
    "confidence_gap",
    "raw_rows",
    "usable_rows",
    "walk_adjusted",
    "sharpe",
    "bootstrap_sharpe_lower",
    "bootstrap_sharpe_upper",
    "annualized_return",
    "max_drawdown",
    "trades",
    "confidence_coverage",
    "stability_score",
    "selection_sharpe",
    "confirmation_sharpe",
    "fold_consistency_gap",
    "benchmark_sharpe",
    "benchmark_annualized_return",
    "robustness_median_sharpe",
    "cost_break_bps",
    "research_score",
    "status",
    "artifact_path",
    "notes",
)

DEFAULT_RESEARCH_INTERVALS: tuple[Interval, ...] = ("4hour", "1day", "1hour")
INTERVAL_PRIORITY: dict[Interval, int] = {"4hour": 0, "1day": 1, "1hour": 2}


@dataclass(frozen=True)
class ResearchProgram:
    symbol: str = "BTCUSD"
    intervals: tuple[Interval, ...] = DEFAULT_RESEARCH_INTERVALS
    feature_packs: tuple[str, ...] = ("baseline", "trend", "volatility", "regime_mix")
    limit: int = 5000
    robustness_symbols: tuple[str, ...] = ("BTCUSD", "ETHUSD", "SOLUSD")
    state_counts: tuple[int, ...] = (5, 6, 7, 8, 9)
    posterior_thresholds: tuple[float, ...] = (0.6, 0.65, 0.7)
    min_hold_bars: tuple[int, ...] = (6, 12)
    cooldown_bars: tuple[int, ...] = (2, 4, 8)
    required_confirmations: tuple[int, ...] = (2, 3, 4)
    confidence_gap: float = 0.06
    max_candidates: int = 24
    artifact_top_k: int = 3
    auto_adjust_windows: bool = True


def _json_block_pattern() -> re.Pattern[str]:
    return re.compile(r"```json\s*(\{.*?\})\s*```", flags=re.DOTALL)


def _program_to_payload(program: ResearchProgram) -> dict[str, Any]:
    payload = asdict(program)
    payload["intervals"] = list(program.intervals)
    payload["feature_packs"] = list(program.feature_packs)
    payload["robustness_symbols"] = list(program.robustness_symbols)
    payload["state_counts"] = list(program.state_counts)
    payload["posterior_thresholds"] = list(program.posterior_thresholds)
    payload["min_hold_bars"] = list(program.min_hold_bars)
    payload["cooldown_bars"] = list(program.cooldown_bars)
    payload["required_confirmations"] = list(program.required_confirmations)
    return payload


def write_research_program(
    path: str | Path = "research_program.md",
    program: ResearchProgram | None = None,
) -> Path:
    current = program or ResearchProgram()
    target = Path(path)
    payload = json.dumps(_program_to_payload(current), indent=2)
    content = f"""# Research Program

Safe local autoresearch for the Markov regime app.

## Frozen Evaluator

- Do not mutate the leakage defenses or scoring harness during unattended runs.
- Keep these modules fixed unless a human explicitly asks to change methodology:
  - `src/markov_regime/walkforward.py`
  - `src/markov_regime/bootstrap.py`
  - `src/markov_regime/robustness.py`
  - `src/markov_regime/artifacts.py`

## Allowed Experiment Surface

- `src/markov_regime/features.py`
- `src/markov_regime/strategy.py`
- `src/markov_regime/config.py`
- this `research_program.md`

## Objective

Prefer robust `4hour` BTC research, keep `1day` as a slower confirmation lane,
and treat `1hour` mostly as a noisy baseline instead of the default optimization target.
Keep only changes that improve out-of-sample quality without weakening stability,
bootstrap confidence, or cross-asset robustness.

## Program Spec

```json
{payload}
```
"""
    target.write_text(content, encoding="utf-8")
    return target


def load_research_program(path: str | Path = "research_program.md") -> ResearchProgram:
    content = Path(path).read_text(encoding="utf-8")
    match = _json_block_pattern().search(content)
    if not match:
        raise ValueError("research_program.md is missing a fenced JSON program spec.")
    payload = json.loads(match.group(1))
    return ResearchProgram(
        symbol=str(payload.get("symbol", "BTCUSD")).upper(),
        intervals=tuple(str(item) for item in payload.get("intervals", list(DEFAULT_RESEARCH_INTERVALS))),
        feature_packs=tuple(str(item) for item in payload.get("feature_packs", list_feature_packs())),
        limit=int(payload.get("limit", 5000)),
        robustness_symbols=tuple(parse_symbol_list(payload.get("robustness_symbols", ["BTCUSD", "ETHUSD", "SOLUSD"]))),
        state_counts=tuple(int(item) for item in payload.get("state_counts", [5, 6, 7, 8, 9])),
        posterior_thresholds=tuple(float(item) for item in payload.get("posterior_thresholds", [0.6, 0.65, 0.7])),
        min_hold_bars=tuple(int(item) for item in payload.get("min_hold_bars", [6, 12])),
        cooldown_bars=tuple(int(item) for item in payload.get("cooldown_bars", [2, 4, 8])),
        required_confirmations=tuple(int(item) for item in payload.get("required_confirmations", [2, 3, 4])),
        confidence_gap=float(payload.get("confidence_gap", 0.06)),
        max_candidates=int(payload.get("max_candidates", 24)),
        artifact_top_k=int(payload.get("artifact_top_k", 3)),
        auto_adjust_windows=bool(payload.get("auto_adjust_windows", True)),
    )


def ensure_results_tsv(path: str | Path = "results.tsv") -> Path:
    target = Path(path)
    if not target.exists():
        header = "\t".join(RESULT_COLUMNS) + "\n"
        target.write_text(header, encoding="utf-8")
    return target


def _fetch_feature_context(
    *,
    symbol: str,
    interval: Interval,
    feature_columns: tuple[str, ...],
    limit: int,
    cache: dict[tuple[str, Interval, int, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]],
) -> tuple[DataFetchResult, pd.DataFrame]:
    key = (symbol, interval, limit, feature_columns)
    cached = cache.get(key)
    if cached is not None:
        return cached

    fetched = fetch_price_data(DataConfig(symbol=symbol, interval=interval, limit=limit))
    feature_frame = build_feature_frame(fetched.frame, feature_columns=feature_columns)
    cache[key] = (fetched, feature_frame)
    return fetched, feature_frame


def _resolve_walk_config(
    feature_frame: pd.DataFrame,
    interval: Interval,
    auto_adjust_windows: bool,
) -> tuple[WalkForwardConfig, bool]:
    requested = default_walk_forward_config(interval)
    if not auto_adjust_windows:
        return requested, False
    return suggest_walk_forward_config(len(feature_frame), requested)


def _bootstrap_interval(result, metric: str) -> tuple[float, float]:
    row = result.bootstrap.loc[result.bootstrap["metric"] == metric]
    if row.empty:
        return 0.0, 0.0
    return float(row.iloc[0]["lower"]), float(row.iloc[0]["upper"])


def _cost_break_bps(result) -> float:
    cost_row = result.cost_stress.loc[result.cost_stress["sharpe"] <= 0.0]
    if cost_row.empty:
        return float(result.cost_stress["cost_bps"].max()) if not result.cost_stress.empty else 0.0
    return float(cost_row.iloc[0]["cost_bps"])


def _fold_consistency_metrics(result) -> dict[str, float]:
    diagnostics = result.fold_diagnostics.sort_values("fold_id").reset_index(drop=True)
    if diagnostics.empty:
        return {
            "selection_sharpe": 0.0,
            "confirmation_sharpe": 0.0,
            "fold_consistency_gap": 0.0,
        }

    if len(diagnostics) == 1:
        sharpe = float(diagnostics.iloc[0]["sharpe"])
        return {
            "selection_sharpe": sharpe,
            "confirmation_sharpe": sharpe,
            "fold_consistency_gap": 0.0,
        }

    split_index = max(1, len(diagnostics) // 2)
    selection = diagnostics.iloc[:split_index]
    confirmation = diagnostics.iloc[split_index:]
    selection_sharpe = float(selection["sharpe"].median())
    confirmation_sharpe = float(confirmation["sharpe"].median()) if not confirmation.empty else selection_sharpe
    return {
        "selection_sharpe": selection_sharpe,
        "confirmation_sharpe": confirmation_sharpe,
        "fold_consistency_gap": float(abs(selection_sharpe - confirmation_sharpe)),
    }


def _score_candidate(result, robustness: pd.DataFrame) -> tuple[float, str, str, dict[str, float]]:
    bootstrap_lower, bootstrap_upper = _bootstrap_interval(result, "sharpe")
    stability = float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0
    benchmark_gap = result.metrics["annualized_return"] - result.benchmark_metrics["annualized_return"]
    cost_break = _cost_break_bps(result)
    ok_rows = robustness.loc[robustness["status"] == "ok"] if "status" in robustness.columns else pd.DataFrame()
    robustness_median = float(ok_rows["sharpe"].median()) if not ok_rows.empty else float("nan")
    consistency = _fold_consistency_metrics(result)

    score = float(result.metrics["sharpe"])
    score += 0.35 * bootstrap_lower
    score += 0.75 * (stability - 0.5)
    score += 2.0 * benchmark_gap
    score += 0.2 * min(cost_break, 20.0) / 20.0
    score += 0.25 * consistency["confirmation_sharpe"]
    score -= 0.15 * consistency["fold_consistency_gap"]
    if pd.notna(robustness_median):
        score += 0.15 * robustness_median
    if result.metrics["trades"] < 3:
        score -= 0.75
    if bootstrap_lower <= 0.0:
        score -= 0.5
    if stability < 0.45:
        score -= 0.5

    notes: list[str] = []
    if result.metrics["sharpe"] <= 0.0:
        notes.append("negative_sharpe")
    if bootstrap_lower <= 0.0:
        notes.append("bootstrap_crosses_zero")
    if stability < 0.5:
        notes.append("unstable_states")
    if result.metrics["trades"] < 3:
        notes.append("too_few_trades")
    if pd.notna(robustness_median) and robustness_median <= 0.0:
        notes.append("weak_cross_asset_robustness")
    if consistency["confirmation_sharpe"] <= 0.0:
        notes.append("late_fold_confirmation_weak")
    if consistency["fold_consistency_gap"] > 1.0:
        notes.append("fold_performance_inconsistent")

    if (
        result.metrics["sharpe"] > 0.0
        and bootstrap_lower > 0.0
        and stability >= 0.5
        and consistency["confirmation_sharpe"] > 0.0
        and (pd.isna(robustness_median) or robustness_median > 0.0)
    ):
        status = "keep"
    elif result.metrics["sharpe"] > 0.0:
        status = "candidate"
    else:
        status = "discard"
    return score, status, ",".join(notes) if notes else "passed_gates", consistency


def _single_state_comparison_row(result, n_states: int) -> pd.DataFrame:
    sharpe_lower, sharpe_upper = _bootstrap_interval(result, "sharpe")
    return pd.DataFrame(
        [
            {
                "n_states": n_states,
                "sharpe": result.metrics["sharpe"],
                "annualized_return": result.metrics["annualized_return"],
                "max_drawdown": result.metrics["max_drawdown"],
                "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                "mean_bic": float(result.fold_diagnostics["bic"].mean()) if "bic" in result.fold_diagnostics else 0.0,
                "mean_aic": float(result.fold_diagnostics["aic"].mean()) if "aic" in result.fold_diagnostics else 0.0,
                "confidence_coverage": result.metrics["confidence_coverage"],
                "converged_ratio": result.converged_ratio,
                "bootstrap_sharpe_lower": sharpe_lower,
                "bootstrap_sharpe_upper": sharpe_upper,
            }
        ]
    )


def run_timeframe_comparison(
    *,
    symbol: str,
    limit: int,
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    feature_pack: str = "baseline",
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    auto_adjust_windows: bool = True,
    intervals: tuple[Interval, ...] = DEFAULT_RESEARCH_INTERVALS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, Interval, int, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]] = {}

    for interval in intervals:
        try:
            fetched, feature_frame = _fetch_feature_context(
                symbol=symbol,
                interval=interval,
                feature_columns=feature_columns,
                limit=limit,
                cache=cache,
            )
            walk_config, was_adjusted = _resolve_walk_config(feature_frame, interval, auto_adjust_windows)
            result = run_walk_forward(
                feature_frame=feature_frame,
                feature_columns=feature_columns,
                interval=interval,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=strategy_config,
            )
            bootstrap_lower, bootstrap_upper = _bootstrap_interval(result, "sharpe")
            rows.append(
                {
                    "interval": interval,
                    "feature_pack": feature_pack,
                    "status": "ok",
                    "resolved_symbol": fetched.resolved_symbol,
                    "raw_rows": len(fetched.frame),
                    "usable_rows": len(feature_frame),
                    "walk_adjusted": was_adjusted,
                    "train_bars": walk_config.train_bars,
                    "validate_bars": walk_config.validate_bars,
                    "test_bars": walk_config.test_bars,
                    "sharpe": result.metrics["sharpe"],
                    "annualized_return": result.metrics["annualized_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "trades": result.metrics["trades"],
                    "confidence_coverage": result.metrics["confidence_coverage"],
                    "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                    "benchmark_sharpe": result.benchmark_metrics["sharpe"],
                    "bootstrap_sharpe_lower": bootstrap_lower,
                    "bootstrap_sharpe_upper": bootstrap_upper,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "interval": interval,
                    "feature_pack": feature_pack,
                    "status": "error",
                    "error": str(exc),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty or "interval" not in frame.columns:
        return frame
    return frame.sort_values(
        by="interval",
        key=lambda values: values.map(lambda item: INTERVAL_PRIORITY.get(str(item), len(INTERVAL_PRIORITY))),
        kind="stable",
    ).reset_index(drop=True)


def run_feature_pack_comparison(
    *,
    price_frame: pd.DataFrame,
    interval: Interval,
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    feature_packs: tuple[str, ...] | None = None,
    auto_adjust_windows: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_packs = feature_packs or list_feature_packs()

    for feature_pack in selected_packs:
        feature_columns = get_feature_columns(feature_pack)
        try:
            feature_frame = build_feature_frame(price_frame, feature_columns=feature_columns)
            walk_config, was_adjusted = _resolve_walk_config(feature_frame, interval, auto_adjust_windows)
            result = run_walk_forward(
                feature_frame=feature_frame,
                feature_columns=feature_columns,
                interval=interval,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=strategy_config,
            )
            bootstrap_lower, bootstrap_upper = _bootstrap_interval(result, "sharpe")
            consistency = _fold_consistency_metrics(result)
            rows.append(
                {
                    "feature_pack": feature_pack,
                    "status": "ok",
                    "usable_rows": len(feature_frame),
                    "walk_adjusted": was_adjusted,
                    "train_bars": walk_config.train_bars,
                    "validate_bars": walk_config.validate_bars,
                    "test_bars": walk_config.test_bars,
                    "sharpe": result.metrics["sharpe"],
                    "annualized_return": result.metrics["annualized_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "trades": result.metrics["trades"],
                    "confidence_coverage": result.metrics["confidence_coverage"],
                    "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                    "benchmark_sharpe": result.benchmark_metrics["sharpe"],
                    "bootstrap_sharpe_lower": bootstrap_lower,
                    "bootstrap_sharpe_upper": bootstrap_upper,
                    **consistency,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature_pack": feature_pack,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def _candidate_priority(candidate: tuple[Interval, str, int, float, int, int, int]) -> tuple[object, ...]:
    interval, feature_pack, n_states, posterior_threshold, min_hold_bars, cooldown_bars, required_confirmations = candidate
    interval_rank = INTERVAL_PRIORITY.get(interval, len(INTERVAL_PRIORITY))
    feature_rank = list_feature_packs().index(feature_pack) if feature_pack in list_feature_packs() else len(list_feature_packs())
    return (
        interval_rank,
        feature_rank,
        abs(n_states - 6),
        abs(posterior_threshold - 0.65),
        -min_hold_bars,
        abs(cooldown_bars - 4),
        abs(required_confirmations - 3),
    )


def _candidate_grid(program: ResearchProgram) -> list[tuple[Interval, str, int, float, int, int, int]]:
    grid = list(
        product(
            program.intervals,
            program.feature_packs,
            program.state_counts,
            program.posterior_thresholds,
            program.min_hold_bars,
            program.cooldown_bars,
            program.required_confirmations,
        )
    )
    grid.sort(key=_candidate_priority)
    return grid[: program.max_candidates]


def run_autoresearch(
    *,
    program: ResearchProgram,
    results_path: str | Path = "results.tsv",
) -> pd.DataFrame:
    ensure_results_tsv(results_path)
    cache: dict[tuple[str, Interval, int, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]] = {}
    rows: list[dict[str, object]] = []
    contexts: dict[str, dict[str, object]] = {}
    created_at = pd.Timestamp.utcnow()

    for candidate_index, candidate in enumerate(_candidate_grid(program), start=1):
        interval, feature_pack, n_states, posterior_threshold, min_hold_bars, cooldown_bars, required_confirmations = candidate
        experiment_id = f"{created_at.strftime('%Y%m%d_%H%M%S')}_{candidate_index:03d}"
        candidate_feature_columns = get_feature_columns(feature_pack)
        try:
            fetched, feature_frame = _fetch_feature_context(
                symbol=program.symbol,
                interval=interval,
                feature_columns=candidate_feature_columns,
                limit=program.limit,
                cache=cache,
            )
            walk_config, was_adjusted = _resolve_walk_config(feature_frame, interval, program.auto_adjust_windows)
            strategy_config = StrategyConfig(
                posterior_threshold=posterior_threshold,
                min_hold_bars=min_hold_bars,
                cooldown_bars=cooldown_bars,
                required_confirmations=required_confirmations,
                confidence_gap=program.confidence_gap,
            )
            model_config = ModelConfig(n_states=n_states)
            result = run_walk_forward(
                feature_frame=feature_frame,
                feature_columns=candidate_feature_columns,
                interval=interval,
                model_config=model_config,
                walk_config=walk_config,
                strategy_config=strategy_config,
            )

            robustness_rows: list[dict[str, object]] = []
            for robust_symbol in program.robustness_symbols:
                try:
                    robust_fetched, robust_features = _fetch_feature_context(
                        symbol=robust_symbol,
                        interval=interval,
                        feature_columns=candidate_feature_columns,
                        limit=program.limit,
                        cache=cache,
                    )
                    robust_walk_config, _ = _resolve_walk_config(robust_features, interval, program.auto_adjust_windows)
                    robust_result = run_walk_forward(
                        feature_frame=robust_features,
                        feature_columns=candidate_feature_columns,
                        interval=interval,
                        model_config=model_config,
                        walk_config=robust_walk_config,
                        strategy_config=strategy_config,
                    )
                    robustness_rows.append(
                        {
                            "symbol": robust_symbol,
                            "resolved_symbol": robust_fetched.resolved_symbol,
                            "status": "ok",
                            "sharpe": robust_result.metrics["sharpe"],
                        }
                    )
                except Exception as robust_exc:
                    robustness_rows.append(
                        {
                            "symbol": robust_symbol,
                            "resolved_symbol": robust_symbol,
                            "status": "error",
                            "error": str(robust_exc),
                        }
                    )
            robustness = pd.DataFrame(robustness_rows)
            score, status, notes, consistency = _score_candidate(result, robustness)
            bootstrap_lower, bootstrap_upper = _bootstrap_interval(result, "sharpe")
            ok_rows = robustness.loc[robustness["status"] == "ok"] if "status" in robustness.columns else pd.DataFrame()
            robustness_median = float(ok_rows["sharpe"].median()) if not ok_rows.empty else float("nan")

            rows.append(
                {
                    "created_at_utc": created_at.isoformat(),
                    "experiment_id": experiment_id,
                    "symbol": program.symbol,
                    "interval": interval,
                    "feature_pack": feature_pack,
                    "n_states": n_states,
                    "posterior_threshold": posterior_threshold,
                    "min_hold_bars": min_hold_bars,
                    "cooldown_bars": cooldown_bars,
                    "required_confirmations": required_confirmations,
                    "confidence_gap": program.confidence_gap,
                    "raw_rows": len(fetched.frame),
                    "usable_rows": len(feature_frame),
                    "walk_adjusted": was_adjusted,
                    "sharpe": result.metrics["sharpe"],
                    "bootstrap_sharpe_lower": bootstrap_lower,
                    "bootstrap_sharpe_upper": bootstrap_upper,
                    "annualized_return": result.metrics["annualized_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "trades": result.metrics["trades"],
                    "confidence_coverage": result.metrics["confidence_coverage"],
                    "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                    "selection_sharpe": consistency["selection_sharpe"],
                    "confirmation_sharpe": consistency["confirmation_sharpe"],
                    "fold_consistency_gap": consistency["fold_consistency_gap"],
                    "benchmark_sharpe": result.benchmark_metrics["sharpe"],
                    "benchmark_annualized_return": result.benchmark_metrics["annualized_return"],
                    "robustness_median_sharpe": robustness_median,
                    "cost_break_bps": _cost_break_bps(result),
                    "research_score": score,
                    "status": status,
                    "artifact_path": "",
                    "notes": notes,
                }
            )
            contexts[experiment_id] = {
                "feature_pack": feature_pack,
                "feature_columns": candidate_feature_columns,
                "fetched": fetched,
                "feature_frame": feature_frame,
                "model_config": model_config,
                "walk_config": walk_config,
                "strategy_config": strategy_config,
                "selected_result": result,
                "robustness": robustness,
            }
        except Exception as exc:
            rows.append(
                {
                    "created_at_utc": created_at.isoformat(),
                    "experiment_id": experiment_id,
                    "symbol": program.symbol,
                    "interval": interval,
                    "feature_pack": feature_pack,
                    "n_states": n_states,
                    "posterior_threshold": posterior_threshold,
                    "min_hold_bars": min_hold_bars,
                    "cooldown_bars": cooldown_bars,
                    "required_confirmations": required_confirmations,
                    "confidence_gap": program.confidence_gap,
                    "raw_rows": 0,
                    "usable_rows": 0,
                    "walk_adjusted": False,
                    "sharpe": 0.0,
                    "bootstrap_sharpe_lower": 0.0,
                    "bootstrap_sharpe_upper": 0.0,
                    "annualized_return": 0.0,
                    "max_drawdown": 0.0,
                    "trades": 0.0,
                    "confidence_coverage": 0.0,
                    "stability_score": 0.0,
                    "selection_sharpe": 0.0,
                    "confirmation_sharpe": 0.0,
                    "fold_consistency_gap": 0.0,
                    "benchmark_sharpe": 0.0,
                    "benchmark_annualized_return": 0.0,
                    "robustness_median_sharpe": 0.0,
                    "cost_break_bps": 0.0,
                    "research_score": -999.0,
                    "status": "error",
                    "artifact_path": "",
                    "notes": str(exc),
                }
            )

    leaderboard = pd.DataFrame(rows).sort_values(["research_score", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    artifact_candidates = leaderboard.loc[leaderboard["status"].isin(["keep", "candidate", "discard"])].head(program.artifact_top_k)
    for row in artifact_candidates.itertuples(index=False):
        context = contexts.get(str(row.experiment_id))
        if context is None:
            continue
        candidate_feature_columns = tuple(context["feature_columns"])
        comparison = _single_state_comparison_row(context["selected_result"], int(row.n_states))
        sweep_config = SweepConfig(
            posterior_thresholds=tuple(sorted({max(0.5, round(float(row.posterior_threshold) - 0.05, 2)), float(row.posterior_threshold), min(0.9, round(float(row.posterior_threshold) + 0.05, 2))})),
            min_hold_bars=tuple(sorted({max(1, int(row.min_hold_bars) // 2), int(row.min_hold_bars), int(row.min_hold_bars) + 4})),
            cooldown_bars=tuple(sorted({max(0, int(row.cooldown_bars) - 2), int(row.cooldown_bars), int(row.cooldown_bars) + 2})),
            required_confirmations=tuple(sorted({max(1, int(row.required_confirmations) - 1), int(row.required_confirmations), int(row.required_confirmations) + 1})),
        )
        sweep_results = parameter_sweep(
            predictions=context["selected_result"].predictions,
            n_states=int(row.n_states),
            base_config=context["strategy_config"],
            sweep_config=sweep_config,
            interval=row.interval,
        )
        notes = build_research_notes(context["selected_result"], comparison)
        notes.append(f"feature_pack={context['feature_pack']}")
        notes.append(f"research_score={float(row.research_score):.4f}")
        artifact = write_run_artifact_bundle(
            symbol=str(row.symbol),
            resolved_symbol=context["fetched"].resolved_symbol,
            interval=str(row.interval),
            data_url=context["fetched"].source_url,
            raw_frame=context["fetched"].frame,
            feature_frame=context["feature_frame"],
            data_config=DataConfig(symbol=str(row.symbol), interval=row.interval, limit=program.limit),
            model_config=context["model_config"],
            walk_config=context["walk_config"],
            strategy_config=context["strategy_config"],
            selected_result=context["selected_result"],
            comparison=comparison,
            sweep_results=sweep_results,
            notes=notes,
            robustness=context["robustness"],
            feature_columns=candidate_feature_columns,
            metadata={
                "experiment_id": row.experiment_id,
                "feature_pack": context["feature_pack"],
                "research_score": float(row.research_score),
                "status": row.status,
            },
        )
        leaderboard.loc[leaderboard["experiment_id"] == row.experiment_id, "artifact_path"] = str(artifact.root)

    output = leaderboard.loc[:, list(RESULT_COLUMNS)]

    results_file = Path(results_path)
    if results_file.exists() and results_file.stat().st_size > 0:
        existing = pd.read_csv(results_file, sep="\t")
        combined = pd.concat([existing, output], ignore_index=True) if not existing.empty else output
    else:
        combined = output
    combined.to_csv(results_file, sep="\t", index=False)
    return leaderboard
