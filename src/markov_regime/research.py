from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from markov_regime.config import (
    DataConfig,
    HistoricalProvider,
    Interval,
    ModelConfig,
    StrategyConfig,
    SweepConfig,
    WalkForwardConfig,
    default_walk_forward_config,
)
from markov_regime.artifacts import write_run_artifact_bundle
from markov_regime.consensus import apply_consensus_confirmation, run_consensus_diagnostics
from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.data import DataFetchResult, fetch_price_data
from markov_regime.features import FEATURE_COLUMNS, build_feature_frame, get_feature_columns, list_feature_packs
from markov_regime.interpretation import build_promotion_gate_rows, recommend_strategy_engine, summarize_promotion_gates
from markov_regime.research_notes import build_research_notes
from markov_regime.robustness import parse_symbol_list, run_multi_asset_robustness
from markov_regime.strategy import parameter_sweep, replay_strategy
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
    "outer_holdout_folds",
    "outer_holdout_sharpe",
    "outer_holdout_annualized_return",
    "outer_holdout_trades",
    "selected_inner_posterior_threshold",
    "selected_inner_min_hold_bars",
    "selected_inner_cooldown_bars",
    "selected_inner_required_confirmations",
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
CANDIDATE_CONFIRMATION_MODES: tuple[str, ...] = ("off", "daily", "consensus_entry", "daily_consensus_entry")
CANDIDATE_SEARCH_COLUMNS: tuple[str, ...] = (
    "rank",
    "symbol",
    "interval",
    "provider",
    "feature_pack",
    "n_states",
    "shorting_mode",
    "confirmation_mode",
    "usable_rows",
    "walk_adjusted",
    "sharpe",
    "bootstrap_sharpe_lower",
    "annualized_return",
    "max_drawdown",
    "trades",
    "stability_score",
    "outer_holdout_sharpe",
    "ensemble_sharpe",
    "ensemble_outer_holdout_sharpe",
    "ensemble_trades",
    "ensemble_evaluated",
    "seed_median_sharpe",
    "seed_sharpe_std",
    "seed_median_stability",
    "seed_latest_candidate_share",
    "seed_avg_position_share",
    "seed_converged_ratio",
    "seed_member_count",
    "seed_evaluated",
    "robustness_median_sharpe",
    "robustness_evaluated",
    "best_baseline",
    "best_baseline_sharpe",
    "promotion_verdict",
    "engine_recommendation",
    "recommendation_detail",
    "candidate_score",
    "candidate_status",
    "notes",
)


@dataclass(frozen=True)
class ResearchProgram:
    symbol: str = "BTCUSD"
    intervals: tuple[Interval, ...] = DEFAULT_RESEARCH_INTERVALS
    feature_packs: tuple[str, ...] = ("baseline", "trend", "volatility", "regime_mix", "trend_context", "regime_context")
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
    outer_holdout_folds: int = 1


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
        outer_holdout_folds=int(payload.get("outer_holdout_folds", 1)),
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
    history_provider: HistoricalProvider,
    cache: dict[tuple[str, Interval, int, HistoricalProvider, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]],
) -> tuple[DataFetchResult, pd.DataFrame]:
    key = (symbol, interval, limit, history_provider, feature_columns)
    cached = cache.get(key)
    if cached is not None:
        return cached

    fetched = fetch_price_data(DataConfig(symbol=symbol, interval=interval, limit=limit, provider=history_provider))
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


def _maybe_apply_daily_confirmation(
    *,
    result,
    symbol: str,
    limit: int,
    history_provider: HistoricalProvider,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    auto_adjust_windows: bool,
    cache: dict[tuple[str, Interval, int, HistoricalProvider, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]],
    interval: Interval,
):
    if interval != "4hour" or not strategy_config.require_daily_confirmation:
        return result

    confirmation_strategy_config = replace(strategy_config, require_daily_confirmation=False)
    _, confirmation_features = _fetch_feature_context(
        symbol=symbol,
        interval="1day",
        feature_columns=feature_columns,
        limit=limit,
        history_provider=history_provider,
        cache=cache,
    )
    confirmation_walk_config, _ = _resolve_walk_config(confirmation_features, "1day", auto_adjust_windows)
    confirmation_result = run_walk_forward(
        feature_frame=confirmation_features,
        feature_columns=feature_columns,
        interval="1day",
        model_config=model_config,
        walk_config=confirmation_walk_config,
        strategy_config=confirmation_strategy_config,
    )
    return apply_higher_timeframe_confirmation(
        result,
        confirmation_result,
        interval=interval,
        strategy_config=strategy_config,
        confirmation_interval="1day",
    )


def _apply_optional_overlays(
    *,
    result,
    symbol: str,
    interval: Interval,
    limit: int,
    history_provider: HistoricalProvider,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    auto_adjust_windows: bool,
    cache: dict[tuple[str, Interval, int, HistoricalProvider, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]],
):
    adjusted = _maybe_apply_daily_confirmation(
        result=result,
        symbol=symbol,
        limit=limit,
        history_provider=history_provider,
        feature_columns=feature_columns,
        model_config=model_config,
        strategy_config=strategy_config,
        auto_adjust_windows=auto_adjust_windows,
        cache=cache,
        interval=interval,
    )
    if not strategy_config.require_consensus_confirmation:
        return adjusted

    consensus = run_consensus_diagnostics(
        symbol=symbol,
        interval=interval,
        limit=limit,
        history_provider=history_provider,
        feature_columns=feature_columns,
        model_config=model_config,
        strategy_config=replace(strategy_config, require_consensus_confirmation=False),
        auto_adjust_windows=auto_adjust_windows,
    )
    return apply_consensus_confirmation(
        adjusted,
        consensus,
        interval=interval,
        strategy_config=strategy_config,
    )


def _best_baseline_details(baseline_comparison: pd.DataFrame) -> tuple[str, float]:
    if baseline_comparison.empty:
        return "unavailable", float("nan")
    best_row = baseline_comparison.sort_values("sharpe", ascending=False).iloc[0]
    return str(best_row["baseline"]), float(best_row["sharpe"])


def _build_candidate_search_strategy_config(
    *,
    base_config: StrategyConfig,
    allow_short: bool,
    confirmation_mode: str,
) -> StrategyConfig:
    strategy_config = replace(
        base_config,
        allow_short=allow_short,
        require_daily_confirmation=False,
        require_consensus_confirmation=False,
        consensus_gate_mode="entry_only",
    )
    if confirmation_mode == "off":
        return strategy_config
    if confirmation_mode == "daily":
        return replace(strategy_config, require_daily_confirmation=True)
    if confirmation_mode == "consensus_entry":
        return replace(strategy_config, require_consensus_confirmation=True)
    if confirmation_mode == "daily_consensus_entry":
        return replace(strategy_config, require_daily_confirmation=True, require_consensus_confirmation=True)
    raise ValueError(f"Unsupported confirmation mode: {confirmation_mode}")


def _candidate_confirmation_priority(mode: str) -> int:
    order = {
        "off": 0,
        "daily": 1,
        "consensus_entry": 2,
        "daily_consensus_entry": 3,
    }
    return order.get(mode, len(order))


def _candidate_search_priority(candidate: tuple[str, int, bool, str]) -> tuple[object, ...]:
    feature_pack, n_states, allow_short, confirmation_mode = candidate
    pack_priority = [
        "mean_reversion",
        "trend",
        "baseline",
        "regime_mix",
        "atr_causal",
        "trend_context",
        "regime_context",
        "volatility",
        "trend_strength",
        "vol_surface",
        "regime_mix_v2",
    ]
    feature_rank = pack_priority.index(feature_pack) if feature_pack in pack_priority else len(pack_priority)
    return (
        abs(n_states - 8),
        _candidate_confirmation_priority(confirmation_mode),
        int(allow_short),
        feature_rank,
    )


def _candidate_search_grid(
    *,
    feature_packs: tuple[str, ...],
    state_counts: tuple[int, ...],
    short_modes: tuple[bool, ...],
    confirmation_modes: tuple[str, ...],
    max_candidates: int | None,
) -> list[tuple[str, int, bool, str]]:
    grid = list(product(feature_packs, state_counts, short_modes, confirmation_modes))
    grid.sort(key=_candidate_search_priority)
    return grid[:max_candidates] if max_candidates is not None else grid


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


def _empty_seed_metrics() -> dict[str, float | int | bool]:
    return {
        "seed_median_sharpe": float("nan"),
        "seed_sharpe_std": float("nan"),
        "seed_median_stability": float("nan"),
        "seed_latest_candidate_share": float("nan"),
        "seed_avg_position_share": float("nan"),
        "seed_converged_ratio": float("nan"),
        "seed_member_count": 0,
        "seed_evaluated": False,
    }


def _empty_ensemble_metrics() -> dict[str, float | bool]:
    return {
        "ensemble_sharpe": float("nan"),
        "ensemble_outer_holdout_sharpe": float("nan"),
        "ensemble_trades": float("nan"),
        "ensemble_evaluated": False,
    }


def _seed_robustness_metrics(consensus) -> dict[str, float | int | bool]:
    if consensus is None or consensus.members.empty or consensus.timeline.empty:
        return _empty_seed_metrics()

    members = consensus.members
    timeline = consensus.timeline
    latest = timeline.iloc[-1]
    return {
        "seed_median_sharpe": float(members["sharpe"].median()),
        "seed_sharpe_std": float(members["sharpe"].std(ddof=0) if len(members) > 1 else 0.0),
        "seed_median_stability": float(members["stability_score"].median()) if "stability_score" in members.columns else float("nan"),
        "seed_latest_candidate_share": float(latest.get("candidate_consensus_share", float("nan"))),
        "seed_avg_position_share": float(timeline["position_consensus_share"].mean()),
        "seed_converged_ratio": float(members["converged_ratio"].mean()) if "converged_ratio" in members.columns else float("nan"),
        "seed_member_count": int(len(members)),
        "seed_evaluated": True,
    }


def _score_candidate(
    result,
    robustness: pd.DataFrame,
    nested_summary: dict[str, object] | None = None,
    seed_metrics: dict[str, float | int | bool] | None = None,
    ensemble_metrics: dict[str, float | bool] | None = None,
) -> tuple[float, str, str, dict[str, float]]:
    bootstrap_lower, bootstrap_upper = _bootstrap_interval(result, "sharpe")
    stability = float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0
    benchmark_gap = result.metrics["annualized_return"] - result.benchmark_metrics["annualized_return"]
    cost_break = _cost_break_bps(result)
    ok_rows = robustness.loc[robustness["status"] == "ok"] if "status" in robustness.columns else pd.DataFrame()
    robustness_median = float(ok_rows["sharpe"].median()) if not ok_rows.empty else float("nan")
    consistency = _fold_consistency_metrics(result)
    outer_holdout_sharpe = float(nested_summary.get("outer_holdout_sharpe", 0.0)) if nested_summary else 0.0
    seed_context = seed_metrics or _empty_seed_metrics()
    seed_median_sharpe = float(seed_context.get("seed_median_sharpe", float("nan")))
    seed_sharpe_std = float(seed_context.get("seed_sharpe_std", float("nan")))
    seed_median_stability = float(seed_context.get("seed_median_stability", float("nan")))
    seed_latest_candidate_share = float(seed_context.get("seed_latest_candidate_share", float("nan")))
    seed_avg_position_share = float(seed_context.get("seed_avg_position_share", float("nan")))
    seed_converged_ratio = float(seed_context.get("seed_converged_ratio", float("nan")))
    seed_evaluated = bool(seed_context.get("seed_evaluated", False))
    ensemble_context = ensemble_metrics or _empty_ensemble_metrics()
    ensemble_sharpe = float(ensemble_context.get("ensemble_sharpe", float("nan")))
    ensemble_outer_holdout_sharpe = float(ensemble_context.get("ensemble_outer_holdout_sharpe", float("nan")))
    ensemble_trades = float(ensemble_context.get("ensemble_trades", float("nan")))
    ensemble_evaluated = bool(ensemble_context.get("ensemble_evaluated", False))

    score = float(result.metrics["sharpe"])
    score += 0.35 * bootstrap_lower
    score += 0.75 * (stability - 0.5)
    score += 2.0 * benchmark_gap
    score += 0.2 * min(cost_break, 20.0) / 20.0
    score += 0.25 * consistency["confirmation_sharpe"]
    score -= 0.15 * consistency["fold_consistency_gap"]
    score += 0.35 * outer_holdout_sharpe
    if pd.notna(robustness_median):
        score += 0.15 * robustness_median
    if seed_evaluated and pd.notna(seed_median_sharpe):
        score += 0.30 * seed_median_sharpe
    if seed_evaluated and pd.notna(seed_avg_position_share):
        score += 0.60 * (seed_avg_position_share - 0.5)
    if seed_evaluated and pd.notna(seed_latest_candidate_share):
        score += 0.40 * (seed_latest_candidate_share - 0.5)
    if seed_evaluated and pd.notna(seed_median_stability):
        score += 0.30 * (seed_median_stability - 0.5)
    if seed_evaluated and pd.notna(seed_sharpe_std):
        score -= 0.20 * seed_sharpe_std
    if seed_evaluated and pd.notna(seed_converged_ratio):
        score -= 0.30 * max(0.0, 1.0 - seed_converged_ratio)
    if ensemble_evaluated and pd.notna(ensemble_sharpe):
        score += 0.45 * ensemble_sharpe
    if ensemble_evaluated and pd.notna(ensemble_outer_holdout_sharpe):
        score += 0.30 * ensemble_outer_holdout_sharpe
    if ensemble_evaluated and pd.notna(ensemble_trades) and ensemble_trades < 3:
        score -= 0.25
    if result.metrics["trades"] < 3:
        score -= 0.75
    if bootstrap_lower <= 0.0:
        score -= 0.5
    if stability < 0.45:
        score -= 0.5
    if seed_evaluated and pd.notna(seed_latest_candidate_share) and seed_latest_candidate_share < 0.67:
        score -= 0.5
    if seed_evaluated and pd.notna(seed_median_sharpe) and seed_median_sharpe <= 0.0:
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
    if nested_summary and nested_summary.get("status") == "ok" and outer_holdout_sharpe <= 0.0:
        notes.append("outer_holdout_weak")
    if seed_evaluated and pd.notna(seed_median_sharpe) and seed_median_sharpe <= 0.0:
        notes.append("weak_seed_median_sharpe")
    if seed_evaluated and pd.notna(seed_latest_candidate_share) and seed_latest_candidate_share < 0.67:
        notes.append("weak_seed_consensus")
    if seed_evaluated and pd.notna(seed_sharpe_std) and seed_sharpe_std > 1.0:
        notes.append("seed_performance_dispersion_high")
    if seed_evaluated and pd.notna(seed_converged_ratio) and seed_converged_ratio < 1.0:
        notes.append("partial_seed_convergence")
    if ensemble_evaluated and pd.notna(ensemble_sharpe) and ensemble_sharpe <= 0.0:
        notes.append("weak_ensemble_sharpe")
    if ensemble_evaluated and pd.notna(ensemble_outer_holdout_sharpe) and ensemble_outer_holdout_sharpe <= 0.0:
        notes.append("weak_ensemble_outer_holdout")

    seed_keep_ok = (
        not seed_evaluated
        or (
            pd.notna(seed_median_sharpe)
            and seed_median_sharpe > 0.0
            and pd.notna(seed_latest_candidate_share)
            and seed_latest_candidate_share >= 0.67
            and (pd.isna(seed_converged_ratio) or seed_converged_ratio >= 0.95)
        )
    )
    ensemble_keep_ok = (
        not ensemble_evaluated
        or (
            pd.notna(ensemble_sharpe)
            and ensemble_sharpe > 0.0
            and (
                pd.isna(ensemble_outer_holdout_sharpe)
                or ensemble_outer_holdout_sharpe > 0.0
                or not nested_summary
                or nested_summary.get("status") != "ok"
            )
        )
    )
    if (
        result.metrics["sharpe"] > 0.0
        and bootstrap_lower > 0.0
        and stability >= 0.5
        and consistency["confirmation_sharpe"] > 0.0
        and (not nested_summary or nested_summary.get("status") != "ok" or outer_holdout_sharpe > 0.0)
        and (pd.isna(robustness_median) or robustness_median > 0.0)
        and seed_keep_ok
        and ensemble_keep_ok
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
    history_provider: HistoricalProvider = "auto",
    model_config: ModelConfig,
    strategy_config: StrategyConfig,
    feature_pack: str = "baseline",
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    auto_adjust_windows: bool = True,
    intervals: tuple[Interval, ...] = DEFAULT_RESEARCH_INTERVALS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, Interval, int, HistoricalProvider, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]] = {}

    for interval in intervals:
        try:
            fetched, feature_frame = _fetch_feature_context(
                symbol=symbol,
                interval=interval,
                feature_columns=feature_columns,
                limit=limit,
                history_provider=history_provider,
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
            result = _apply_optional_overlays(
                result=result,
                symbol=symbol,
                interval=interval,
                limit=limit,
                history_provider=history_provider,
                feature_columns=feature_columns,
                model_config=model_config,
                strategy_config=strategy_config,
                auto_adjust_windows=auto_adjust_windows,
                cache=cache,
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
    symbol: str | None = None,
    limit: int | None = None,
    history_provider: HistoricalProvider = "auto",
    feature_packs: tuple[str, ...] | None = None,
    auto_adjust_windows: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected_packs = feature_packs or list_feature_packs()
    cache: dict[tuple[str, Interval, int, HistoricalProvider, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]] = {}

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
            if symbol is not None and limit is not None:
                result = _apply_optional_overlays(
                    result=result,
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    history_provider=history_provider,
                    feature_columns=feature_columns,
                    model_config=model_config,
                    strategy_config=strategy_config,
                    auto_adjust_windows=auto_adjust_windows,
                    cache=cache,
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


def run_candidate_search(
    *,
    symbol: str,
    interval: Interval,
    limit: int,
    history_provider: HistoricalProvider = "auto",
    base_model_config: ModelConfig | None = None,
    base_strategy_config: StrategyConfig | None = None,
    feature_packs: tuple[str, ...] | None = None,
    state_counts: tuple[int, ...] = (5, 6, 7, 8, 9),
    short_modes: tuple[bool, ...] = (False, True),
    confirmation_modes: tuple[str, ...] = CANDIDATE_CONFIRMATION_MODES,
    robustness_symbols: tuple[str, ...] = ("BTCUSD", "ETHUSD", "SOLUSD"),
    auto_adjust_windows: bool = True,
    max_candidates: int | None = 32,
    robustness_top_k: int = 2,
    seed_robustness_top_k: int = 2,
) -> pd.DataFrame:
    model_config = base_model_config or ModelConfig()
    strategy_template = base_strategy_config or StrategyConfig()
    selected_feature_packs = feature_packs or tuple(list_feature_packs())
    cache: dict[tuple[str, Interval, int, HistoricalProvider, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]] = {}
    rows: list[dict[str, object]] = []
    contexts: dict[tuple[str, int, bool, str], dict[str, object]] = {}

    for feature_pack, n_states, allow_short, confirmation_mode in _candidate_search_grid(
        feature_packs=tuple(selected_feature_packs),
        state_counts=tuple(sorted(set(int(item) for item in state_counts))),
        short_modes=tuple(short_modes),
        confirmation_modes=tuple(confirmation_modes),
        max_candidates=max_candidates,
    ):
        feature_columns = get_feature_columns(feature_pack)
        try:
            fetched, feature_frame = _fetch_feature_context(
                symbol=symbol,
                interval=interval,
                feature_columns=feature_columns,
                limit=limit,
                history_provider=history_provider,
                cache=cache,
            )
            walk_config, was_adjusted = _resolve_walk_config(feature_frame, interval, auto_adjust_windows)
            strategy_config = _build_candidate_search_strategy_config(
                base_config=strategy_template,
                allow_short=allow_short,
                confirmation_mode=confirmation_mode,
            )
            current_model_config = replace(model_config, n_states=n_states)
            result = run_walk_forward(
                feature_frame=feature_frame,
                feature_columns=feature_columns,
                interval=interval,
                model_config=current_model_config,
                walk_config=walk_config,
                strategy_config=replace(
                    strategy_config,
                    require_daily_confirmation=False,
                    require_consensus_confirmation=False,
                ),
            )
            result = _apply_optional_overlays(
                result=result,
                symbol=symbol,
                interval=interval,
                limit=limit,
                history_provider=history_provider,
                feature_columns=feature_columns,
                model_config=current_model_config,
                strategy_config=strategy_config,
                auto_adjust_windows=auto_adjust_windows,
                cache=cache,
            )
            nested_summary = nested_holdout_evaluation(
                predictions=result.predictions,
                n_states=n_states,
                base_config=strategy_config,
                interval=interval,
                outer_holdout_folds=1,
            )
            candidate_score, candidate_status, notes, _ = _score_candidate(result, pd.DataFrame(), nested_summary)
            bootstrap_lower, _ = _bootstrap_interval(result, "sharpe")
            best_baseline, best_baseline_sharpe = _best_baseline_details(result.baseline_comparison)
            contexts[(feature_pack, n_states, allow_short, confirmation_mode)] = {
                "feature_columns": feature_columns,
                "model_config": current_model_config,
                "strategy_config": strategy_config,
                "walk_config": walk_config,
                "result": result,
                "was_adjusted": was_adjusted,
            }
            rows.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "provider": fetched.provider,
                    "feature_pack": feature_pack,
                    "n_states": n_states,
                    "shorting_mode": "long_short" if allow_short else "long_only",
                    "confirmation_mode": confirmation_mode,
                    "usable_rows": len(feature_frame),
                    "walk_adjusted": was_adjusted,
                    "sharpe": result.metrics["sharpe"],
                    "bootstrap_sharpe_lower": bootstrap_lower,
                    "annualized_return": result.metrics["annualized_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "trades": result.metrics["trades"],
                    "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                    "outer_holdout_sharpe": float(nested_summary.get("outer_holdout_sharpe", 0.0)),
                    **_empty_ensemble_metrics(),
                    **_empty_seed_metrics(),
                    "robustness_median_sharpe": float("nan"),
                    "robustness_evaluated": False,
                    "best_baseline": best_baseline,
                    "best_baseline_sharpe": best_baseline_sharpe,
                    "promotion_verdict": "Pending Robustness",
                    "engine_recommendation": "Pending robustness check",
                    "recommendation_detail": "Cross-asset robustness and multi-seed HMM checks are only run on the top ranked primary-symbol variants to keep the search tractable.",
                    "candidate_score": candidate_score,
                    "candidate_status": candidate_status,
                    "notes": f"{notes},robustness_not_evaluated,seed_robustness_not_evaluated" if notes else "robustness_not_evaluated,seed_robustness_not_evaluated",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "provider": history_provider,
                    "feature_pack": feature_pack,
                    "n_states": n_states,
                    "shorting_mode": "long_short" if allow_short else "long_only",
                    "confirmation_mode": confirmation_mode,
                    "usable_rows": 0,
                    "walk_adjusted": False,
                    "sharpe": float("nan"),
                    "bootstrap_sharpe_lower": float("nan"),
                    "annualized_return": float("nan"),
                    "max_drawdown": float("nan"),
                    "trades": 0.0,
                    "stability_score": float("nan"),
                    "outer_holdout_sharpe": float("nan"),
                    **_empty_ensemble_metrics(),
                    **_empty_seed_metrics(),
                    "robustness_median_sharpe": float("nan"),
                    "robustness_evaluated": False,
                    "best_baseline": "unavailable",
                    "best_baseline_sharpe": float("nan"),
                    "promotion_verdict": "Error",
                    "engine_recommendation": "Research failed",
                    "recommendation_detail": str(exc),
                    "candidate_score": -999.0,
                    "candidate_status": "error",
                    "notes": str(exc),
                }
            )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return pd.DataFrame(columns=CANDIDATE_SEARCH_COLUMNS)

    leaderboard = leaderboard.sort_values(
        ["candidate_score", "sharpe", "outer_holdout_sharpe"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    if robustness_top_k > 0 or seed_robustness_top_k > 0:
        top_k = min(max(int(robustness_top_k), int(seed_robustness_top_k)), len(leaderboard))

        def _evaluate_leaderboard_row(row_index: int) -> None:
            row = leaderboard.iloc[row_index]
            context = contexts.get(
                (
                    str(row["feature_pack"]),
                    int(row["n_states"]),
                    str(row["shorting_mode"]) == "long_short",
                    str(row["confirmation_mode"]),
                )
            )
            if context is None:
                return
            result = context["result"]
            robustness = pd.DataFrame()
            robustness_was_evaluated = bool(row_index < int(robustness_top_k))
            if robustness_was_evaluated:
                try:
                    robustness = run_multi_asset_robustness(
                        symbols=list(robustness_symbols),
                        interval=interval,
                        limit=limit,
                        history_provider=history_provider,
                        feature_columns=tuple(context["feature_columns"]),
                        model_config=context["model_config"],
                        walk_config=context["walk_config"],
                        strategy_config=context["strategy_config"],
                        auto_adjust_windows=auto_adjust_windows,
                    )
                except Exception:
                    robustness = pd.DataFrame()
                    robustness_was_evaluated = False
            seed_metrics = _empty_seed_metrics()
            ensemble_metrics = _empty_ensemble_metrics()
            if row_index < int(seed_robustness_top_k):
                try:
                    seed_consensus = run_consensus_diagnostics(
                        symbol=symbol,
                        interval=interval,
                        limit=limit,
                        history_provider=history_provider,
                        feature_columns=tuple(context["feature_columns"]),
                        model_config=context["model_config"],
                        strategy_config=context["strategy_config"],
                        auto_adjust_windows=auto_adjust_windows,
                        state_counts=(int(row["n_states"]),),
                    )
                    seed_metrics = _seed_robustness_metrics(seed_consensus)
                    ensemble_strategy_config = replace(
                        context["strategy_config"],
                        require_consensus_confirmation=True,
                        consensus_gate_mode="entry_only",
                    )
                    ensemble_result = apply_consensus_confirmation(
                        result,
                        seed_consensus,
                        interval=interval,
                        strategy_config=ensemble_strategy_config,
                    )
                    ensemble_nested_summary = nested_holdout_evaluation(
                        predictions=ensemble_result.predictions,
                        n_states=int(row["n_states"]),
                        base_config=ensemble_strategy_config,
                        interval=interval,
                        outer_holdout_folds=1,
                    )
                    ensemble_metrics = {
                        "ensemble_sharpe": float(ensemble_result.metrics.get("sharpe", float("nan"))),
                        "ensemble_outer_holdout_sharpe": float(ensemble_nested_summary.get("outer_holdout_sharpe", float("nan"))),
                        "ensemble_trades": float(ensemble_result.metrics.get("trades", float("nan"))),
                        "ensemble_evaluated": True,
                    }
                except Exception:
                    seed_metrics = _empty_seed_metrics()
                    ensemble_metrics = _empty_ensemble_metrics()
            nested_summary = nested_holdout_evaluation(
                predictions=result.predictions,
                n_states=int(row["n_states"]),
                base_config=context["strategy_config"],
                interval=interval,
                outer_holdout_folds=1,
            )
            candidate_score, candidate_status, notes, _ = _score_candidate(
                result,
                robustness,
                nested_summary,
                seed_metrics,
                ensemble_metrics,
            )
            promotion_gates = build_promotion_gate_rows(
                metrics=result.metrics,
                bootstrap=result.bootstrap,
                state_stability=result.state_stability,
                robustness=robustness,
                baseline_comparison=result.baseline_comparison,
                interval=interval,
                available_rows=int(row["usable_rows"]),
                walk_adjusted=bool(context["was_adjusted"]),
                fold_count=int(len(result.fold_diagnostics)),
                nested_holdout=nested_summary,
            )
            promotion_summary = summarize_promotion_gates(promotion_gates)
            recommendation = recommend_strategy_engine(
                strategy_metrics=result.metrics,
                baseline_comparison=result.baseline_comparison,
                promotion_summary=promotion_summary,
            )
            ok_robustness = robustness.loc[robustness["status"] == "ok"] if "status" in robustness.columns else pd.DataFrame()
            robustness_median = float(ok_robustness["sharpe"].median()) if not ok_robustness.empty else float("nan")
            leaderboard.loc[row_index, "outer_holdout_sharpe"] = float(nested_summary.get("outer_holdout_sharpe", 0.0))
            for key, value in ensemble_metrics.items():
                leaderboard.loc[row_index, key] = value
            for key, value in seed_metrics.items():
                leaderboard.loc[row_index, key] = value
            leaderboard.loc[row_index, "robustness_median_sharpe"] = robustness_median
            leaderboard.loc[row_index, "robustness_evaluated"] = robustness_was_evaluated
            if robustness_was_evaluated:
                leaderboard.loc[row_index, "promotion_verdict"] = promotion_summary["verdict"]
                leaderboard.loc[row_index, "engine_recommendation"] = recommendation["headline"]
                leaderboard.loc[row_index, "recommendation_detail"] = recommendation["summary"]
            leaderboard.loc[row_index, "candidate_score"] = candidate_score
            leaderboard.loc[row_index, "candidate_status"] = candidate_status
            note_parts = [notes] if notes else []
            if not robustness_was_evaluated:
                note_parts.append("robustness_not_evaluated")
            if not bool(seed_metrics.get("seed_evaluated", False)):
                note_parts.append("seed_robustness_not_evaluated")
            leaderboard.loc[row_index, "notes"] = ",".join(part for part in note_parts if part)

        for _ in range(len(leaderboard)):
            for row_index in range(top_k):
                needs_robustness = row_index < int(robustness_top_k) and not bool(leaderboard.iloc[row_index].get("robustness_evaluated", False))
                needs_seed = row_index < int(seed_robustness_top_k) and not bool(leaderboard.iloc[row_index].get("seed_evaluated", False))
                if needs_robustness or needs_seed:
                    _evaluate_leaderboard_row(row_index)

            leaderboard = leaderboard.sort_values(
                ["candidate_score", "sharpe", "outer_holdout_sharpe"],
                ascending=[False, False, False],
                kind="stable",
            ).reset_index(drop=True)
            top_rows_ready = True
            for row_index in range(top_k):
                needs_robustness = row_index < int(robustness_top_k) and not bool(leaderboard.iloc[row_index].get("robustness_evaluated", False))
                needs_seed = row_index < int(seed_robustness_top_k) and not bool(leaderboard.iloc[row_index].get("seed_evaluated", False))
                if needs_robustness or needs_seed:
                    top_rows_ready = False
                    break
            if top_rows_ready:
                break
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    return leaderboard.loc[:, list(CANDIDATE_SEARCH_COLUMNS)]


def summarize_candidate_search(leaderboard: pd.DataFrame) -> dict[str, object]:
    if leaderboard.empty:
        return {
            "status": "unavailable",
            "headline": "Candidate search was not run.",
            "summary": "No ranked candidate search results are available for this session.",
        }

    best = leaderboard.iloc[0]
    seed_sharpe = float(best.get("seed_median_sharpe", float("nan")))
    seed_share = float(best.get("seed_latest_candidate_share", float("nan")))
    ensemble_sharpe = float(best.get("ensemble_sharpe", float("nan")))
    seed_sharpe_text = f"{seed_sharpe:.2f}" if pd.notna(seed_sharpe) else "n/a"
    seed_share_text = f"{seed_share:.0%}" if pd.notna(seed_share) else "n/a"
    ensemble_sharpe_text = f"{ensemble_sharpe:.2f}" if pd.notna(ensemble_sharpe) else "n/a"
    return {
        "status": str(best.get("candidate_status", "unknown")),
        "headline": str(best.get("engine_recommendation", "Candidate search complete")),
        "summary": (
            f"Top ranked variant is {best['feature_pack']} with {int(best['n_states'])} states, "
            f"{best['shorting_mode']}, and `{best['confirmation_mode']}` confirmation. "
            f"Sharpe {float(best['sharpe']):.2f}, outer holdout {float(best['outer_holdout_sharpe']):.2f}, "
            f"ensemble Sharpe {ensemble_sharpe_text}, "
            f"robustness median {float(best['robustness_median_sharpe']):.2f}, "
            f"seed median Sharpe {seed_sharpe_text}, "
            f"latest seed candidate share {seed_share_text}. "
            f"{best['recommendation_detail']}"
        ),
        "best_row": best.to_dict(),
    }


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


def _centered_sweep_config(base_config: StrategyConfig) -> SweepConfig:
    return SweepConfig(
        posterior_thresholds=tuple(
            sorted(
                {
                    max(0.5, round(base_config.posterior_threshold - 0.05, 2)),
                    round(base_config.posterior_threshold, 2),
                    min(0.9, round(base_config.posterior_threshold + 0.05, 2)),
                }
            )
        ),
        min_hold_bars=tuple(sorted({max(1, base_config.min_hold_bars // 2), base_config.min_hold_bars, base_config.min_hold_bars + 4})),
        cooldown_bars=tuple(sorted({max(0, base_config.cooldown_bars - 2), base_config.cooldown_bars, base_config.cooldown_bars + 2})),
        required_confirmations=tuple(
            sorted({max(1, base_config.required_confirmations - 1), base_config.required_confirmations, base_config.required_confirmations + 1})
        ),
    )


def _parameter_row_score(row: pd.Series) -> float:
    score = float(row.get("sharpe", 0.0))
    score += 0.2 * float(row.get("confidence_coverage", 0.0))
    score += 0.1 * min(float(row.get("trades", 0.0)), 10.0) / 10.0
    score -= 2.0 * abs(min(float(row.get("max_drawdown", 0.0)), 0.0))
    if float(row.get("trades", 0.0)) < 2.0:
        score -= 0.5
    return score


def nested_holdout_evaluation(
    *,
    predictions: pd.DataFrame,
    n_states: int,
    base_config: StrategyConfig,
    interval: Interval,
    outer_holdout_folds: int = 1,
) -> dict[str, object]:
    fold_ids = sorted(int(fold_id) for fold_id in predictions["fold_id"].dropna().unique())
    if len(fold_ids) < 3:
        return {
            "status": "insufficient_folds",
            "outer_holdout_folds": 0.0,
            "outer_holdout_sharpe": 0.0,
            "outer_holdout_annualized_return": 0.0,
            "outer_holdout_trades": 0.0,
            "selected_inner_posterior_threshold": float(base_config.posterior_threshold),
            "selected_inner_min_hold_bars": float(base_config.min_hold_bars),
            "selected_inner_cooldown_bars": float(base_config.cooldown_bars),
            "selected_inner_required_confirmations": float(base_config.required_confirmations),
        }

    outer_count = max(1, min(outer_holdout_folds, len(fold_ids) - 1))
    inner_fold_ids = fold_ids[:-outer_count]
    outer_fold_ids = fold_ids[-outer_count:]
    inner_predictions = predictions.loc[predictions["fold_id"].isin(inner_fold_ids)].reset_index(drop=True)
    outer_predictions = predictions.loc[predictions["fold_id"].isin(outer_fold_ids)].reset_index(drop=True)

    inner_sweep = parameter_sweep(
        predictions=inner_predictions,
        n_states=n_states,
        base_config=base_config,
        sweep_config=_centered_sweep_config(base_config),
        interval=interval,
    )
    ranked = inner_sweep.copy()
    ranked["selection_score"] = ranked.apply(_parameter_row_score, axis=1)
    best_row = ranked.sort_values(["selection_score", "sharpe"], ascending=[False, False]).iloc[0]
    selected_config = replace(
        base_config,
        posterior_threshold=float(best_row["posterior_threshold"]),
        min_hold_bars=int(best_row["min_hold_bars"]),
        cooldown_bars=int(best_row["cooldown_bars"]),
        required_confirmations=int(best_row["required_confirmations"]),
    )
    _, outer_metrics = replay_strategy(outer_predictions, n_states, selected_config, interval)
    return {
        "status": "ok",
        "outer_holdout_folds": float(outer_count),
        "outer_holdout_sharpe": float(outer_metrics["sharpe"]),
        "outer_holdout_annualized_return": float(outer_metrics["annualized_return"]),
        "outer_holdout_trades": float(outer_metrics["trades"]),
        "selected_inner_posterior_threshold": float(best_row["posterior_threshold"]),
        "selected_inner_min_hold_bars": float(best_row["min_hold_bars"]),
        "selected_inner_cooldown_bars": float(best_row["cooldown_bars"]),
        "selected_inner_required_confirmations": float(best_row["required_confirmations"]),
        "selection_score": float(best_row["selection_score"]),
    }


def nested_holdout_summary_frame(summary: dict[str, object]) -> pd.DataFrame:
    status = str(summary.get("status", "unknown"))
    status_interpretation = {
        "ok": "Inner folds picked the sweep settings, and the untouched outer fold scored those settings afterward.",
        "insufficient_folds": "There were not enough walk-forward folds to reserve a separate outer holdout slice.",
    }.get(status, "Nested holdout status was not recognized.")
    rows = [
        {
            "component": "Nested Holdout Status",
            "value": status.replace("_", " ").title(),
            "interpretation": status_interpretation,
        },
        {
            "component": "Outer Holdout Folds",
            "value": int(float(summary.get("outer_holdout_folds", 0.0))),
            "interpretation": "These are the most recent untouched folds used only to judge the settings chosen on earlier folds.",
        },
        {
            "component": "Outer Holdout Sharpe",
            "value": f"{float(summary.get('outer_holdout_sharpe', 0.0)):.2f}",
            "interpretation": "This is the holdout-aware Sharpe after inner-fold selection, so it is more trustworthy than the best row from the diagnostic sweep.",
        },
        {
            "component": "Outer Holdout Annualized Return",
            "value": f"{float(summary.get('outer_holdout_annualized_return', 0.0)):.1%}",
            "interpretation": "This is the stitched return across the reserved outer folds only, not the in-sample or validation slices.",
        },
        {
            "component": "Outer Holdout Trades",
            "value": int(float(summary.get("outer_holdout_trades", 0.0))),
            "interpretation": "Very low trade counts make even a positive holdout score fragile, so read this together with Sharpe and bootstrap results.",
        },
        {
            "component": "Selected Posterior Threshold",
            "value": f"{float(summary.get('selected_inner_posterior_threshold', 0.0)):.2f}",
            "interpretation": "Chosen on the inner folds only. Higher values demand cleaner state confidence before entering.",
        },
        {
            "component": "Selected Min Hold",
            "value": int(float(summary.get("selected_inner_min_hold_bars", 0.0))),
            "interpretation": "Chosen on the inner folds only. Higher values make positions stickier and reduce churn.",
        },
        {
            "component": "Selected Cooldown",
            "value": int(float(summary.get("selected_inner_cooldown_bars", 0.0))),
            "interpretation": "Chosen on the inner folds only. Higher values force the strategy to wait longer after exiting.",
        },
        {
            "component": "Selected Confirmations",
            "value": int(float(summary.get("selected_inner_required_confirmations", 0.0))),
            "interpretation": "Chosen on the inner folds only. Higher values require more repeated directional agreement before entering.",
        },
    ]
    if "selection_score" in summary:
        rows.append(
            {
                "component": "Inner Selection Score",
                "value": f"{float(summary['selection_score']):.2f}",
                "interpretation": "This ranks candidate parameter rows on the inner folds. It is useful for selection, but the outer holdout decides whether that choice generalizes.",
            }
        )
    return pd.DataFrame(rows)


def run_autoresearch(
    *,
    program: ResearchProgram,
    results_path: str | Path = "results.tsv",
) -> pd.DataFrame:
    ensure_results_tsv(results_path)
    cache: dict[tuple[str, Interval, int, HistoricalProvider, tuple[str, ...]], tuple[DataFetchResult, pd.DataFrame]] = {}
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
                history_provider="auto",
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
                        history_provider="auto",
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
            nested_summary = nested_holdout_evaluation(
                predictions=result.predictions,
                n_states=n_states,
                base_config=strategy_config,
                interval=interval,
                outer_holdout_folds=program.outer_holdout_folds,
            )
            score, status, notes, consistency = _score_candidate(result, robustness, nested_summary)
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
                    "outer_holdout_folds": nested_summary["outer_holdout_folds"],
                    "outer_holdout_sharpe": nested_summary["outer_holdout_sharpe"],
                    "outer_holdout_annualized_return": nested_summary["outer_holdout_annualized_return"],
                    "outer_holdout_trades": nested_summary["outer_holdout_trades"],
                    "selected_inner_posterior_threshold": nested_summary["selected_inner_posterior_threshold"],
                    "selected_inner_min_hold_bars": nested_summary["selected_inner_min_hold_bars"],
                    "selected_inner_cooldown_bars": nested_summary["selected_inner_cooldown_bars"],
                    "selected_inner_required_confirmations": nested_summary["selected_inner_required_confirmations"],
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
                "nested_summary": nested_summary,
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
                    "outer_holdout_folds": 0.0,
                    "outer_holdout_sharpe": 0.0,
                    "outer_holdout_annualized_return": 0.0,
                    "outer_holdout_trades": 0.0,
                    "selected_inner_posterior_threshold": 0.0,
                    "selected_inner_min_hold_bars": 0.0,
                    "selected_inner_cooldown_bars": 0.0,
                    "selected_inner_required_confirmations": 0.0,
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
        notes.append(f"outer_holdout_sharpe={float(row.outer_holdout_sharpe):.4f}")
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
            nested_holdout_summary=pd.DataFrame([context["nested_summary"]]),
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
