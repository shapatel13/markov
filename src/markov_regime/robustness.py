from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import pandas as pd

from markov_regime.confirmation import apply_higher_timeframe_confirmation
from markov_regime.config import DataConfig, Interval, ModelConfig, StrategyConfig, WalkForwardConfig, default_walk_forward_config
from markov_regime.data import fetch_price_data
from markov_regime.features import build_feature_frame
from markov_regime.walkforward import run_walk_forward, suggest_walk_forward_config


def parse_symbol_list(symbols: str | Iterable[str]) -> list[str]:
    if isinstance(symbols, str):
        items = [item.strip().upper() for item in symbols.split(",")]
    else:
        items = [str(item).strip().upper() for item in symbols]
    return [item for item in items if item]


def run_multi_asset_robustness(
    *,
    symbols: list[str],
    interval: Interval,
    source: str = "yahoo",
    limit: int,
    feature_columns: tuple[str, ...],
    model_config: ModelConfig,
    walk_config: WalkForwardConfig,
    strategy_config: StrategyConfig,
    auto_adjust_windows: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for symbol in symbols:
        try:
            fetched = fetch_price_data(DataConfig(symbol=symbol, interval=interval, source=source, limit=limit))
            feature_frame = build_feature_frame(fetched.frame, feature_columns=feature_columns)
            effective_walk_config, was_adjusted = (
                suggest_walk_forward_config(len(feature_frame), walk_config)
                if auto_adjust_windows
                else (walk_config, False)
            )
            result = run_walk_forward(
                feature_frame=feature_frame,
                feature_columns=feature_columns,
                interval=interval,
                model_config=model_config,
                walk_config=effective_walk_config,
                strategy_config=strategy_config,
            )
            if interval == "4hour" and strategy_config.require_daily_confirmation:
                confirmation_fetched = fetch_price_data(DataConfig(symbol=symbol, interval="1day", source=source, limit=limit))
                confirmation_features = build_feature_frame(confirmation_fetched.frame, feature_columns=feature_columns)
                requested_confirmation_walk = default_walk_forward_config("1day")
                confirmation_walk_config, _ = (
                    suggest_walk_forward_config(len(confirmation_features), requested_confirmation_walk)
                    if auto_adjust_windows
                    else (requested_confirmation_walk, False)
                )
                confirmation_result = run_walk_forward(
                    feature_frame=confirmation_features,
                    feature_columns=feature_columns,
                    interval="1day",
                    model_config=model_config,
                    walk_config=confirmation_walk_config,
                    strategy_config=replace(strategy_config, require_daily_confirmation=False),
                )
                result = apply_higher_timeframe_confirmation(
                    result,
                    confirmation_result,
                    interval=interval,
                    strategy_config=strategy_config,
                    confirmation_interval="1day",
                )
            rows.append(
                {
                    "symbol": symbol,
                    "resolved_symbol": fetched.resolved_symbol,
                    "status": "ok",
                    "raw_rows": len(fetched.frame),
                    "usable_rows": len(feature_frame),
                    "walk_adjusted": was_adjusted,
                    "train_bars": effective_walk_config.train_bars,
                    "validate_bars": effective_walk_config.validate_bars,
                    "test_bars": effective_walk_config.test_bars,
                    "purge_bars": effective_walk_config.purge_bars,
                    "embargo_bars": effective_walk_config.embargo_bars,
                    "sharpe": result.metrics["sharpe"],
                    "annualized_return": result.metrics["annualized_return"],
                    "max_drawdown": result.metrics["max_drawdown"],
                    "exposure": result.metrics["exposure"],
                    "benchmark_sharpe": result.benchmark_metrics["sharpe"],
                    "benchmark_annualized_return": result.benchmark_metrics["annualized_return"],
                    "stability_score": float(result.state_stability["stability_score"].median()) if not result.state_stability.empty else 0.0,
                    "latest_close": float(fetched.frame["close"].iloc[-1]),
                    "start_time": str(pd.to_datetime(fetched.frame["timestamp"].iloc[0])),
                    "end_time": str(pd.to_datetime(fetched.frame["timestamp"].iloc[-1])),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "resolved_symbol": symbol,
                    "status": "error",
                    "error": str(exc),
                }
            )

    result_frame = pd.DataFrame(rows)
    if "status" in result_frame.columns:
        result_frame = result_frame.sort_values(["status", "symbol"]).reset_index(drop=True)
    return result_frame
