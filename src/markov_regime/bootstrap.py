from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from markov_regime.config import AssetClass, Interval, bars_per_year


def _return_metrics(returns: np.ndarray, annualization: int) -> dict[str, float]:
    cumulative = np.cumprod(1.0 + returns)
    total_return = float(cumulative[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (annualization / max(len(returns), 1)) - 1.0) if total_return > -1.0 else -1.0
    annualized_volatility = float(np.std(returns, ddof=0) * np.sqrt(annualization))
    sharpe = float(np.mean(returns) / np.std(returns, ddof=0) * np.sqrt(annualization)) if np.std(returns, ddof=0) > 0 else 0.0
    drawdown = cumulative / np.maximum.accumulate(cumulative) - 1.0
    max_drawdown = float(drawdown.min())
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _moving_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    starts = rng.integers(0, max(length - block_length + 1, 1), size=int(np.ceil(length / block_length)))
    indices = np.concatenate([np.arange(start, min(start + block_length, length)) for start in starts])
    return indices[:length]


def block_bootstrap_confidence_intervals(
    returns: Iterable[float],
    interval: Interval,
    asset_class: AssetClass = "crypto",
    block_length: int = 24,
    samples: int = 300,
    alpha: float = 0.1,
    seed: int = 7,
) -> pd.DataFrame:
    sampled_returns = np.asarray(list(returns), dtype=float)
    if sampled_returns.size == 0:
        raise ValueError("Cannot bootstrap an empty return series.")

    annualization = bars_per_year(interval, asset_class)
    rng = np.random.default_rng(seed)
    point_estimate = _return_metrics(sampled_returns, annualization)
    bootstrap_rows: list[dict[str, float]] = []

    for _ in range(samples):
        indices = _moving_block_indices(len(sampled_returns), block_length, rng)
        bootstrap_rows.append(_return_metrics(sampled_returns[indices], annualization))

    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    rows: list[dict[str, float | str]] = []
    for metric_name, point_value in point_estimate.items():
        lower = float(bootstrap_frame[metric_name].quantile(alpha / 2.0))
        upper = float(bootstrap_frame[metric_name].quantile(1.0 - alpha / 2.0))
        rows.append(
            {
                "metric": metric_name,
                "point_estimate": point_value,
                "lower": lower,
                "upper": upper,
                "method": "moving_block_bootstrap",
                "samples": float(samples),
            }
        )
    return pd.DataFrame(rows)
