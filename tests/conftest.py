from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from markov_regime.features import build_feature_frame


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    periods = 720
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    regimes = np.repeat([0, 1, 2, 1, 0, 2], periods // 6)
    regime_drifts = np.select(
        [regimes == 0, regimes == 1, regimes == 2],
        [0.0012, -0.0009, 0.0003],
        default=0.0,
    )
    noise = rng.normal(0.0, 0.002, size=periods)
    returns = regime_drifts + noise
    close = 100.0 * np.cumprod(1.0 + returns)
    open_prices = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_prices, close) * (1.0 + rng.uniform(0.0005, 0.003, size=periods))
    low = np.minimum(open_prices, close) * (1.0 - rng.uniform(0.0005, 0.003, size=periods))
    volume = rng.integers(800_000, 1_500_000, size=periods)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture()
def synthetic_feature_frame(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(synthetic_prices)

