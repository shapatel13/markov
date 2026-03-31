from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from markov_regime.config import DataConfig


@dataclass(frozen=True)
class DataFetchResult:
    frame: pd.DataFrame
    source_url: str
    requested_symbol: str
    resolved_symbol: str


CRYPTO_ALIASES: dict[str, str] = {
    "BTC": "BTCUSD",
    "BTC-USD": "BTCUSD",
    "ETH": "ETHUSD",
    "ETH-USD": "ETHUSD",
    "SOL": "SOLUSD",
    "SOL-USD": "SOLUSD",
    "DOGE": "DOGEUSD",
    "DOGE-USD": "DOGEUSD",
    "ADA": "ADAUSD",
    "ADA-USD": "ADAUSD",
    "XRP": "XRPUSD",
    "XRP-USD": "XRPUSD",
    "BNB": "BNBUSD",
    "BNB-USD": "BNBUSD",
}


def load_api_key(explicit_key: str | None = None) -> str:
    load_dotenv()
    api_key = explicit_key or os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY is not set. Add it to .env or pass an explicit key.")
    return api_key


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    return CRYPTO_ALIASES.get(cleaned, cleaned)


def _hourly_url(symbol: str) -> str:
    return "https://financialmodelingprep.com/stable/historical-chart/1hour"


def _daily_url(symbol: str) -> str:
    return "https://financialmodelingprep.com/stable/historical-price-eod/full"


def _legacy_hourly_url(symbol: str) -> str:
    return f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol}"


def _legacy_daily_url(symbol: str) -> str:
    return f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"


def _normalize_frame(payload: Any, interval: str) -> pd.DataFrame:
    if interval == "1day" and isinstance(payload, dict):
        records = payload.get("historical", [])
    else:
        records = payload

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No price data returned from Financial Modeling Prep.")

    if "date" not in frame.columns:
        raise ValueError("Price payload is missing a date field.")

    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Price payload is missing required columns: {missing}")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    normalized = (
        frame.loc[:, ["date", "open", "high", "low", "close", "volume"]]
        .rename(columns={"date": "timestamp"})
        .assign(timestamp=lambda item: pd.to_datetime(item["timestamp"], utc=False))
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )
    return normalized


def fetch_price_data(
    config: DataConfig,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> DataFetchResult:
    key = load_api_key(api_key)
    client = session or requests.Session()
    resolved_symbol = normalize_symbol(config.symbol)
    candidate_urls = (
        [_daily_url(resolved_symbol), _legacy_daily_url(resolved_symbol)]
        if config.interval == "1day"
        else [_hourly_url(resolved_symbol), _legacy_hourly_url(resolved_symbol)]
    )
    params: dict[str, str] = {"apikey": key, "symbol": resolved_symbol}
    if config.start:
        params["from"] = config.start
    if config.end:
        params["to"] = config.end

    last_error: Exception | None = None
    response: requests.Response | None = None
    payload: Any = None
    for base_url in candidate_urls:
        try:
            response = client.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("Error Message"):
                raise ValueError(payload["Error Message"])
            break
        except Exception as exc:  # pragma: no cover - exercised via live API fallback
            last_error = exc
            response = None
            payload = None
    else:
        raise ValueError(f"Unable to fetch price data from Financial Modeling Prep: {last_error}") from last_error

    frame = _normalize_frame(payload, config.interval)
    if config.start:
        frame = frame.loc[frame["timestamp"] >= pd.Timestamp(config.start)].copy()
    if config.end:
        frame = frame.loc[frame["timestamp"] <= pd.Timestamp(config.end)].copy()
    if config.limit > 0:
        frame = frame.tail(config.limit).reset_index(drop=True)

    return DataFetchResult(
        frame=frame,
        source_url=response.url if response else candidate_urls[0],
        requested_symbol=config.symbol,
        resolved_symbol=resolved_symbol,
    )
