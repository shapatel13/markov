from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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


@dataclass(frozen=True)
class LiveQuote:
    symbol: str
    price: float
    change: float | None
    change_percentage: float | None
    volume: float | None
    open: float | None
    previous_close: float | None
    day_low: float | None
    day_high: float | None
    market_cap: float | None
    exchange: str | None
    timestamp: pd.Timestamp | None
    source_url: str


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
    if not os.getenv("FMP_API_KEY"):
        example_path = Path(".env.example")
        if example_path.exists():
            load_dotenv(example_path, override=False)
    api_key = explicit_key or os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY is not set. Add it to .env, .env.example, or pass an explicit key.")
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


def _quote_url(symbol: str) -> str:
    return "https://financialmodelingprep.com/stable/quote"


def _quote_short_url(symbol: str) -> str:
    return "https://financialmodelingprep.com/stable/quote-short"


def _legacy_quote_url(symbol: str) -> str:
    return f"https://financialmodelingprep.com/api/v3/quote/{symbol}"


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


def _redact_api_key(url: str) -> str:
    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True)
    redacted_query = [(key, "***" if key.lower() == "apikey" else value) for key, value in query_items]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted_query), parts.fragment))


def _normalize_quote(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("No quote data returned from Financial Modeling Prep.")
        quote = payload[0]
    elif isinstance(payload, dict):
        quote = payload
    else:
        raise ValueError("Quote payload from Financial Modeling Prep is not in a supported format.")

    if "price" not in quote:
        raise ValueError("Quote payload is missing a price field.")
    return quote


def _resample_ohlcv(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    if interval != "4hour":
        return frame

    indexed = frame.sort_values("timestamp").set_index("timestamp")
    aggregated = indexed.resample("4h", label="right", closed="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bar_count=("close", "size"),
    )
    complete = aggregated.loc[aggregated["bar_count"] == 4].drop(columns="bar_count")
    complete = complete.dropna(subset=["open", "high", "low", "close"]).reset_index()
    if complete.empty:
        raise ValueError("Unable to build complete 4-hour candles from the fetched hourly series.")
    return complete


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
    frame = _resample_ohlcv(frame, config.interval)
    if config.limit > 0:
        frame = frame.tail(config.limit).reset_index(drop=True)

    return DataFetchResult(
        frame=frame,
        source_url=_redact_api_key(response.url) if response else candidate_urls[0],
        requested_symbol=config.symbol,
        resolved_symbol=resolved_symbol,
    )


def fetch_live_quote(
    symbol: str,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> LiveQuote:
    key = load_api_key(api_key)
    client = session or requests.Session()
    resolved_symbol = normalize_symbol(symbol)
    candidate_urls = [_quote_url(resolved_symbol), _quote_short_url(resolved_symbol), _legacy_quote_url(resolved_symbol)]
    params: dict[str, str] = {"apikey": key, "symbol": resolved_symbol}

    last_error: Exception | None = None
    response: requests.Response | None = None
    payload: Any = None
    for base_url in candidate_urls:
        try:
            request_params = params if "stable" in base_url else {"apikey": key}
            response = client.get(base_url, params=request_params, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("Error Message"):
                raise ValueError(payload["Error Message"])
            quote = _normalize_quote(payload)
            timestamp = quote.get("timestamp")
            quote_time = pd.to_datetime(int(timestamp), unit="s", utc=True) if timestamp is not None else None
            return LiveQuote(
                symbol=str(quote.get("symbol", resolved_symbol)),
                price=float(quote["price"]),
                change=float(quote["change"]) if quote.get("change") is not None else None,
                change_percentage=float(quote["changePercentage"]) if quote.get("changePercentage") is not None else None,
                volume=float(quote["volume"]) if quote.get("volume") is not None else None,
                open=float(quote["open"]) if quote.get("open") is not None else None,
                previous_close=float(quote["previousClose"]) if quote.get("previousClose") is not None else None,
                day_low=float(quote["dayLow"]) if quote.get("dayLow") is not None else None,
                day_high=float(quote["dayHigh"]) if quote.get("dayHigh") is not None else None,
                market_cap=float(quote["marketCap"]) if quote.get("marketCap") is not None else None,
                exchange=str(quote["exchange"]) if quote.get("exchange") is not None else None,
                timestamp=quote_time,
                source_url=_redact_api_key(response.url) if response else base_url,
            )
        except Exception as exc:  # pragma: no cover - exercised via live API fallback
            last_error = exc
            response = None
            payload = None
    raise ValueError(f"Unable to fetch live quote from Financial Modeling Prep: {last_error}") from last_error
