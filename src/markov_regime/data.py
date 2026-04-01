from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from dotenv import load_dotenv

from markov_regime.config import DataConfig, DataSource


@dataclass(frozen=True)
class DataFetchResult:
    frame: pd.DataFrame
    source_url: str
    requested_symbol: str
    resolved_symbol: str
    source_provider: DataSource


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


def provider_symbol(symbol: str, source: DataSource) -> str:
    canonical = normalize_symbol(symbol)
    if source == "fmp":
        return canonical
    if canonical.endswith("USD") and canonical[:-3].isalpha():
        return f"{canonical[:-3]}-USD"
    return canonical


def _hourly_url(symbol: str) -> str:
    return "https://financialmodelingprep.com/stable/historical-chart/1hour"


def _daily_url(symbol: str) -> str:
    return "https://financialmodelingprep.com/stable/historical-price-eod/full"


def _legacy_hourly_url(symbol: str) -> str:
    return f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{symbol}"


def _legacy_daily_url(symbol: str) -> str:
    return f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"


def _normalize_fmp_frame(payload: Any, interval: str) -> pd.DataFrame:
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


def _normalize_yahoo_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("No price data returned from Yahoo Finance.")

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [str(levels[0]) for levels in normalized.columns]
    normalized = normalized.reset_index()
    timestamp_column = "Datetime" if "Datetime" in normalized.columns else "Date"
    if timestamp_column not in normalized.columns:
        raise ValueError("Yahoo Finance payload is missing a timestamp column.")

    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Yahoo Finance payload is missing required columns: {missing}")

    if "Volume" not in normalized.columns:
        normalized["Volume"] = 0.0

    timestamps = pd.to_datetime(normalized[timestamp_column], utc=False)
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert(None)

    return (
        normalized.loc[:, [timestamp_column, "Open", "High", "Low", "Close", "Volume"]]
        .rename(
            columns={
                timestamp_column: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .assign(timestamp=timestamps)
        .sort_values("timestamp")
        .dropna(subset=["open", "high", "low", "close"])
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )


def _redact_api_key(url: str) -> str:
    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True)
    redacted_query = [(key, "***" if key.lower() == "apikey" else value) for key, value in query_items]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted_query), parts.fragment))


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


def _yahoo_period(config: DataConfig) -> str:
    if config.interval == "1day":
        return "max"
    requested_hourly_bars = config.limit if config.interval == "1hour" else config.limit * 4
    warmup_bars = max(24 * 120, requested_hourly_bars // 5)
    requested_days = int(math.ceil((requested_hourly_bars + warmup_bars) / 24))
    return f"{min(max(requested_days, 180), 730)}d"


def _fetch_from_yahoo(config: DataConfig) -> DataFetchResult:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ValueError("Yahoo Finance support requires `yfinance`. Install dependencies with `python -m pip install -e .[dev]`.") from exc

    resolved_symbol = normalize_symbol(config.symbol)
    provider_ticker = provider_symbol(resolved_symbol, "yahoo")
    fetch_interval = "1d" if config.interval == "1day" else "1h"
    download_kwargs: dict[str, Any] = {
        "tickers": provider_ticker,
        "interval": fetch_interval,
        "auto_adjust": False,
        "progress": False,
        "actions": False,
        "threads": False,
    }
    if config.start or config.end:
        if config.start:
            download_kwargs["start"] = config.start
        if config.end:
            download_kwargs["end"] = config.end
        range_hint = f"start={config.start or 'min'}&end={config.end or 'now'}"
    else:
        period = _yahoo_period(config)
        download_kwargs["period"] = period
        range_hint = f"period={period}"

    payload = yf.download(**download_kwargs)
    frame = _normalize_yahoo_frame(payload)
    frame = _resample_ohlcv(frame, config.interval)
    if config.limit > 0:
        frame = frame.tail(config.limit).reset_index(drop=True)

    return DataFetchResult(
        frame=frame,
        source_url=f"yfinance://{provider_ticker}?interval={fetch_interval}&{range_hint}",
        requested_symbol=config.symbol,
        resolved_symbol=resolved_symbol,
        source_provider="yahoo",
    )


def _fetch_from_fmp(
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
    params: dict[str, str] = {"apikey": key, "symbol": provider_symbol(resolved_symbol, "fmp")}
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

    frame = _normalize_fmp_frame(payload, config.interval)
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
        source_provider="fmp",
    )


def fetch_price_data(
    config: DataConfig,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> DataFetchResult:
    if config.source == "fmp":
        return _fetch_from_fmp(config, api_key=api_key, session=session)
    return _fetch_from_yahoo(config)
