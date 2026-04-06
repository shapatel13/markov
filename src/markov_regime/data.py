from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from dotenv import load_dotenv

from markov_regime.config import AssetClass, DataConfig, HistoricalProvider, infer_asset_class


@dataclass(frozen=True)
class DataFetchResult:
    frame: pd.DataFrame
    source_url: str
    requested_symbol: str
    resolved_symbol: str
    provider: HistoricalProvider
    provider_note: str | None = None


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

AUTO_INTRADAY_TARGET_ROWS: dict[str, int] = {
    "1hour": 2500,
    "4hour": 1800,
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


def _is_crypto_symbol(symbol: str) -> bool:
    return symbol.endswith("USD") and symbol[:-3].isalpha() and len(symbol[:-3]) >= 2


def _to_yahoo_symbol(symbol: str) -> str:
    if _is_crypto_symbol(symbol):
        return f"{symbol[:-3]}-USD"
    return symbol


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


def _yahoo_chart_url(symbol: str) -> str:
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _coinbase_candles_url(symbol: str) -> str:
    return f"https://api.exchange.coinbase.com/products/{symbol}/candles"


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


def _normalize_yahoo_frame(payload: Any, interval: str) -> pd.DataFrame:
    chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
    if chart.get("error"):
        raise ValueError(f"Yahoo Finance chart error: {chart['error']}")

    result = chart.get("result") if isinstance(chart, dict) else None
    if not result:
        raise ValueError("No price data returned from Yahoo Finance.")

    first = result[0]
    timestamps = first.get("timestamp") or []
    quotes = ((first.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps:
        raise ValueError("Yahoo Finance payload did not include timestamps.")

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", []),
        }
    )
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    frame["volume"] = frame["volume"].fillna(0.0)

    if interval in {"1hour", "4hour"}:
        frame = frame.loc[
            frame["timestamp"].dt.minute.eq(0) & frame["timestamp"].dt.second.eq(0),
        ].copy()

    normalized = (
        frame.sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )
    if normalized.empty:
        raise ValueError("Yahoo Finance returned only partial or invalid bars.")
    return normalized


def _normalize_coinbase_frame(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        raise ValueError("No price data returned from Coinbase.")

    frame = pd.DataFrame(payload, columns=["epoch", "low", "high", "open", "close", "volume"])
    timestamps = pd.to_datetime(frame["epoch"], unit="s", utc=True).dt.tz_convert(None)
    normalized = (
        frame.assign(timestamp=timestamps)
        .loc[:, ["timestamp", "open", "high", "low", "close", "volume"]]
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )
    if normalized.empty:
        raise ValueError("Coinbase returned an empty candle frame after normalization.")
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


def _attach_market_metadata(frame: pd.DataFrame, resolved_symbol: str) -> pd.DataFrame:
    asset_class: AssetClass = infer_asset_class(resolved_symbol)
    enriched = frame.copy()
    enriched["resolved_symbol"] = resolved_symbol
    enriched["asset_class"] = asset_class
    return enriched


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


def _to_utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _apply_time_filters(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    filtered = frame
    if config.start:
        filtered = filtered.loc[filtered["timestamp"] >= pd.Timestamp(config.start)].copy()
    if config.end:
        filtered = filtered.loc[filtered["timestamp"] <= pd.Timestamp(config.end)].copy()
    filtered = _resample_ohlcv(filtered, config.interval)
    if config.limit > 0:
        filtered = filtered.tail(config.limit).reset_index(drop=True)
    return filtered


def _fetch_fmp_price_data(
    *,
    config: DataConfig,
    key: str,
    client: requests.Session,
    resolved_symbol: str,
) -> DataFetchResult:
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

    frame = _attach_market_metadata(_apply_time_filters(_normalize_frame(payload, config.interval), config), resolved_symbol)
    return DataFetchResult(
        frame=frame,
        source_url=_redact_api_key(response.url) if response else candidate_urls[0],
        requested_symbol=config.symbol,
        resolved_symbol=resolved_symbol,
        provider="fmp",
    )


def _yahoo_range_for_interval(interval: str) -> str:
    if interval in {"1hour", "4hour"}:
        return "730d"
    return "10y"


def _fetch_yahoo_price_data(
    *,
    config: DataConfig,
    client: requests.Session,
    resolved_symbol: str,
) -> DataFetchResult:
    yahoo_symbol = _to_yahoo_symbol(resolved_symbol)
    url = _yahoo_chart_url(yahoo_symbol)
    params: dict[str, str | int] = {
        "interval": "1h" if config.interval in {"1hour", "4hour"} else "1d",
        "includePrePost": "false",
        "events": "div,splits,capitalGains",
    }
    if config.start or config.end:
        if config.start:
            params["period1"] = int(_to_utc_timestamp(config.start).timestamp())
        if config.end:
            params["period2"] = int(_to_utc_timestamp(config.end).timestamp())
    else:
        params["range"] = _yahoo_range_for_interval(config.interval)

    response = client.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    frame = _attach_market_metadata(_apply_time_filters(_normalize_yahoo_frame(response.json(), config.interval), config), resolved_symbol)
    return DataFetchResult(
        frame=frame,
        source_url=response.url,
        requested_symbol=config.symbol,
        resolved_symbol=resolved_symbol,
        provider="yahoo",
    )


def _target_auto_rows(config: DataConfig) -> int:
    baseline = AUTO_INTRADAY_TARGET_ROWS.get(config.interval, 0)
    if config.limit > 0:
        return min(config.limit, baseline) if baseline else config.limit
    return baseline


def _should_try_long_history_fallback(frame: pd.DataFrame, config: DataConfig, resolved_symbol: str) -> bool:
    if config.interval not in {"1hour", "4hour"} or not _is_crypto_symbol(resolved_symbol):
        return False
    if frame.empty:
        return True
    if config.start and frame["timestamp"].min() > pd.Timestamp(config.start):
        return True
    target_rows = _target_auto_rows(config)
    return target_rows > 0 and len(frame) < target_rows


def _to_coinbase_symbol(symbol: str) -> str:
    if _is_crypto_symbol(symbol):
        return f"{symbol[:-3]}-USD"
    raise ValueError(f"Coinbase long-history backfill only supports crypto USD pairs, received `{symbol}`.")


def _coinbase_source_interval(interval: str) -> str:
    return "1day" if interval == "1day" else "1hour"


def _coinbase_granularity(interval: str) -> int:
    return 86_400 if interval == "1day" else 3_600


def _coinbase_source_limit(config: DataConfig) -> int:
    if config.limit <= 0:
        return 3000 if config.interval == "1day" else 20_000
    if config.interval == "4hour":
        return config.limit * 4
    return config.limit


def _fetch_coinbase_price_data(
    *,
    config: DataConfig,
    client: requests.Session,
    resolved_symbol: str,
) -> DataFetchResult:
    product_symbol = _to_coinbase_symbol(resolved_symbol)
    source_interval = _coinbase_source_interval(config.interval)
    granularity = _coinbase_granularity(source_interval)
    source_limit = _coinbase_source_limit(config)
    chunk_size = 300
    seconds_per_bar = granularity

    if config.end:
        end_time = _to_utc_timestamp(config.end)
    else:
        end_time = pd.Timestamp.now(tz="UTC").floor("h" if source_interval == "1hour" else "D")

    if config.start:
        start_time = _to_utc_timestamp(config.start)
    else:
        start_time = end_time - pd.Timedelta(seconds=max(source_limit - 1, 1) * seconds_per_bar)

    request_cursor = start_time
    responses: list[pd.DataFrame] = []
    last_url = _coinbase_candles_url(product_symbol)

    while request_cursor < end_time:
        request_end = min(request_cursor + pd.Timedelta(seconds=seconds_per_bar * chunk_size), end_time)
        params = {
            "granularity": str(granularity),
            "start": request_cursor.isoformat().replace("+00:00", "Z"),
            "end": request_end.isoformat().replace("+00:00", "Z"),
        }
        response = client.get(
            _coinbase_candles_url(product_symbol),
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            request_cursor = request_end
            continue
        normalized = _normalize_coinbase_frame(payload)
        responses.append(normalized)
        last_url = response.url
        request_cursor = request_end

    if not responses:
        raise ValueError("Coinbase returned no candles for the requested range.")
    combined = (
        pd.concat(responses, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )
    frame = _attach_market_metadata(_apply_time_filters(combined, config), resolved_symbol)
    return DataFetchResult(
        frame=frame,
        source_url=last_url,
        requested_symbol=config.symbol,
        resolved_symbol=resolved_symbol,
        provider="coinbase",
    )


def fetch_price_data(
    config: DataConfig,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> DataFetchResult:
    client = session or requests.Session()
    resolved_symbol = normalize_symbol(config.symbol)
    fmp_key: str | None = None
    fmp_key_error: Exception | None = None
    if config.provider in {"auto", "fmp"}:
        try:
            fmp_key = load_api_key(api_key)
        except Exception as exc:  # pragma: no cover - defensive fallback
            fmp_key_error = exc

    if config.provider == "fmp":
        if fmp_key is None:
            raise ValueError(str(fmp_key_error or "FMP API key is unavailable."))
        return _fetch_fmp_price_data(config=config, key=fmp_key, client=client, resolved_symbol=resolved_symbol)

    if config.provider == "yahoo":
        return _fetch_yahoo_price_data(config=config, client=client, resolved_symbol=resolved_symbol)

    if config.provider == "coinbase":
        return _fetch_coinbase_price_data(config=config, client=client, resolved_symbol=resolved_symbol)

    if fmp_key is None:
        if _is_crypto_symbol(resolved_symbol):
            try:
                return _fetch_coinbase_price_data(config=config, client=client, resolved_symbol=resolved_symbol)
            except Exception:
                return _fetch_yahoo_price_data(config=config, client=client, resolved_symbol=resolved_symbol)
        return _fetch_yahoo_price_data(config=config, client=client, resolved_symbol=resolved_symbol)

    fmp_error: Exception | None = None
    try:
        fmp_result = _fetch_fmp_price_data(config=config, key=fmp_key, client=client, resolved_symbol=resolved_symbol)
    except Exception as exc:
        fmp_error = exc
        fmp_result = None

    if fmp_result is None:
        if _is_crypto_symbol(resolved_symbol):
            try:
                fallback = _fetch_coinbase_price_data(config=config, client=client, resolved_symbol=resolved_symbol)
                return DataFetchResult(
                    frame=fallback.frame,
                    source_url=fallback.source_url,
                    requested_symbol=config.symbol,
                    resolved_symbol=resolved_symbol,
                    provider=fallback.provider,
                    provider_note="Auto-kept Financial Modeling Prep as the primary source but fell back because the FMP fetch failed.",
                )
            except Exception:
                yahoo_result = _fetch_yahoo_price_data(config=config, client=client, resolved_symbol=resolved_symbol)
                return DataFetchResult(
                    frame=yahoo_result.frame,
                    source_url=yahoo_result.source_url,
                    requested_symbol=config.symbol,
                    resolved_symbol=resolved_symbol,
                    provider=yahoo_result.provider,
                    provider_note="Auto-kept Financial Modeling Prep as the primary source but used Yahoo fallback because the FMP and Coinbase fetches failed.",
                )
        yahoo_result = _fetch_yahoo_price_data(config=config, client=client, resolved_symbol=resolved_symbol)
        return DataFetchResult(
            frame=yahoo_result.frame,
            source_url=yahoo_result.source_url,
            requested_symbol=config.symbol,
            resolved_symbol=resolved_symbol,
            provider=yahoo_result.provider,
            provider_note=f"Auto-kept Financial Modeling Prep as the primary source but used Yahoo fallback because the FMP fetch failed: {fmp_error}",
        )

    if not _should_try_long_history_fallback(fmp_result.frame, config, resolved_symbol):
        return fmp_result

    coinbase_result: DataFetchResult | None = None
    yahoo_result: DataFetchResult | None = None
    try:
        coinbase_result = _fetch_coinbase_price_data(config=config, client=client, resolved_symbol=resolved_symbol)
    except Exception:  # pragma: no cover - exercised only when a live fallback is unavailable
        coinbase_result = None

    if coinbase_result is not None and len(coinbase_result.frame) > len(fmp_result.frame):
        return DataFetchResult(
            frame=coinbase_result.frame,
            source_url=coinbase_result.source_url,
            requested_symbol=config.symbol,
            resolved_symbol=resolved_symbol,
            provider=coinbase_result.provider,
            provider_note=(
                f"Auto-kept Financial Modeling Prep as the primary source and switched to Coinbase deep-history backfill because "
                f"FMP returned {len(fmp_result.frame)} usable `{config.interval}` rows, which is too thin for the requested research depth."
            ),
        )

    try:
        yahoo_result = _fetch_yahoo_price_data(config=config, client=client, resolved_symbol=resolved_symbol)
    except Exception:  # pragma: no cover - exercised only when a live fallback is unavailable
        yahoo_result = None

    if yahoo_result is not None and len(yahoo_result.frame) > len(fmp_result.frame):
        return DataFetchResult(
            frame=yahoo_result.frame,
            source_url=yahoo_result.source_url,
            requested_symbol=config.symbol,
            resolved_symbol=resolved_symbol,
            provider=yahoo_result.provider,
            provider_note=(
                f"Auto-kept Financial Modeling Prep as the primary source and used Yahoo deep-history fallback because "
                f"FMP returned {len(fmp_result.frame)} usable `{config.interval}` rows, which is too thin for the requested research depth, "
                "and Coinbase backfill was unavailable or insufficient."
            ),
        )

    return fmp_result


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
