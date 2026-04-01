# Markov Regime Research

Local, inspectable Hidden Markov Model regime research app with walk-forward retraining, parameter sweeps, guardrails, timeframe comparison, and report export. The app is intentionally built to emphasize methodological caution over opaque optimization.

## What It Does

- Fetches daily or hourly bars from Financial Modeling Prep and locally resamples complete `4hour` candles from hourly data.
- Builds multiple feature packs spanning return, trend, volatility, range, EMA distance, compression, ADX/DI trend strength, RSI/Bollinger mean reversion, Donchian breakout context, rolling VWAP gap, and realized skew/kurtosis structure.
- Runs explicit purged train / validate / embargo / test walk-forward retraining with rolling refits.
- Stitches performance only from the blind test slices and keeps train / validate periods out of headline return metrics.
- Compares 5 through 9 HMM states side by side.
- Compares `4hour`, `1day`, and `1hour` operating modes so higher-timeframe trades can be judged against both the slower swing lane and the noisier intraday baseline.
- Supports an optional `4hour` execution filter that only blocks trades when the daily lane clearly disagrees, while allowing neutral daily bars to pass through without adding conviction.
- Aligns states across retrains and reports stability diagnostics.
- Sweeps posterior threshold, minimum hold, cooldown, and required confirmations.
- Reports block-bootstrap confidence intervals and transaction cost stress tests.
- Adds dynamic crypto execution frictions using fees, spread, slippage, intrabar range, and liquidity impact.
- Logs reproducible local run artifacts with config snapshots, manifests, and exported tables.
- Checks robustness across a multi-asset basket instead of only the primary symbol.
- Exports the signal history as both CSV and JSON.
- Benchmarks the HMM against tougher simple references including ATR trend, ATR breakout-stop, and daily-trend-filter baselines.
- Adds a constrained local `autoresearch` loop with a frozen evaluator, feature-pack candidates, `research_program.md`, local `results.tsv` logging, and artifact export for the best runs.

## Quick Start

1. Create a virtual environment and install the package.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

2. Add your FMP API key to `.env`.

```text
FMP_API_KEY=your_key_here
```

The loader prefers `.env` and will also fall back to `.env.example` for local testing, but `.env` is the safer place for a real key.

3. Run the local app.

```powershell
streamlit run app.py
```

## CLI

Backtest:

```powershell
python -m markov_regime backtest --symbol BTCUSD --interval 4hour --states 6
```

Sweep:

```powershell
python -m markov_regime sweep --symbol BTCUSD --interval 4hour --states 6
```

Export the signal report:

```powershell
python -m markov_regime export-report --symbol BTCUSD --interval 4hour --states 6
```

Compare `4hour`, `1day`, and `1hour`:

```powershell
python -m markov_regime compare-timeframes --symbol BTCUSD --states 6
```

Backtest `4hour` with daily confirmation turned on:

```powershell
python -m markov_regime backtest --symbol BTCUSD --interval 4hour --feature-pack regime_mix --require-daily-confirmation
```

Compare feature packs on the same timeframe:

```powershell
python -m markov_regime compare-feature-packs --symbol BTCUSD --interval 4hour --states 6
```

Initialize the local autoresearch files:

```powershell
python -m markov_regime init-research
```

Run the constrained autoresearch batch:

```powershell
python -m markov_regime autoresearch
```

Use a non-default feature pack in any CLI evaluation:

```powershell
python -m markov_regime backtest --symbol BTCUSD --interval 4hour --feature-pack regime_mix
```

Or launch the app directly:

```powershell
python -m streamlit run app.py
```


## Methodology Notes

- Training, validation, and test windows are fully separated in each fold, with optional purge and embargo bars to reduce leakage around window boundaries.
- The app now includes a dedicated `Methodology` panel showing the walk-forward schedule, current friction assumptions, and promotion gates for the active run.
- The app now defaults to BTC `4hour`, with `1day` alongside it as a slower confirmation lane, because those higher timeframes tend to produce more stable regime structure than `1hour` noise.
- The higher-timeframe defaults now approximate a `12 months train / 3 months validate / 3 months blind test` cadence on `4hour` and `1day`.
- The optional daily confirmation filter acts as an execution veto, not a second alpha model. Daily neutrality is allowed; clear daily opposition blocks the `4hour` trade.
- State labels are re-aligned after every refit because raw HMM state IDs are not stable.
- Directional actions come from validation-window forward returns, not the in-sample training fit.
- The strategy prefers flat exposure when posterior confidence is marginal or when validation support is weak.
- Default direct fee friction is now `10 bps` before spread, slippage, and impact, so backtests do not assume unrealistically cheap crypto execution.
- Major metrics are paired with moving block bootstrap confidence intervals to avoid treating a single backtest path as definitive.
- Every app run writes a local artifact bundle in `artifacts/` with the config snapshot, manifest, model tables, robustness table, and signal report.
- The displayed strategy performance is now compared against buy-and-hold on the same test slices.
- The local `autoresearch` loop never mutates the evaluator; it only ranks candidate interval, feature-pack, state-count, and strategy-control combinations defined by [`research_program.md`](research_program.md).

## Outputs

- Model comparison table for 5-9 states.
- Fold-by-fold diagnostics including AIC, BIC, convergence, and strategy metrics.
- State stability table across retrains.
- Regime-conditioned forward return table for 1, 3, 6, 12, 24, and 72 bars.
- Sensitivity plots for posterior threshold, min hold, cooldown, and confirmation count.
- Methodology and promotion-gate tables for the active run.
- Cross-asset robustness table for the configured basket.
- `4hour` vs `1day` vs `1hour` comparison table for the primary symbol.
- Artifact manifests and CSV snapshots in `artifacts/`.
- Exported signal reports in `exports/`.
- Local autoresearch leaderboard in ignored `results.tsv`.
- Feature-pack-aware autoresearch artifacts for the strongest candidates.

## Autoresearch

- [`research_program.md`](research_program.md) defines the safe search space and the frozen evaluator rules.
- `results.tsv` is local and ignored by Git, so you can run repeated experiment batches without dirtying the repo.
- The scoring loop is intentionally conservative: it ranks candidates using Sharpe, bootstrap lower bounds, state stability, benchmark-relative return, cost breakpoints, and robustness instead of raw win rate alone.
- UI parameter sweeps are diagnostic only. They replay alternate thresholds on the already-observed out-of-sample path and should not be treated as blind model selection.
- The search space now includes feature packs, so autoresearch can test whether the signal improves because the model sees a better market description, not just because entry filters were retuned.
- The autoresearch score now also checks early-vs-late fold confirmation so a candidate that only looks good in one part of the sample is penalized.
- The top-ranked candidates automatically get artifact bundles in `artifacts/` so they can be inspected in the same format as manual app runs.
- This is a constrained optimization harness, not an unrestricted self-modifying agent. That limitation is intentional so the evaluation logic stays trustworthy.

## Known Failure Modes

- HMM state identities are latent and can rotate or split across retrains even after alignment, so regime names remain heuristic.
- Small validation windows can make state-to-action mappings noisy, especially for infrequent states.
- State-count selection matters: if 5-9 states produce materially different results, the regime story is fragile.
- Backtest quality can deteriorate quickly once transaction costs, slippage, or stale fills are introduced.
- Even the richer friction model is still an approximation; it is not a substitute for exchange-specific order book replay or venue-level fill data.
- The `4hour` candles are aggregated locally from hourly API data, so missing hourly bars or late prints can distort the higher-timeframe candle set.
- Annualization is crypto-first and assumes 24/7 bars; if you use equity symbols, annualized metrics are only approximate unless you adapt the calendar assumptions.
- Bootstrap intervals may still understate uncertainty during structural breaks because the future may not resemble any resampled past block.
- Forward-return tables can look attractive when a regime occurs rarely; sample count should always be read with the mean.
- Hourly bars from API vendors can contain gaps, late prints, or symbol-specific quirks that distort fitted transitions.
- A conservative flat signal is a feature, not a bug: low-confidence bars are intentionally filtered instead of forced into trades.
- Multi-asset robustness helps, but three or four liquid coins still do not guarantee the signal generalizes across regimes or venues.
- The autoresearch score is a ranking heuristic for candidate selection, not an economic truth; human review should still approve any change promoted to the main app defaults.
- Daily confirmation can reduce trade count sharply if the daily lane is weak or chronically flat, so always read it together with coverage and trade-count diagnostics.
