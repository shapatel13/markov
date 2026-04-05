# Markov Regime Research

Local, inspectable Hidden Markov Model regime research app with walk-forward retraining, parameter sweeps, guardrails, timeframe comparison, and report export. The app is intentionally built to emphasize methodological caution over opaque optimization.

## What It Does

- Fetches live quotes and recent bars from Financial Modeling Prep, and can automatically backfill deeper crypto intraday history from Coinbase or Yahoo when FMP's hourly cap would starve the walk-forward test.
- Builds multiple feature packs spanning return, trend, volatility, range, EMA distance, compression, ADX/DI trend strength, RSI/Bollinger mean reversion, Donchian breakout context, rolling VWAP gap, realized skew/kurtosis structure, and a causal ATR-normalized momentum lane.
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
- Adds a dedicated candidate-search workflow that ranks feature pack, state count, shorting mode, and confirmation mode on deeper Coinbase-backed crypto history.
- Makes an explicit engine recommendation in the app: use the HMM, use the best simple baseline, or stay in research / flat mode.
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
python -m markov_regime backtest --symbol BTCUSD --interval 4hour --feature-pack mean_reversion --states 8 --provider auto
```

Sweep:

```powershell
python -m markov_regime sweep --symbol BTCUSD --interval 4hour --feature-pack mean_reversion --states 8 --provider auto
```

Export the signal report:

```powershell
python -m markov_regime export-report --symbol BTCUSD --interval 4hour --feature-pack mean_reversion --states 8 --provider auto
```

Compare `4hour`, `1day`, and `1hour`:

```powershell
python -m markov_regime compare-timeframes --symbol BTCUSD --feature-pack mean_reversion --states 8 --provider auto
```

Backtest `4hour` with daily confirmation turned on:

```powershell
python -m markov_regime backtest --symbol BTCUSD --interval 4hour --feature-pack regime_mix --require-daily-confirmation
```

Compare feature packs on the same timeframe:

```powershell
python -m markov_regime compare-feature-packs --symbol BTCUSD --interval 4hour --states 8 --provider auto
```

Rank serious HMM variants on deeper Coinbase-backed history:

```powershell
python -m markov_regime candidate-search --symbol BTCUSD --interval 4hour --provider coinbase --limit 5000
```

Initialize the local autoresearch files:

```powershell
python -m markov_regime init-research
```

Run the constrained autoresearch batch:

```powershell
python -m markov_regime autoresearch
```

Run the full primetime readiness audit:

```powershell
python -m markov_regime readiness-audit --symbol BTCUSD --interval 4hour --feature-pack mean_reversion --states 8 --limit 5000 --provider auto
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
- That same `Methodology` panel now includes a nested holdout check, where inner folds choose sweep settings and the most recent untouched outer folds judge whether those settings still work.
- The current exploratory default operating profile is `BTCUSD` on `4hour` with the `mean_reversion` feature pack, `8` states, and `auto` historical provider. This is a research preset, not a promoted live strategy.
- In `auto` provider mode, the app keeps FMP for live quotes and will prefer Coinbase for long-history crypto intraday bars, with Yahoo as a backup if Coinbase is unavailable.
- The app now defaults to BTC `4hour`, with `1day` alongside it as a slower confirmation lane, because those higher timeframes tend to produce more stable regime structure than `1hour` noise.
- The higher-timeframe defaults now approximate a `12 months train / 3 months validate / 3 months blind test` cadence on `4hour` and `1day`.
- The daily lane is still available as slower context, but it is no longer a hard default veto because deeper-sample testing showed that the current best exploratory candidate did not improve when daily confirmation was forced on.
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
- Nested holdout summary showing inner-selected settings versus untouched outer-fold performance.
- Cross-asset robustness table for the configured basket.
- `4hour` vs `1day` vs `1hour` comparison table for the primary symbol.
- Artifact manifests and CSV snapshots in `artifacts/`.
- Exported signal reports in `exports/`.
- Local autoresearch leaderboard in ignored `results.tsv`.
- Feature-pack-aware autoresearch artifacts for the strongest candidates.
- Primetime readiness reports in `artifacts/primetime/`.
- Candidate-search leaderboards with promotion verdicts and engine recommendations.

## Primetime Audit

- `python -m markov_regime readiness-audit ...` runs both operational and strategy-level checks.
- Operational checks currently verify:
  - the full pytest suite
  - compile/import health
  - historical data fetch
  - live quote fetch and freshness
  - signal export smoke
  - artifact bundle smoke
  - blind out-of-sample integrity
- Strategy checks reuse the app's promotion gates, including:
  - walk-forward fold count
  - nested holdout availability
  - bootstrap support
  - trade count
  - state stability
  - cross-asset robustness
  - baseline comparison
  - sample depth and window auto-adjustment
- The audit intentionally separates platform readiness from strategy promotion:
  - a healthy app can still correctly say the current live setup is not ready for deployment
  - `No Entry` is not a failure if the model does not see enough validated edge

## Autoresearch

- [`research_program.md`](research_program.md) defines the safe search space and the frozen evaluator rules.
- `results.tsv` is local and ignored by Git, so you can run repeated experiment batches without dirtying the repo.
- The scoring loop is intentionally conservative: it ranks candidates using Sharpe, bootstrap lower bounds, state stability, benchmark-relative return, cost breakpoints, and robustness instead of raw win rate alone.
- UI parameter sweeps are diagnostic only. They replay alternate thresholds on the already-observed out-of-sample path and should not be treated as blind model selection.
- The search space now includes feature packs, so autoresearch can test whether the signal improves because the model sees a better market description, not just because entry filters were retuned.
- The autoresearch score now also checks early-vs-late fold confirmation so a candidate that only looks good in one part of the sample is penalized.
- The top-ranked candidates automatically get artifact bundles in `artifacts/` so they can be inspected in the same format as manual app runs.
- This is a constrained optimization harness, not an unrestricted self-modifying agent. That limitation is intentional so the evaluation logic stays trustworthy.

## Candidate Search

- `python -m markov_regime candidate-search ...` ranks feature pack, state count, shorting mode, and confirmation mode on a deeper Coinbase-backed history lane.
- Candidate search is staged: it ranks all requested variants on the primary symbol first, then runs the heavier cross-asset robustness check only on the top-ranked variants.
- The search score includes:
  - stitched blind-OOS Sharpe
  - bootstrap lower bound
  - state stability
  - outer holdout Sharpe
  - cross-asset robustness median
  - best-baseline comparison
- Search output is intentionally allowed to recommend `Use baseline, not HMM` or `Use no active deployment yet` when the HMM is still not promoted.
- In the Streamlit app this panel is optional because it is materially heavier than a single run.

## Known Failure Modes

- HMM state identities are latent and can rotate or split across retrains even after alignment, so regime names remain heuristic.
- Small validation windows can make state-to-action mappings noisy, especially for infrequent states.
- State-count selection matters: if 5-9 states produce materially different results, the regime story is fragile.
- Backtest quality can deteriorate quickly once transaction costs, slippage, or stale fills are introduced.
- Even the richer friction model is still an approximation; it is not a substitute for exchange-specific order book replay or venue-level fill data.
- The `4hour` candles are aggregated locally from hourly API data, so missing hourly bars or late prints can distort the higher-timeframe candle set.
- Financial Modeling Prep currently appears capped to roughly the most recent few thousand hourly bars for crypto, so long-history intraday research should use `provider=auto`, `coinbase`, or `yahoo` instead of raw `fmp`.
- Coinbase and Yahoo are useful research backfills, but they are not perfect substitutes for exchange-level trade and order-book replay.
- Annualization is crypto-first and assumes 24/7 bars; if you use equity symbols, annualized metrics are only approximate unless you adapt the calendar assumptions.
- Bootstrap intervals may still understate uncertainty during structural breaks because the future may not resemble any resampled past block.
- Forward-return tables can look attractive when a regime occurs rarely; sample count should always be read with the mean.
- Hourly bars from API vendors can contain gaps, late prints, or symbol-specific quirks that distort fitted transitions.
- A conservative flat signal is a feature, not a bug: low-confidence bars are intentionally filtered instead of forced into trades.
- Multi-asset robustness helps, but three or four liquid coins still do not guarantee the signal generalizes across regimes or venues.
- The autoresearch score is a ranking heuristic for candidate selection, not an economic truth; human review should still approve any change promoted to the main app defaults.
- Daily confirmation can reduce trade count sharply if the daily lane is weak or chronically flat, so always read it together with coverage and trade-count diagnostics.
