# Markov Regime Research

Local, inspectable Hidden Markov Model regime research app with walk-forward retraining, parameter sweeps, guardrails, and report export. The app is intentionally built to emphasize methodological caution over opaque optimization.

## What It Does

- Fetches daily or hourly bars from Financial Modeling Prep.
- Builds a compact return, volatility, range, and volume feature set.
- Runs explicit train / validate / test walk-forward retraining with rolling refits.
- Compares 5 through 9 HMM states side by side.
- Aligns states across retrains and reports stability diagnostics.
- Sweeps posterior threshold, minimum hold, cooldown, and required confirmations.
- Reports block-bootstrap confidence intervals and transaction cost stress tests.
- Exports the signal history as both CSV and JSON.

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

3. Run the local app.

```powershell
streamlit run app.py
```

## CLI

Backtest:

```powershell
python -m markov_regime backtest --symbol SPY --interval 1hour --states 6
```

Sweep:

```powershell
python -m markov_regime sweep --symbol SPY --interval 1hour --states 6
```

Export the signal report:

```powershell
python -m markov_regime export-report --symbol SPY --interval 1hour --states 6
```

Or launch the app directly:

```powershell
python -m streamlit run app.py
```


## Methodology Notes

- Training, validation, and test windows are fully separated in each fold.
- State labels are re-aligned after every refit because raw HMM state IDs are not stable.
- Directional actions come from validation-window forward returns, not the in-sample training fit.
- The strategy prefers flat exposure when posterior confidence is marginal or when validation support is weak.
- Major metrics are paired with moving block bootstrap confidence intervals to avoid treating a single backtest path as definitive.

## Outputs

- Model comparison table for 5-9 states.
- Fold-by-fold diagnostics including AIC, BIC, convergence, and strategy metrics.
- State stability table across retrains.
- Regime-conditioned forward return table for 1, 3, 6, 12, 24, and 72 bars.
- Sensitivity plots for posterior threshold, min hold, cooldown, and confirmation count.
- Exported signal reports in `exports/`.

## Known Failure Modes

- HMM state identities are latent and can rotate or split across retrains even after alignment, so regime names remain heuristic.
- Small validation windows can make state-to-action mappings noisy, especially for infrequent states.
- State-count selection matters: if 5-9 states produce materially different results, the regime story is fragile.
- Backtest quality can deteriorate quickly once transaction costs, slippage, or stale fills are introduced.
- Bootstrap intervals may still understate uncertainty during structural breaks because the future may not resemble any resampled past block.
- Forward-return tables can look attractive when a regime occurs rarely; sample count should always be read with the mean.
- Hourly bars from API vendors can contain gaps, late prints, or symbol-specific quirks that distort fitted transitions.
- A conservative flat signal is a feature, not a bug: low-confidence bars are intentionally filtered instead of forced into trades.
