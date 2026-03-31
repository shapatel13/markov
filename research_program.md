# Research Program

Safe local autoresearch for the Markov regime app.

## Frozen Evaluator

- Do not mutate the leakage defenses or scoring harness during unattended runs.
- Keep these modules fixed unless a human explicitly asks to change methodology:
  - `src/markov_regime/walkforward.py`
  - `src/markov_regime/bootstrap.py`
  - `src/markov_regime/robustness.py`
  - `src/markov_regime/artifacts.py`

## Allowed Experiment Surface

- `src/markov_regime/features.py`
- `src/markov_regime/strategy.py`
- `src/markov_regime/config.py`
- this `research_program.md`

## Objective

Prefer robust `4hour` BTC research, keep `1day` as a slower confirmation lane,
and treat `1hour` mostly as a noisy baseline instead of the default optimization target.
Keep only changes that improve out-of-sample quality without weakening stability,
bootstrap confidence, or cross-asset robustness.

## Program Spec

```json
{
  "symbol": "BTCUSD",
  "intervals": [
    "4hour",
    "1day",
    "1hour"
  ],
  "feature_packs": [
    "baseline",
    "trend",
    "volatility",
    "regime_mix"
  ],
  "limit": 5000,
  "robustness_symbols": [
    "BTCUSD",
    "ETHUSD",
    "SOLUSD"
  ],
  "state_counts": [
    5,
    6,
    7,
    8,
    9
  ],
  "posterior_thresholds": [
    0.6,
    0.65,
    0.7
  ],
  "min_hold_bars": [
    6,
    12
  ],
  "cooldown_bars": [
    2,
    4,
    8
  ],
  "required_confirmations": [
    2,
    3,
    4
  ],
  "confidence_gap": 0.06,
  "max_candidates": 24,
  "artifact_top_k": 3,
  "auto_adjust_windows": true
}
```
