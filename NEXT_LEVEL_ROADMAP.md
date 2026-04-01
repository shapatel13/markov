# Next-Level Roadmap

This roadmap is for pushing the current local HMM regime app toward serious research quality without pretending it is an execution-grade live trading system.

## Frozen Baseline

- Stable app baseline remains on `main` at commit `633b678`.
- New work happens on `codex/next-level-research`.
- Rule: do not degrade the current inspectable app while experimenting.
- Rule: keep the current Streamlit app usable even if new lab features are added elsewhere.

## Target

Build a stronger delayed-data regime research stack that is:

- more robust than the current app
- harder to overfit
- more interpretable for discretionary review
- closer to professional research quality even without real-time execution data

This does not aim to become a live brokerage or exchange execution engine.

## What We Have Now

Current strengths:

- purged and embargoed walk-forward validation
- block-bootstrap intervals
- cost stress testing
- 4H plus daily confirmation flow
- model comparison and stability diagnostics
- parameter sweeps and feature-pack comparisons
- robustness basket checks
- exportable artifacts and signal reports

Current bottlenecks:

- compact feature space
- simple state-to-action mapping based on one horizon mean edge
- no ensemble or seed-consensus layer
- no true nested model/feature selection layer
- limited trade-level diagnostics
- no external context data beyond OHLCV
- small-sample inflation still easy on 4H BTC

## Success Criteria

Promote changes only if they improve the stack across multiple dimensions:

1. higher out-of-sample Sharpe after costs
2. stronger bootstrap lower bounds
3. more stable state alignment across retrains
4. better robustness across BTC, ETH, SOL and at least one additional asset
5. acceptable trade count and coverage
6. no dependence on one lucky state count or one narrow sample slice

Good changes should improve the median case, not just the best screenshot.

## Research Lanes

### Lane 1: Better Features

Highest priority. The current model likely needs better market description more than more threshold tuning.

Add next:

1. Trend structure
   - ADX
   - directional movement (`+DI`, `-DI`)
   - rolling slope of log price
   - distance from multi-horizon moving averages
   - Donchian channel position
   - breakout distance from rolling highs/lows

2. Mean reversion and stretch
   - RSI
   - stochastic oscillator
   - Bollinger z-score
   - percentile rank of returns over rolling windows
   - distance from rolling VWAP proxy

3. Volatility regime
   - realized skew
   - realized kurtosis
   - vol-of-vol
   - Parkinson volatility
   - Garman-Klass volatility
   - Bollinger bandwidth
   - Keltner width
   - squeeze flags

4. Candle and range structure
   - body-to-range ratio
   - upper/lower wick ratios
   - gapless trend persistence
   - close location within bar
   - rolling expansion vs compression

5. Volume and participation
   - OBV trend
   - Chaikin money flow proxy
   - volume trend slope
   - relative volume across session buckets
   - volume-volatility interaction terms

6. Time and seasonality
   - hour-of-day embeddings or sin/cos transforms
   - day-of-week transforms
   - weekend vs weekday indicator
   - month-end / quarter-end proximity for risk regime shifts

7. Multi-timeframe context
   - daily trend state mapped onto 4H bars
   - daily vol state mapped onto 4H bars
   - 4H-to-daily trend disagreement features
   - rolling higher-timeframe compression / expansion flags

8. External delayed context, if available
   - funding rate
   - perp basis
   - open interest
   - stablecoin dominance / BTC dominance
   - equity index and dollar index regime proxies
   - on-chain or exchange flow proxies

### Lane 2: Better Regime-to-Trade Mapping

Current mapping is too simple. It uses one validation horizon mean.

Upgrade path:

1. Multi-horizon regime scoring
   - score each state using 6, 12, 24, and 72 bar forward returns together
   - require sign consistency across horizons
   - penalize states that only work at one horizon

2. Shrinkage and support weighting
   - shrink small-sample state edges toward zero
   - require minimum effective sample support
   - prefer lower confidence bound of state edge instead of raw mean

3. Better labels
   - move from `risk_on` / `flat` only to:
     - persistent trend
     - squeeze / setup
     - unstable / avoid
     - mean-reverting / no-trade

4. Better action policy
   - state can be interesting but still non-tradable
   - separate `state quality` from `trade permission`

### Lane 3: Ensemble and Consensus

Very high value. The model is still too easy to fool with one seed and one state count.

Add:

1. seed ensemble
   - same state count, multiple random seeds
   - only trade when state direction is consistent across seeds

2. state-count ensemble
   - trade only when 5/6/7 or 6/7/8 state models broadly agree

3. fold-memory consensus
   - require the regime label to have historically stable directional meaning across recent refits

4. higher-timeframe consensus
   - extend the current daily confirmation into richer agreement logic
   - daily neutral can pass
   - daily opposition blocks
   - daily strong agreement can size conviction score, even if not position size yet

### Lane 4: Validation and Anti-Overfitting

This is where we get closer to professional research discipline.

Add:

1. nested selection
   - use an inner layer for feature/parameter selection
   - evaluate promoted candidates on an untouched outer holdout

2. regime-bucket holdouts
   - explicitly test bull, bear, and chop eras separately

3. stress perturbations
   - small timestamp shifts
   - feature noise injection
   - slight cost increases
   - missing-bar simulations

4. seed stability requirement
   - reject candidates that only work for one random initialization

5. benchmark ladder
   - buy and hold
   - vol filter only
   - simple trend filter
   - breakout baseline
   - moving-average crossover baseline

### Lane 5: Diagnostics and Interpretability

Add:

1. trade-level analytics
   - trade win rate
   - average winner
   - average loser
   - expectancy
   - profit factor
   - MAE / MFE
   - holding-time distribution

2. feature usefulness diagnostics
   - ablation leaderboard
   - permutation degradation
   - state signature tables by feature pack

3. confirmation diagnostics
   - 4H raw vs 4H+daily side-by-side
   - daily confirm / neutral / block timeline
   - blocked trade audit table

4. quality gates
   - auto-flag when sample is too small
   - auto-flag when bootstrap crosses zero
   - auto-flag when robustness median is negative

## Best Feature Engineering Candidates To Build First

If we want the highest ROI first, do these in order:

1. ADX, `+DI`, `-DI`
2. RSI
3. Bollinger z-score and bandwidth
4. Donchian channel position and breakout distance
5. realized skew and kurtosis
6. Parkinson and Garman-Klass volatility
7. body/wick structure features
8. hour-of-day and weekend seasonality
9. daily-on-4H context features
10. external delayed context such as funding and open interest

These have a better chance of changing regime separability than more threshold tuning alone.

## Specific Experiments Worth Running

### Experiment Set A: Feature Packs

Goal:
test whether better market description materially improves regime quality

Candidates:

- `trend_strength`
- `mean_reversion`
- `vol_surface`
- `candle_structure`
- `multi_timeframe_context`
- `market_context`
- `regime_mix_v2`

Primary checks:

- out-of-sample Sharpe
- bootstrap lower bound
- stability score
- robustness median Sharpe

### Experiment Set B: State Scoring

Goal:
replace the fragile one-horizon state edge mapping

Candidates:

- multi-horizon mean score
- lower-confidence-bound score
- shrinkage score
- consistency score across horizons
- support-weighted score

Primary checks:

- fewer false positives
- better precision without collapsing trade count

### Experiment Set C: Consensus Filters

Goal:
prefer fewer but more trustworthy trades

Candidates:

- daily confirmation on/off
- seed consensus on/off
- state-count consensus on/off
- stability gating on/off

Primary checks:

- does Sharpe improve without killing all coverage?
- do blocked trades look like bad trades in hindsight?

### Experiment Set D: Robustness Expansion

Goal:
make sure we are not just fitting one recent BTC slice

Add:

- BCH or LTC
- a liquid non-crypto proxy like GLD or QQQ for method stress
- older BTC samples split by distinct eras

Primary checks:

- median robustness
- variance across assets
- consistency of sign, not just magnitude

## Overnight Work Queue

This is the order I would attack if I were running a serious overnight research push.

### Phase 1: Safe Research Infrastructure

1. Keep the app frozen on `main`
2. Build a lab module and CLI for feature-pack experiments
3. Add nested selection and promoted-candidate holdout scoring
4. Add trade-level analytics

### Phase 2: Core Feature Expansion

1. Add trend-strength indicators
2. Add mean-reversion indicators
3. Add richer volatility estimators
4. Add candle-structure features
5. Add seasonality features
6. Add daily-on-4H context features

### Phase 3: Better Decision Rules

1. multi-horizon state scoring
2. shrinkage for weak states
3. seed-consensus and state-count-consensus filters
4. stronger `avoid / no-trade` states

### Phase 4: Robustness Promotion

1. expand asset basket
2. compare across market eras
3. reject fragile candidates automatically
4. promote only candidates that beat simple baselines

## What I Would Not Over-Invest In Yet

Avoid spending too much time right now on:

- more threshold-only tuning
- more state counts beyond the current neighborhood
- live execution logic
- broker integration
- intrabar modeling without better data
- fancy model classes before feature and scoring upgrades

The current bottleneck is more likely representation and validation discipline than model complexity.

## Realistic Near-Term Goal

Without real-time execution data, the right target is:

- serious delayed-data swing research
- stronger end-of-bar signal quality
- better filtered 4H decisions
- richer daily confirmation
- more honest uncertainty reporting

That can get us close to professional research process quality, even if it cannot become professional live execution quality yet.

## Recommended Build Order

If we want the best chance of meaningful improvement, do this next:

1. add feature families: ADX, RSI, Bollinger, Donchian, realized skew/kurtosis
2. add multi-horizon shrinkage-based state scoring
3. add trade-level analytics
4. add nested selection and promoted holdout
5. add seed and state-count consensus
6. add delayed external context data

That is the path most likely to improve real trustworthiness rather than just making the app look more sophisticated.
