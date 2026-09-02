# Optimal Build — BTC Systematic Trading on Alpaca

**An engineering plan, ordered by expected return on effort**

*September 2026*

---

## 0. Design principles

Five rules, all derived from the cost arithmetic. Everything downstream follows from them.

1. **Execution before signal.** Maker-vs-taker moves the viable holding-period floor from ~12 hours to ~4. No model improvement you will realistically achieve is worth that much. Build the execution layer first.
2. **Model volatility, not returns.** Volatility is genuinely predictable. Returns barely are. Put effort where the signal is.
3. **Overlays outrank strategies.** Vol targeting, maker conversion, regime gating, and posterior sizing all compose with whatever signal you run. Build them once, apply everywhere.
4. **Buy-and-hold is the gate, not the floor.** Any component that doesn't beat 25 bps once and then nothing does not get capital.
5. **The harness precedes the model.** Build the thing that can prove a strategy wrong before building strategies. Otherwise you'll build strategies that can't be proven wrong.

**Reuse over rewrite.** Singularity's signal families, ATR risk logic, and InfluxDB/Grafana stack carry over. What must change: the crypto adapter (no brackets, no shorting), the cost model (25 bps, not equity commission-free), and the walk-forward harness. Treat this as a new asset-class module inside the existing system, not a parallel system.

---

## 1. Target architecture

```
singularity/
├── adapters/
│   ├── alpaca_crypto/
│   │   ├── stream.py          # WS: trades, quotes, orderbooks → Influx
│   │   ├── rest.py            # bars, latest orderbook, assets
│   │   ├── orders.py          # market/limit/stop_limit, gtc/ioc only
│   │   └── activities.py      # CFEE/FEE reconciliation (T+1)
│   └── external/
│       └── derivs.py          # funding, basis, OI (Binance/Bybit/Deribit)
├── costs/
│   ├── model.py               # fee tier + spread + slippage + fill prob
│   └── calibration.py         # fits model against realized fills
├── execution/
│   ├── passive.py             # cancel/replace maker loop
│   ├── supervisor.py          # client-side bracket state machine
│   ├── reconcile.py           # startup + periodic state repair
│   └── watchdog.py            # heartbeat → flatten on failure
├── features/
│   ├── price.py               # TA families, multi-window
│   ├── vol.py                 # realized, bipower, GARCH, vol-of-vol
│   ├── micro.py               # book imbalance, OFI, depth slope
│   ├── cross.py               # panel, PC loadings, equity proxies
│   ├── derivs.py              # funding z-score, basis, OI delta
│   └── selection.py           # block-rank stability selection
├── signals/
│   ├── tsmom.py               # blended lookback + hysteresis
│   ├── xsec.py                # cross-sectional top-k
│   └── breakout.py            # Donchian
├── overlays/
│   ├── voltarget.py           # banded rebalance
│   ├── regime.py              # sticky HMM / BOCPD gate
│   └── sizing.py              # posterior fractional Kelly
├── ml/
│   ├── labels.py              # triple-barrier, meta-labels
│   ├── model.py               # XGBoost quantile objectives
│   └── tune.py                # Optuna TPE, fold-local
├── harness/
│   ├── walkforward.py         # 12/3/3 rolling, 27 folds
│   ├── simulate.py            # fill simulation w/ queue + miss modeling
│   └── stats.py               # DSR, PBO, fold decomposition
└── ops/
    ├── state.py               # durable position/intent store
    └── dashboards/            # Grafana JSON
```

**Module contracts.** Each layer emits a typed object and knows nothing about the layer above:

```
Signal    → target_weight ∈ [0, 1]      (before risk)
Overlay   → target_weight ∈ [0, 1]      (after risk)
Sizer     → target_qty in BTC, rounded to 0.0001
Execution → OrderIntent → fills → realized cost
```

Enforce that `[0, 1]` clamp in the type system if you can. It is the one constraint that no amount of model cleverness can violate on this venue, and the clamp is where a lot of bugs would otherwise surface as phantom shorts in backtests.

---

## 2. Phase 0 — Data capture (start today, ~1 day)

This runs before everything else because the data has strictly increasing option value and cannot be backfilled.

**Deliverable:** a supervised process writing the Alpaca crypto WS stream to InfluxDB, running continuously.

```python
# subscribe
{"action":"subscribe",
 "trades":["BTC/USD"],
 "quotes":["BTC/USD","ETH/USD","ETH/BTC"],
 "orderbooks":["BTC/USD","ETH/USD"]}
```

**Influx measurements:**

| Measurement | Tags | Fields | Retention |
|---|---|---|---|
| `crypto_trade` | symbol | price, size, side | 90d raw → 5y downsampled |
| `crypto_quote` | symbol | bid_px, bid_sz, ask_px, ask_sz | 30d raw |
| `crypto_book` | symbol, level | bid_px, bid_sz, ask_px, ask_sz | 14d raw → derived features 5y |
| `book_features` | symbol | imb_1, imb_5, imb_10, ofi_1m, depth_slope, spread_bps | 5y |

Raw L2 is heavy. Compute `book_features` on ingest at 1-second cadence and set aggressive retention on the raw book — you want the *features* forever, not the snapshots.

Handle the `"r": true` full-snapshot flag: rebuild book state on snapshot, apply deltas otherwise. Log every reconnect and gap; a feature computed across a silent gap is a landmine.

**Gate to proceed:** 7 consecutive days of capture with <0.5% gap time.

---

## 3. Phase 1 — Cost model (~3 days)

Nothing downstream means anything until this is right. This is the module that decides whether every other module is telling you the truth.

### 3.1 Specification

```python
def round_trip_cost(qty, side, book, is_maker) -> Cost:
    fee      = TIER[volume_30d][is_maker]      # 0.0015 / 0.0025
    spread   = crossing_cost(qty, book, is_maker)
    impact   = walk_the_book(qty, book)
    return Cost(fee=fee, spread=spread, impact=impact)
```

Three components, tracked separately so you can attribute where the money goes.

- `fee` — from the tier table, on the **received asset**. For `ETH/BTC` you pay ETH in, BTC out. Per-asset accounting, not dollar accounting.
- `spread` — zero if you rest passively and get filled; half-spread or worse if you cross.
- `impact` — walk the actual recorded book from Phase 0. At retail size this is near zero for BTC/USD and non-trivial for thin `*/BTC` pairs.

### 3.2 Fill probability model

The piece most backtests omit and the one that determines whether Phase 2 works.

```python
def fill_prob(offset_bps, wait_seconds, vol_regime) -> float
```

Estimate empirically from Phase 2's live logs. Until you have those, use a conservative placeholder: 60% fill at the touch within 60s, and assume **the 40% you miss are adversely selected** — the market moved away from you. Model the miss cost as the subsequent price move, not as zero.

### 3.3 Calibration loop

Every order logs: intent price, submitted price, fill price(s), fill/no-fill, timestamp, book snapshot at submission. Nightly, join against `CFEE`/`FEE` records from the Activities API and fit:

- realized maker ratio vs. modeled
- implementation shortfall vs. modeled
- fee accrual vs. actual

**Gate:** modeled vs. realized cost within 3 bps on a rolling 100-trade window. Until this gate passes, treat every backtest number as unverified.

---

## 4. Phase 2 — Execution layer (~1 week)

The highest-value engineering in the whole build.

### 4.1 Passive fill loop

```
submit limit @ bid (gtc)
  ├─ filled            → done, maker fee
  ├─ t > T1  → cancel, resubmit @ bid + 1 tick
  ├─ t > T2  → cancel, resubmit @ mid
  └─ t > T3  → cross with ioc, accept taker
```

Tune `T1..T3` against **measured implementation shortfall**, not intuition. For a 1-week holding period, `T3` of several minutes costs nothing in signal decay and saves 20 bps round trip.

Handle partial fills explicitly — track filled qty, cancel-replace only the remainder, and never let a partial leave your position accounting inconsistent.

### 4.2 Bracket supervisor

Since bracket/OCO don't exist for crypto:

1. Entry fills → compute `stop = entry − k·ATR`, `target = entry + m·ATR` → **persist to durable store before anything else**
2. Supervisor consumes the trade stream, evaluates exits
3. Trigger → submit marketable `ioc`. **Never a resting `stop_limit`** as the primary exit; it won't fill in a gap-through, and crypto gaps through routinely
4. Optionally place a *wide* `stop_limit` as a catastrophic backstop — free to have, unreliable to depend on

### 4.3 Reconciliation and watchdog

On every startup, before any trading logic runs:

```
positions = GET /v2/positions
orders    = GET /v2/orders?status=open
intent    = state_store.load()
diff      = reconcile(positions, orders, intent)
if diff: alert + repair, do not trade until clean
```

Heartbeat every 30s. If the supervisor dies holding a position, you have unprotected exposure in a 24/7 market with no server-side stop. A dead-man's-switch that flattens on heartbeat loss is not paranoid on this venue — it's the difference between a bad week and a blown account.

**Gate:** 30 days of paper trading with zero unreconciled state events and zero stranded positions.

---

## 5. Phase 3 — Backtest harness (~1 week)

Build this *before* the strategies. It is what stops you from believing things that aren't true.

### 5.1 Walk-forward protocol

- Non-anchored rolling: 12mo train / 3mo validation / 3mo test, advance 3mo → ~27 folds over 2018–2026
- Parameters fit on train only; validation for tuning, early stopping, selection; **test touched once**
- Retrain on train+validation before the test window
- Feature engineering **inside each fold** with a warm-up buffer that is then discarded. Scaling and target standardization refit per fold. A rolling z-score computed over the full sample before splitting is leakage, and it is the single most common way a crypto backtest lies.

### 5.2 Fill simulation

Not a price series multiplication. For each intended trade:

```
draw fill outcome from fill_prob(offset, wait, vol_regime)
  filled → maker fee + queue-position penalty
  missed → apply adverse-selection cost (subsequent move)
```

### 5.3 Statistics — the non-negotiable list

| Test | Purpose | Threshold |
|---|---|---|
| Deflated Sharpe Ratio | corrects for search over configs | DSR > 0 |
| PBO (CSCV) | probability of backtest overfitting | < 0.5 |
| Minimum trade count | prevents single-trade artifacts | ≥ 20 OOS trades |
| Fold decomposition | is it 3 folds carrying 27? | report all 27 |
| vs. buy-and-hold | the real benchmark | bootstrap Sharpe diff |

Use paired circular block bootstrap for Sharpe comparisons — returns are serially dependent and standard tests will overstate your confidence.

**Report fold-level results always.** If 24 of 27 folds are flat and 3 carry everything, you have a bull-market beta, not a strategy. The aggregate hides this by construction.

**Gate:** harness reproduces a known-null strategy (random entry, same turnover) as statistically indistinguishable from zero after costs. If random noise looks profitable in your harness, the harness is broken.

---

## 6. Phase 4 — Core strategy (~2 weeks)

### 6.1 TSMOM primary

```python
signals = [sign(ret(L) / vol(L)) for L in (30, 60, 90, 180)]  # days
raw     = mean(signals)
weight  = hysteresis(raw, enter=+0.25, exit=-0.10)   # asymmetric band
```

Blend lookbacks; never optimize a single one. The hysteresis band *is* the cost-aware filter in its simplest form — it directly attacks the whipsaw that destroys TSMOM in chop. Tune `enter`/`exit` on the cost model, not on returns: the entry bar should be a multiple of round-trip cost expressed in signal units.

### 6.2 Volatility target overlay

```python
w_target = min(1.0, sigma_target / sigma_hat)
if abs(w_target - w_current) > 0.15:    # banded — do NOT rebalance continuously
    rebalance()
```

`sigma_hat` from realized vol or bipower variation, not just EWMA of squared returns. The band is what keeps this from generating its own cost drag.

### 6.3 Regime gate

Sticky HDP-HMM or BOCPD on (return, realized vol, volume). Output multiplies gross exposure:

```python
final_weight = tsmom * voltarget * regime_gate
```

**Risk gate only.** The reliable content is "volatility is exploding / my training distribution no longer resembles today." Regime-conditional *return* forecasts multiply two error sources and are much less trustworthy.

**Gate:** net-of-cost Sharpe > 0.4 across ≥20 of 27 folds, DSR > 0, PBO < 0.5, and drawdown materially below buy-and-hold. If it fails, stop here — do not proceed to Phase 5 hoping ML rescues it.

---

## 7. Phase 5 — Alpha extensions (~3 weeks)

Only after Phase 4 passes its gate.

### 7.1 Derivatives positioning features

Highest expected value of anything remaining. Funding rate, futures basis, OI change, long/short ratio from Binance/Bybit/Deribit — data you can't trade, feeding the spot position you can.

Construct as z-scores against trailing windows. Extreme positive funding marks crowded longs and precedes liquidation cascades; this is a direct measurement of leveraged positioning that price alone doesn't give you.

Enters as a feature to §7.2 and as a standalone risk-off trigger.

### 7.2 XGBoost meta-labeling

```
primary   = TSMOM long/flat signal          (direction — already works)
labels    = triple_barrier(entry, ATR_stop, ATR_target, time_limit)
meta      = XGBoost → P(this trade profitable net of cost)
size      = posterior_kelly(meta_output) clipped to [0,1]
```

The ML model **filters and sizes; it does not pick direction.** Direction is the hard part and the primary rule handles it robustly. Precision is what a high-cost environment needs, and that's what meta-labeling delivers.

**Target design:** use quantile objectives (pinball loss at several quantiles) rather than binary classification. A classifier gives you `P(up)` with no magnitude, and the cost filter requires magnitude. Quantiles give you `E[r]` and `Var(r)` → feeding §7.3 directly. This is the bridge between the ML and Bayesian layers, and it's the piece most systems skip.

**Feature selection:** generate the candidate pool, map into ~10 groups, then within each fold split *training* into 4 sequential blocks, rank by absolute Spearman within each block, average ranks, keep one per group. Block-averaging favors stable predictors over ones that spike in a single sub-period. Naive top-k-by-correlation is a disaster here.

**Sample weighting** by label uniqueness — triple-barrier labels always overlap in time, and unweighted training silently double-counts.

### 7.3 Posterior sizing

```python
w = clip(E[r] / Var(r) * kelly_fraction, 0, 1)   # Var includes parameter uncertainty
```

Because `Var` carries parameter uncertainty and not just return variance, the position **shrinks automatically when the model is uncertain** — after regime changes, early in a fold, on out-of-distribution features. Point-estimate Kelly does the opposite. Use ¼ to ½ Kelly; full Kelly on a misspecified crypto model is ruinous.

**Gate:** meta-labeled system beats bare TSMOM on net Sharpe across folds, with DSR > 0. If it doesn't, ship Phase 4 and stop. That is a legitimate outcome.

---

## 8. Phase 6 — Relative value (optional, ~2 weeks)

Only worth building if Phases 4–5 are live and stable, and only with discipline about the cost gate.

**Numéraire rotation.** Hold BTC, trade `*/BTC` pairs. Long `ETH/BTC` is economically long-ETH/short-BTC in cash spot — market-neutral without margin or borrow.

**Selection pipeline:**

```
1. Johansen on the */BTC panel        → cointegrating vectors
2. Fit OU: dX = θ(μ − X)dt + σdW      → half-life = ln2/θ
3. Amplitude = σ / sqrt(2θ)
4. HARD GATE: 2 × amplitude > 4 × round_trip_cost   else DISCARD
5. Kalman filter for time-varying hedge ratio
6. Re-validate cointegration every window; drop failures
```

Step 4 is what separates a relative-value book that works from one that grinds to zero. Most crypto pairs at short half-lives fail it at 50 bps. Be ruthless.

**Do not** put BTC in the residual position. BTC is essentially PC1 in a crypto cross-section — it has almost no idiosyncratic component to trade. It's the hedge instrument.

---

## 9. Monitoring (Grafana)

Panels, in priority order:

1. **Position + unprotected-exposure alarm** — position ≠ 0 with no live supervisor heartbeat. Red, loud, phone.
2. **Realized vs. modeled cost** — rolling 100 trades, split fee/spread/impact. Drift here invalidates every backtest.
3. **Maker ratio** — the number Phase 2 exists to move.
4. **Implementation shortfall distribution** — watch the tail, not the mean.
5. **Live vs. backtest signal agreement** — same inputs should produce the same weight. Divergence means a look-ahead bug in the backtest or a data bug in live.
6. **Fold-equivalent rolling Sharpe** — live performance against the walk-forward distribution. If live sits below the 10th percentile of folds, something broke.
7. **Feature drift** — PSI or KL on each feature vs. training distribution.
8. **Fee reconciliation gap** — T+1 modeled accrual vs. `CFEE` actuals.

---

## 10. Capital ramp and kill criteria

| Stage | Duration | Capital | Advance if |
|---|---|---|---|
| Paper | 30 days | — | zero state errors, cost model within 3 bps |
| Live micro | 60 days | ~$1–2k | maker ratio > 50%, live/backtest signal agreement > 99% |
| Live small | 90 days | 10–20% of allocation | rolling Sharpe within fold distribution |
| Live full | ongoing | full allocation | quarterly re-validation passes |

**Kill criteria — write these down before you deploy, not after:**

- Drawdown exceeds the worst walk-forward fold drawdown × 1.5
- Live Sharpe below the 5th percentile of the fold distribution over 90 days
- Realized cost exceeds model by >10 bps for 2 consecutive weeks
- Any unreconciled position event in live
- Feature PSI > 0.25 on 3+ core features simultaneously

Pre-committing to these is the whole point. A kill criterion invented during a drawdown is not a kill criterion.

---

## 11. Effort summary

| Phase | Effort | Cumulative | Value if you stop here |
|---|---|---|---|
| 0 — Data capture | 1 day | 1d | Optionality, nothing tradable |
| 1 — Cost model | 3 days | 4d | Can evaluate anything honestly |
| 2 — Execution | 1 week | ~2wk | 20 bps/round trip — real money |
| 3 — Harness | 1 week | ~3wk | Can't fool yourself anymore |
| 4 — Core strategy | 2 weeks | ~5wk | **Shippable system** |
| 5 — ML/Bayes | 3 weeks | ~8wk | Incremental, uncertain |
| 6 — Relative value | 2 weeks | ~10wk | Diversification, low P(success) |

**Phase 4 is the shipping point.** Phases 5 and 6 are optional refinements with meaningfully lower expected value per unit of effort. A working, monitored, cost-honest TSMOM + vol-target + regime-gate system beats an elaborate one you never quite trust.

---

## 12. Anti-patterns

- Optimizing a single momentum lookback
- Computing any rolling statistic before the train/test split
- Believing paper-trading fills — no queue, no realistic partials
- Trading to reach a fee tier (costs $250/mo to save 3 bps)
- Reconciling P&L intraday against EOD-posted fees
- Using a resting `stop_limit` as the primary exit
- Regime models producing return forecasts rather than risk gates
- Running TSMOM, cross-sectional, and breakout simultaneously and calling it diversification — they're one bet with three sets of fees
- Reporting aggregate backtest performance without the fold decomposition
- Shipping without pre-committed kill criteria
