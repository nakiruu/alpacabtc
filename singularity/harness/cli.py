"""backtest CLI — walk-forward evaluation with cost sim (batch 3.2).

    uv run backtest --symbol BTC/USD --start 2023-01-01 --end 2026-01-01
    uv run backtest --symbol BTC/USD --start 2023-01-01 --end 2026-01-01 --strategy random
    uv run backtest --symbol BTC/USD --start 2023-01-01 --end 2026-01-01 --cost-mode maker
    uv run backtest --symbol BTC/USD --start 2023-01-01 --end 2026-01-01 --cost-mode none

Timeframe defaults to 1Day. Bars are cached to state/bars/ so re-runs don't
hit the Alpaca API. Force a refresh with --refresh.

Cost modes:
    taker  (default) — every trade pays fee + half spread (conservative)
    maker           — every trade earns rebate (optimistic; only correct if
                      the passive fill loop actually fills as maker)
    none            — cost-free evaluation (batch 3.1 behavior)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..adapters.alpaca_crypto.history import HistoryClient
from ..adapters.alpaca_crypto.history import load_bars_cached as load_alpaca_cached
from ..adapters.binance.history import BinanceHistoryClient
from ..adapters.binance.history import load_bars_cached as load_binance_cached
from ..adapters.bitstamp.history import BitstampHistoryClient
from ..adapters.bitstamp.history import load_bars_cached as load_bs_cached
from ..adapters.coingecko.history import CoinGeckoHistoryClient
from ..adapters.coingecko.history import load_bars_cached as load_cg_cached
from ..adapters.cryptocompare.history import CryptoCompareHistoryClient
from ..adapters.cryptocompare.history import load_bars_cached as load_cc_cached
from ..adapters.kraken.history import KrakenHistoryClient
from ..adapters.kraken.history import load_bars_cached as load_kraken_cached
from ..logs import get_logger
from ..signals.tsmom import tsmom as _make_tsmom
from ..signals.tsmom import tsmom_full as _make_tsmom_full
from ..signals.tsmom import tsmom_voltarget as _make_tsmom_voltarget
from .diagnose import compute_fold_diagnostic, print_diagnostic
from .backtest import (
    BacktestResult,
    buy_and_hold,
    flat,
    random_binary,
    random_matched_turnover,
    run_backtest,
)
from .metrics import deflated_sharpe
from .simulate import CostConfig
from .stats import concat_fold_returns, paired_block_bootstrap
from .walkforward import WalkForwardSpec, WalkForwardSplitter

log = get_logger(__name__)


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "flat": flat,
    "random": random_binary,
    # factories (need config); resolved in _run based on CLI flags.
    "tsmom": None,
    "tsmom_voltarget": None,
    "tsmom_full": None,
}


def _parse_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {s!r}") from e


def _cost_config(mode: str, spread_bps: float) -> CostConfig:
    if mode == "none":
        return CostConfig(enabled=False)
    return CostConfig(is_maker=(mode == "maker"), assumed_spread_bps=spread_bps)


def _print_report(result: BacktestResult) -> None:
    print(f"\n=== backtest — {result.strategy_name} on {result.symbol} ({result.timeframe}) ===")
    print(f"cost mode      : is_maker={result.cost_config.is_maker}, "
          f"spread={result.cost_config.assumed_spread_bps:.1f}bps\n")
    if result.n_folds == 0:
        print("no folds fit — extend the date range")
        return
    header = (f"{'fold':>4} {'n_ret':>6} "
              f"{'gross_SR':>9} {'net_SR':>9} {'DSR':>8} "
              f"{'net_dd':>8} {'net_ret':>9} {'n_trades':>9} {'cost_bps':>9}")
    print(header)
    dsr_values: list[float] = []
    for f in result.per_fold:
        gross = f.gross_metrics
        net = f.net_metrics
        cb = f.cost_breakdown
        dsr = deflated_sharpe(net.annualized_sharpe, f.net_returns,
                              n_trials=1, timeframe=result.timeframe)
        dsr_values.append(dsr)
        print(f"{f.fold.index:>4} {net.n:>6} "
              f"{gross.annualized_sharpe:>9.3f} {net.annualized_sharpe:>9.3f} "
              f"{dsr:>8.3f} "
              f"{net.max_drawdown:>8.2%} {net.total_return:>9.2%} "
              f"{cb.n_trades:>9d} {cb.total_cost_bps:>9.1f}")
    print()
    print(f"folds              : {result.n_folds}")
    print(f"mean gross Sharpe  : {result.mean_sharpe_gross:+.3f}")
    print(f"mean net Sharpe    : {result.mean_sharpe_net:+.3f}")
    if dsr_values:
        import statistics as _s
        print(f"mean DSR           : {_s.fmean(dsr_values):+.3f}")
        print(f"positive DSR folds : {sum(1 for x in dsr_values if x > 0)}/{len(dsr_values)}")
    print(f"negative folds     : {result.n_negative_folds}/{result.n_folds}  (net)")
    print(f"total turnover     : {result.total_turnover:.2f}  (|Δpos| units, aggregated)")


async def _run(args) -> int:
    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.strategy in ("tsmom", "tsmom_voltarget", "tsmom_full"):
        lookbacks = tuple(int(x) for x in args.tsmom_lookbacks.split(","))
        if args.strategy == "tsmom":
            strategy = _make_tsmom(
                lookbacks=lookbacks, enter=args.tsmom_enter, exit_=args.tsmom_exit,
            )
        elif args.strategy == "tsmom_voltarget":
            strategy = _make_tsmom_voltarget(
                lookbacks=lookbacks, enter=args.tsmom_enter, exit_=args.tsmom_exit,
                target_vol=args.target_vol, vol_lookback=args.vol_lookback,
                rebalance_band=args.rebalance_band,
            )
        else:  # tsmom_full
            strategy = _make_tsmom_full(
                lookbacks=lookbacks, enter=args.tsmom_enter, exit_=args.tsmom_exit,
                target_vol=args.target_vol, vol_lookback=args.vol_lookback,
                rebalance_band=args.rebalance_band,
                regime_vol_lookback=args.regime_vol_lookback,
                regime_baseline_lookback=args.regime_baseline_lookback,
                regime_threshold_ratio=args.regime_threshold_ratio,
                regime_risk_off_multiplier=args.regime_risk_off_multiplier,
                regime_sticky_bars=args.regime_sticky_bars,
            )
        if args.warmup_bars == 0:
            args.warmup_bars = max(
                max(lookbacks),
                args.vol_lookback,
                args.regime_baseline_lookback,
            ) + 10
    else:
        strategy = STRATEGIES[args.strategy]

    start = args.start
    end = args.end
    if start >= end:
        print(f"[fatal] --start ({start.date()}) must be before --end ({end.date()})")
        return 2
    cache_dir = Path(args.cache_dir)

    if args.data_source == "binance":
        async with BinanceHistoryClient() as hc:
            bars = await load_binance_cached(
                hc, args.symbol, start, end, cache_dir,
                timeframe=args.timeframe, force_refresh=args.refresh,
            )
    elif args.data_source == "binance-us":
        async with BinanceHistoryClient(base_url="https://api.binance.us") as hc:
            bars = await load_binance_cached(
                hc, args.symbol, start, end, cache_dir,
                timeframe=args.timeframe, force_refresh=args.refresh,
            )
    elif args.data_source == "kraken":
        async with KrakenHistoryClient() as hc:
            bars = await load_kraken_cached(
                hc, args.symbol, start, end, cache_dir,
                timeframe=args.timeframe, force_refresh=args.refresh,
            )
    elif args.data_source == "cryptocompare":
        async with CryptoCompareHistoryClient(api_key=settings.cryptocompare_api_key or None) as hc:
            bars = await load_cc_cached(
                hc, args.symbol, start, end, cache_dir,
                timeframe=args.timeframe, force_refresh=args.refresh,
            )
    elif args.data_source == "coingecko":
        async with CoinGeckoHistoryClient() as hc:
            bars = await load_cg_cached(
                hc, args.symbol, start, end, cache_dir,
                timeframe=args.timeframe, force_refresh=args.refresh,
            )
    elif args.data_source == "bitstamp":
        async with BitstampHistoryClient() as hc:
            bars = await load_bs_cached(
                hc, args.symbol, start, end, cache_dir,
                timeframe=args.timeframe, force_refresh=args.refresh,
            )
    else:
        async with HistoryClient(
            api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key
        ) as hc:
            bars = await load_alpaca_cached(
                hc, args.symbol, start, end, cache_dir,
                timeframe=args.timeframe, force_refresh=args.refresh,
            )
    log.info("bars_loaded", n=len(bars), source=args.data_source, symbol=args.symbol,
             first=bars[0].ts.isoformat() if bars else None,
             last=bars[-1].ts.isoformat() if bars else None)

    if not bars:
        print("[fatal] no bars returned — check date range and credentials")
        return 2

    spec = WalkForwardSpec.default_daily() if args.timeframe == "1Day" else WalkForwardSpec(
        train_bars=args.train_bars, val_bars=args.val_bars,
        test_bars=args.test_bars, advance_bars=args.advance_bars,
    )
    cost_config = _cost_config(args.cost_mode, args.spread_bps)
    splitter = WalkForwardSplitter(spec)
    result = run_backtest(
        strategy=strategy, strategy_name=args.strategy,
        bars=bars, splitter=splitter,
        symbol=args.symbol, timeframe=args.timeframe,
        cost_config=cost_config,
        warmup_bars=args.warmup_bars,
    )
    _print_report(result)

    # ---- Fold diagnostics ----
    _maybe_print_diagnostics(args, bars, splitter, result)

    # ---- Bootstrap vs buy-and-hold benchmark ----
    if args.vs_benchmark != "none" and args.strategy != args.vs_benchmark:
        bench = STRATEGIES[args.vs_benchmark]
        bench_result = run_backtest(
            strategy=bench, strategy_name=args.vs_benchmark,
            bars=bars, splitter=splitter,
            symbol=args.symbol, timeframe=args.timeframe,
            cost_config=cost_config,
            warmup_bars=0,  # benchmarks don't need lookback
        )
        _print_bootstrap_section(
            title=f"strategy vs benchmark ({args.vs_benchmark})",
            result_a=result, result_b=bench_result,
            n_boot=args.bootstrap_n, block=args.bootstrap_block,
            timeframe=args.timeframe, seed=args.bootstrap_seed,
        )

    # ---- Null-gate test ----
    if args.null_gate:
        n_folds = result.n_folds or 1
        target_turnover = result.total_turnover / n_folds
        null_strategy = random_matched_turnover(target_turnover, seed=args.null_seed)
        null_result = run_backtest(
            strategy=null_strategy, strategy_name="null_matched",
            bars=bars, splitter=splitter,
            symbol=args.symbol, timeframe=args.timeframe,
            cost_config=cost_config,
            warmup_bars=0,
        )
        gate = _print_bootstrap_section(
            title=f"NULL-GATE (random, matched turnover ≈ {target_turnover:.1f}/fold, seed={args.null_seed})",
            result_a=result, result_b=null_result,
            n_boot=args.bootstrap_n, block=args.bootstrap_block,
            timeframe=args.timeframe, seed=args.bootstrap_seed,
        )
        # Gate decision: reject null → strategy is real. Fail to reject → null-indistinguishable.
        alpha = args.gate_alpha
        gate_pass = gate.p_value < alpha and gate.observed_diff > 0
        print(f"\ngate ({'p<' + str(alpha):>8}) : [{'PASS' if gate_pass else 'FAIL'}]")
        if not gate_pass:
            print("  strategy cannot be distinguished from a random-timing null "
                  "at the same turnover. Do not deploy.")
        return 0 if gate_pass else 1

    return 0


def _maybe_print_diagnostics(args, bars, splitter, result: BacktestResult) -> None:
    """Dump per-fold diagnostic reports if --diagnose-fold or --diagnose-worst set."""
    fold_indices_to_diagnose: list[int] = []
    if args.diagnose_worst > 0:
        # Pick the N folds with worst net Sharpe
        ranked = sorted(
            result.per_fold,
            key=lambda f: f.net_metrics.annualized_sharpe,
        )[:args.diagnose_worst]
        fold_indices_to_diagnose.extend(f.fold.index for f in ranked)
    if args.diagnose_fold:
        # comma-separated list of fold indices
        for x in args.diagnose_fold.split(","):
            x = x.strip()
            if x:
                fold_indices_to_diagnose.append(int(x))

    if not fold_indices_to_diagnose:
        return

    # Only tsmom_full/voltarget need the extra knobs; for other strategies the
    # diagnostic still runs but the vol/regime numbers reflect what tsmom_full
    # WOULD have done — not what the actual strategy did. Add a warning.
    if args.strategy not in ("tsmom", "tsmom_voltarget", "tsmom_full"):
        print(f"\n[warn] --diagnose-* recomputes tsmom_full signals regardless of --strategy; "
              f"per-bar breakdown for {args.strategy!r} won't match its actual positions")

    lookbacks = tuple(int(x) for x in args.tsmom_lookbacks.split(","))
    folds = splitter.folds(len(bars))
    fold_by_idx = {f.index: f for f in folds}

    print("\n=== fold diagnostics ===")
    for idx in dict.fromkeys(fold_indices_to_diagnose):   # dedupe, preserve order
        fold = fold_by_idx.get(idx)
        if fold is None:
            print(f"[warn] fold {idx} not in result ({len(folds)} total)")
            continue
        diag = compute_fold_diagnostic(
            all_bars=bars, fold=fold, warmup_bars=args.warmup_bars,
            lookbacks=lookbacks, enter=args.tsmom_enter, exit_=args.tsmom_exit,
            target_vol=args.target_vol, vol_lookback=args.vol_lookback,
            rebalance_band=args.rebalance_band,
            regime_vol_lookback=args.regime_vol_lookback,
            regime_baseline_lookback=args.regime_baseline_lookback,
            regime_threshold_ratio=args.regime_threshold_ratio,
            regime_risk_off_multiplier=args.regime_risk_off_multiplier,
            regime_sticky_bars=args.regime_sticky_bars,
        )
        print_diagnostic(diag)


def _print_bootstrap_section(
    *,
    title: str,
    result_a: BacktestResult,
    result_b: BacktestResult,
    n_boot: int,
    block: int | None,
    timeframe: str,
    seed: int,
):
    ra = concat_fold_returns([f.net_returns for f in result_a.per_fold])
    rb = concat_fold_returns([f.net_returns for f in result_b.per_fold])
    boot = paired_block_bootstrap(
        ra, rb, n_bootstrap=n_boot, block_size=block, timeframe=timeframe, seed=seed,
    )
    print(f"\n=== bootstrap — {title} ===")
    print(f"n_bootstrap    : {boot.n_bootstrap}")
    print(f"block_size     : {boot.block_size}")
    print(f"observed diff  : {boot.observed_diff:+.3f} Sharpe (A - B)")
    print(f"95% CI         : [{boot.ci_low:+.3f}, {boot.ci_high:+.3f}]")
    print(f"two-sided p    : {boot.p_value:.4f}")
    return boot


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest CLI")
    parser.add_argument("--symbol", required=True, help="e.g. BTC/USD")
    parser.add_argument("--start", required=True, type=_parse_date, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=_parse_date, help="YYYY-MM-DD")
    parser.add_argument("--timeframe", default="1Day", choices=["1Day", "1Hour", "1Min"])
    parser.add_argument("--strategy", default="buy_and_hold", choices=list(STRATEGIES))
    parser.add_argument("--cost-mode", default="taker", choices=["taker", "maker", "none"],
                        help="taker=conservative default, maker=optimistic (rebate), "
                             "none=cost-free (batch 3.1 behavior)")
    parser.add_argument("--spread-bps", type=float, default=3.0,
                        help="Stylized book spread in bps (default 3, matches live BTC/USD)")
    parser.add_argument("--data-source", default="alpaca",
                        choices=["alpaca", "binance", "binance-us", "kraken",
                                 "cryptocompare", "coingecko", "bitstamp"],
                        help="alpaca (2022+); binance (US-blocked); binance-us (2019+); "
                             "kraken (~720 bars only); cryptocompare (needs API key); "
                             "coingecko (needs API key); "
                             "bitstamp (2011+, OHLC, NO AUTH — RECOMMENDED for US free-tier)")
    parser.add_argument("--cache-dir", default="./state/bars")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache, re-fetch bars")
    parser.add_argument("--train-bars", type=int, default=360)
    parser.add_argument("--val-bars", type=int, default=90)
    parser.add_argument("--test-bars", type=int, default=90)
    parser.add_argument("--advance-bars", type=int, default=90)
    parser.add_argument("--vs-benchmark", default="buy_and_hold",
                        choices=["buy_and_hold", "flat", "none"],
                        help="Bootstrap Sharpe diff vs this benchmark (or 'none' to skip)")
    parser.add_argument("--null-gate", action="store_true",
                        help="Run random-matched-turnover null and require p<alpha to pass")
    parser.add_argument("--gate-alpha", type=float, default=0.05,
                        help="Significance threshold for --null-gate (default 0.05)")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--bootstrap-block", type=int, default=None,
                        help="Circular block size (default: n^(1/3))")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument("--warmup-bars", type=int, default=0,
                        help="Bars of pre-test-window history passed to strategy "
                             "(auto-set for tsmom based on longest lookback)")
    parser.add_argument("--tsmom-lookbacks", default="30,60,90,180",
                        help="Comma-separated lookback windows in bars (default: 30,60,90,180)")
    parser.add_argument("--tsmom-enter", type=float, default=0.25,
                        help="Hysteresis enter threshold (default 0.25)")
    parser.add_argument("--tsmom-exit", type=float, default=-0.10,
                        help="Hysteresis exit threshold (default -0.10)")
    parser.add_argument("--target-vol", type=float, default=0.40,
                        help="Vol-target overlay: annualized target (default 0.40 = 40%)")
    parser.add_argument("--vol-lookback", type=int, default=30,
                        help="Bars for realized-vol estimate (default 30)")
    parser.add_argument("--rebalance-band", type=float, default=0.15,
                        help="Vol-target: only rebalance when |Δ multiplier| > this (default 0.15)")
    parser.add_argument("--regime-vol-lookback", type=int, default=30,
                        help="Regime gate: bars for current-vol estimate (default 30)")
    parser.add_argument("--regime-baseline-lookback", type=int, default=180,
                        help="Regime gate: bars for baseline vol median (default 180)")
    parser.add_argument("--regime-threshold-ratio", type=float, default=1.5,
                        help="Regime gate: trigger risk-off when current > this × baseline (default 1.5)")
    parser.add_argument("--regime-risk-off-multiplier", type=float, default=0.5,
                        help="Regime gate: exposure during risk-off (default 0.5)")
    parser.add_argument("--regime-sticky-bars", type=int, default=20,
                        help="Regime gate: minimum bars to stay risk-off before allowing exit (default 20)")
    parser.add_argument("--diagnose-fold", default="",
                        help="Comma-separated fold indices to print per-bar diagnostics for (e.g. '12,21,29')")
    parser.add_argument("--diagnose-worst", type=int, default=0,
                        help="Diagnose the N worst-Sharpe folds automatically")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
