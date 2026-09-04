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

from ..adapters.alpaca_crypto.history import HistoryClient, load_bars_cached
from ..logs import get_logger
from .backtest import (
    BacktestResult,
    buy_and_hold,
    flat,
    random_binary,
    run_backtest,
)
from .metrics import deflated_sharpe
from .simulate import CostConfig
from .walkforward import WalkForwardSpec, WalkForwardSplitter

log = get_logger(__name__)


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "flat": flat,
    "random": random_binary,
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
    strategy = STRATEGIES[args.strategy]

    start = args.start
    end = args.end
    if start >= end:
        print(f"[fatal] --start ({start.date()}) must be before --end ({end.date()})")
        return 2
    cache_dir = Path(args.cache_dir)

    async with HistoryClient(
        api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key
    ) as hc:
        bars = await load_bars_cached(
            hc, args.symbol, start, end, cache_dir,
            timeframe=args.timeframe, force_refresh=args.refresh,
        )
    log.info("bars_loaded", n=len(bars), symbol=args.symbol,
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
    result = run_backtest(
        strategy=strategy, strategy_name=args.strategy,
        bars=bars, splitter=WalkForwardSplitter(spec),
        symbol=args.symbol, timeframe=args.timeframe,
        cost_config=cost_config,
    )
    _print_report(result)
    return 0


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
    parser.add_argument("--cache-dir", default="./state/bars")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache, re-fetch bars")
    parser.add_argument("--train-bars", type=int, default=360)
    parser.add_argument("--val-bars", type=int, default=90)
    parser.add_argument("--test-bars", type=int, default=90)
    parser.add_argument("--advance-bars", type=int, default=90)
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
