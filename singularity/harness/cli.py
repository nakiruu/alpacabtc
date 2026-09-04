"""backtest CLI — run a walk-forward evaluation of a strategy.

    uv run backtest --symbol BTC/USD --start 2023-01-01 --end 2026-01-01 \\
                    --strategy buy_and_hold

Currently supports strategies: buy_and_hold, flat.

Timeframe defaults to 1Day. Bars are cached to state/bars/ so re-runs don't
hit the Alpaca API. Force a refresh with --refresh.

Cost-free evaluation — batch 3.2 will layer the fill sim + cost model.
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
    run_backtest,
)
from .walkforward import WalkForwardSpec, WalkForwardSplitter

log = get_logger(__name__)


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "flat": flat,
}


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _print_report(result: BacktestResult) -> None:
    print(f"\n=== backtest — {result.strategy_name} on {result.symbol} ({result.timeframe}) ===\n")
    if result.n_folds == 0:
        print("no folds fit — extend the date range")
        return
    print(f"{'fold':>4} {'n_bars':>6} {'sharpe':>8} {'max_dd':>8} {'total_ret':>10} {'hit_rate':>8}")
    for f in result.per_fold:
        m = f.metrics
        print(
            f"{f.fold.index:>4} {m.n:>6} "
            f"{m.annualized_sharpe:>8.3f} {m.max_drawdown:>8.2%} "
            f"{m.total_return:>10.2%} {m.hit_rate:>8.2%}"
        )
    print()
    print(f"folds          : {result.n_folds}")
    print(f"mean sharpe    : {result.mean_sharpe:+.3f}")
    print(f"negative folds : {result.n_negative_folds}/{result.n_folds}")


async def _run(args) -> int:
    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)
    strategy = STRATEGIES.get(args.strategy)
    if strategy is None:
        print(f"[fatal] unknown strategy {args.strategy!r}; try: {', '.join(STRATEGIES)}")
        return 2

    start = _parse_date(args.start)
    end = _parse_date(args.end)
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
    result = run_backtest(
        strategy=strategy, strategy_name=args.strategy,
        bars=bars, splitter=WalkForwardSplitter(spec),
        symbol=args.symbol, timeframe=args.timeframe,
    )
    _print_report(result)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest CLI")
    parser.add_argument("--symbol", required=True, help="e.g. BTC/USD")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--timeframe", default="1Day", choices=["1Day", "1Hour", "1Min"])
    parser.add_argument("--strategy", default="buy_and_hold", choices=list(STRATEGIES))
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
