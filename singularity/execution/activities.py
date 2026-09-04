"""Activities sync — pulls CFEE (crypto fee) records from Alpaca and persists them.

CFEE records arrive T+1 (day after fill). Recommended cadence: run this via
cron once a day, well after midnight UTC.

CLI:
    uv run activities-sync              # pull since last saved date
    uv run activities-sync --days 7     # pull last 7 days regardless

The matching of CFEE records → fills (for calibration) happens in
`singularity.costs.calibration`, not here — this module is the ingestion side.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


async def sync_activities(
    rest: AlpacaRestClient,
    store: StateStore,
    since: datetime | None,
    activity_types: str = "CFEE,FILL",
) -> int:
    """Pull activities and persist them. Returns count of new rows written."""
    after = since.isoformat() if since else None
    activities = await rest.get_activities(activity_types=activity_types, after=after)
    for a in activities:
        await asyncio.to_thread(store.save_activity, a)
    log.info("activities_synced", count=len(activities), since=after)
    return len(activities)


async def _run(days: int | None) -> int:
    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)
    store = StateStore(Path(settings.state_db_path))

    if days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=days)
    else:
        last = store.latest_activity_date("CFEE")
        if last:
            since = datetime.fromisoformat(last.replace("Z", "+00:00") if last.endswith("Z") else last)
        else:
            since = datetime.now(timezone.utc) - timedelta(days=7)

    async with AlpacaRestClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_trading_url,
    ) as rest:
        n = await sync_activities(rest, store, since)
    print(f"synced {n} activity records since {since.isoformat()}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull CFEE/FILL activities from Alpaca")
    parser.add_argument("--days", type=int, default=None,
                        help="Lookback in days (default: since last saved activity)")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.days)))


if __name__ == "__main__":
    main()
