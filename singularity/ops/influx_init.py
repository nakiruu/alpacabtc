"""One-shot bucket / retention setup for InfluxDB.

Plan §2 retention:

    crypto_trade       symbol                                90d raw → 5y downsampled
    crypto_quote       symbol                                30d raw
    crypto_book        symbol, level                         14d raw (features 5y)
    book_features      symbol                                5y

Since a bucket has a single retention policy, we use two buckets:
    crypto_raw        30d  (trades + quotes + raw books all live here)
    crypto_features   5y   (book_features + any derived series)

Then downsample crypto_raw → crypto_features via scheduled Flux tasks
(added later — for Phase 0 we just create the buckets).

Run:
    uv run influx-init
"""

from __future__ import annotations

from datetime import timedelta

from influxdb_client import BucketRetentionRules, InfluxDBClient

from ..config import get_settings
from ..logs import configure as configure_logging
from ..logs import get_logger

log = get_logger(__name__)


def _seconds(td: timedelta) -> int:
    return int(td.total_seconds())


def ensure_bucket(client: InfluxDBClient, org_id: str, name: str, retention_days: int) -> None:
    buckets_api = client.buckets_api()
    existing = buckets_api.find_bucket_by_name(name)
    if existing:
        log.info("bucket_exists", name=name)
        return
    rules = [BucketRetentionRules(type="expire", every_seconds=_seconds(timedelta(days=retention_days)))]
    buckets_api.create_bucket(bucket_name=name, org_id=org_id, retention_rules=rules)
    log.info("bucket_created", name=name, retention_days=retention_days)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    with InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org) as client:
        org = client.organizations_api().find_organizations(org=settings.influx_org)
        if not org:
            raise SystemExit(f"org not found: {settings.influx_org}")
        org_id = org[0].id
        ensure_bucket(client, org_id, settings.influx_bucket_raw, retention_days=30)
        ensure_bucket(client, org_id, settings.influx_bucket_features, retention_days=1825)


if __name__ == "__main__":
    main()
