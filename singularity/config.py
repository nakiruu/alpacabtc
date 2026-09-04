"""Typed settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_crypto_stream_url: str = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
    alpaca_trading_url: str = "https://paper-api.alpaca.markets"

    influx_url: str = "http://localhost:47086"
    influx_token: str = ""
    influx_org: str = "singularity"
    influx_bucket_raw: str = "crypto_raw"
    influx_bucket_features: str = "crypto_features"

    stream_trades: str = "BTC/USD"
    stream_quotes: str = "BTC/USD,ETH/USD,ETH/BTC"
    stream_orderbooks: str = "BTC/USD,ETH/USD"

    book_feature_cadence_s: float = 1.0
    log_level: str = "INFO"

    state_db_path: str = "./state/singularity.db"
    executor_heartbeat_s: float = 15.0
    passive_t1_s: float = 10.0
    passive_t2_s: float = 60.0
    passive_t3_s: float = 180.0

    def trades_symbols(self) -> list[str]:
        return _split_csv(self.stream_trades)

    def quotes_symbols(self) -> list[str]:
        return _split_csv(self.stream_quotes)

    def orderbook_symbols(self) -> list[str]:
        return _split_csv(self.stream_orderbooks)


def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
