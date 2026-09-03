FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml ./
COPY singularity/ ./singularity/

RUN uv pip install --system --no-cache .

RUN useradd -r -u 1000 -m singularity \
    && mkdir -p /app/state /app/logs \
    && chown -R singularity:singularity /app

USER singularity

ENV PYTHONUNBUFFERED=1

CMD ["capture"]
