# syntax=docker/dockerfile:1.7

ARG NODE_VERSION=22
FROM node:${NODE_VERSION}-bookworm-slim AS node-tools

ARG OPENSPEC_VERSION=1.14.0
ARG CODEX_VERSION=0.160.0
ARG CLAUDE_CODE_VERSION=2.1.287

RUN npm install --global --prefix /opt/node-tools \
      "@fission-ai/openspec@${OPENSPEC_VERSION}" \
      "@openai/codex@${CODEX_VERSION}" \
      "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && npm cache clean --force


FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HOME=/home/codepartner \
    OPENSPEC_BIN=/opt/node-tools/bin/openspec \
    DISABLE_AUTOUPDATER=1 \
    PATH=/app/.venv/bin:/opt/node-tools/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git libstdc++6 passwd ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node-tools /usr/local/bin/node /usr/local/bin/node
COPY --from=node-tools /opt/node-tools /opt/node-tools
COPY --from=ghcr.io/astral-sh/uv:0.12.20 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN groupadd --gid 10001 codepartner \
    && useradd --uid 10001 --gid codepartner --create-home codepartner \
    && chown -R codepartner:codepartner /app /home/codepartner

USER codepartner

RUN node --version \
    && openspec --version \
    && codex --version \
    && claude --version

EXPOSE 7777

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7777/v1/models', timeout=3)"]

CMD ["/app/.venv/bin/python", "main.py"]
