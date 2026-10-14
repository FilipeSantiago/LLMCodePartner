"""Lossless-by-check compression of coder tool results.

`ContextCompressionService` sends text Code Partner already authorized to the
context-compress output pipeline (`infra/compress-bridge`) and accepts the
result only when nothing the coder needs was lost. Every rejection, timeout or
backend failure returns the original text unchanged.

The guards are a minimum check, not proof that a compressed result is still
useful: error output is therefore limited to `conservative` (ANSI removal),
and correctness is judged on task outcomes (docs/context-compression.md).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("codepartner.compression")

MODES = ("conservative", "balanced")
CATEGORIES = frozenset({"search", "listing", "command", "other"})
_BRIDGE_DIR = Path(__file__).resolve().parent.parent / "infra" / "compress-bridge"
_MEMO_ENTRIES = 256

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# Diagnostic lines, not identifiers that merely contain "error" (`ErrorEvent`,
# `def tool_failed`): an exception name followed by ":" or line end, test-runner
# failure markers, compiler/npm error prefixes, and non-zero exit codes.
_ERROR_LINE = re.compile(
    r"Traceback \(most recent call last\)|\b\w*(?:Error|Exception|Exit)\b(?::|\s*$)|"
    r"\bFAILED\b|\bFAIL:|\bERROR\b|^E {3}|\bpanicked\b|npm ERR!|"
    r"(?:^|\d: |\): )(?i:error)(?:\[\w+\])?[: ]|(?i:exit[_ ]?code)\D{0,8}[1-9]", re.M)
_IDENTIFIERS = (
    re.compile(r"[\w./-]+\.py::[\w:\[\].,=-]+"),                  # pytest node ids
    re.compile(r"[\w./-]+\.[A-Za-z]\w{0,9}:\d+(?::\d+)?"),        # path:line[:col]
    re.compile(r"[\w./-]*[\w-]+\.(?:py|pyi|ts|tsx|js|jsx|mjs|json|ya?ml|toml|md|"
               r"rs|go|java|kt|cs|rb|php|c|h|cpp|hpp|sh|sql|html|css|cfg|ini|txt)\b"),
    re.compile(r'File "[^"]+", line \d+'),                        # traceback frames
    re.compile(r"\bcall_[0-9a-f]{8,}\b"),
    re.compile(r"\bobs-\d+\b"),
    re.compile(r"\b(?:commit|sha|id)[=: ]+[0-9a-f]{7,64}\b", re.I),
)
# Upstream marks omissions with hints for its own MCP tools, which Code Partner
# does not expose. Point the coder at the authorized retrieval tool instead.
_UPSTREAM_HINT = re.compile(r"use execute\(intent:[^\]\n]*|use search\(\)[^\]\n]*")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def estimate_tokens(text: str) -> int:
    """Deterministic, provider-neutral estimate (≈4 UTF-8 bytes per token)."""
    return math.ceil(len(text.encode("utf-8")) / 4)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identifiers(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in _IDENTIFIERS:
        found.update(pattern.findall(text))
    return found


def error_lines(text: str) -> set[str]:
    return {line.rstrip() for line in text.splitlines() if _ERROR_LINE.search(line)}


def is_error_output(text: str) -> bool:
    return bool(_ERROR_LINE.search(text))


def _json_value(text: str):
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except ValueError:
        return None


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CompressionConfig:
    enabled: bool = False
    backend: str = "noop"
    socket: str = "/run/compress-bridge/compress.sock"
    bridge_dir: str = str(_BRIDGE_DIR)
    mode: str = "conservative"
    min_bytes: int = 4096
    timeout_ms: int = 1500
    categories: frozenset[str] = CATEGORIES
    min_saving_ratio: float = 0.05
    persist: bool = False
    persist_days: int = 7

    @classmethod
    def from_env(cls) -> "CompressionConfig":
        mode = os.getenv("CODEPARTNER_COMPRESS_MODE", "conservative").strip().lower()
        backend = os.getenv("CODEPARTNER_COMPRESS_BACKEND", "sidecar").strip().lower()
        enabled = _env_bool("CODEPARTNER_COMPRESS_ENABLED", False)
        if mode not in MODES:
            # `auto` would send samples to an LLM; `aggressive` drops identifiers.
            log.error("compression disabled: CODEPARTNER_COMPRESS_MODE=%r is not one of %s",
                      mode, MODES)
            enabled = False
        if backend not in {"sidecar", "local", "noop"}:
            log.error("compression disabled: unknown CODEPARTNER_COMPRESS_BACKEND=%r", backend)
            enabled = False
        categories = frozenset(
            item.strip() for item in os.getenv(
                "CODEPARTNER_COMPRESS_CATEGORIES", ",".join(sorted(CATEGORIES))).split(",")
            if item.strip()) & CATEGORIES
        return cls(
            enabled=enabled, backend=backend,
            socket=os.getenv("CODEPARTNER_COMPRESS_SOCKET", cls.socket),
            bridge_dir=os.getenv("CODEPARTNER_COMPRESS_BRIDGE_DIR", cls.bridge_dir),
            mode=mode if mode in MODES else "conservative",
            min_bytes=int(os.getenv("CODEPARTNER_COMPRESS_MIN_BYTES", cls.min_bytes)),
            timeout_ms=int(os.getenv("CODEPARTNER_COMPRESS_TIMEOUT_MS", cls.timeout_ms)),
            categories=categories,
            min_saving_ratio=float(os.getenv("CODEPARTNER_COMPRESS_MIN_SAVING", cls.min_saving_ratio)),
            persist=_env_bool("CODEPARTNER_COMPRESS_PERSIST", False),
            persist_days=int(os.getenv("CODEPARTNER_COMPRESS_PERSIST_DAYS", cls.persist_days)),
        )

    def fingerprint(self) -> str:
        """Every setting that can change compressed bytes, for the memo key."""
        return json.dumps({"backend": self.backend, "min_saving_ratio": self.min_saving_ratio},
                          sort_keys=True)


@dataclass
class CompressionResult:
    text: str
    applied: bool
    reason: str
    mode: str | None = None
    version: str | None = None
    bytes_in: int = 0
    bytes_out: int = 0
    est_tokens_in: int = 0
    est_tokens_out: int = 0
    content_sha256: str = ""

    def as_record(self) -> dict:
        record = asdict(self)
        record.pop("text")
        return record


class CompressorUnavailable(RuntimeError):
    """The backend could not produce a result; the caller keeps the original."""


class NoopBackend:
    name = "noop"

    async def version(self) -> str:
        return "noop"

    async def compress(self, mode: str, text: str) -> str:
        return text


class _JsonLineBackend:
    """One request at a time over a newline-delimited JSON stream."""

    name = "jsonl"

    def __init__(self, timeout_ms: int):
        self.timeout = timeout_ms / 1000
        self._lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._version: str | None = None
        self._next_id = 0

    async def _open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        raise NotImplementedError

    async def _close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._reader = self._writer = None

    async def _request(self, payload: dict) -> dict:
        async with self._lock:
            try:
                if self._writer is None or self._writer.is_closing():
                    self._reader, self._writer = await asyncio.wait_for(self._open(), self.timeout)
                self._next_id += 1
                payload = {**payload, "id": self._next_id}
                self._writer.write((json.dumps(payload) + "\n").encode())
                await asyncio.wait_for(self._writer.drain(), self.timeout)
                line = await asyncio.wait_for(self._reader.readline(), self.timeout)
            except (OSError, asyncio.TimeoutError, ValueError) as exc:
                await self._close()
                raise CompressorUnavailable(f"{type(exc).__name__}: {exc}") from exc
            if not line:
                await self._close()
                raise CompressorUnavailable("compressor closed the connection")
            try:
                response = json.loads(line)
            except ValueError as exc:
                await self._close()
                raise CompressorUnavailable("malformed compressor response") from exc
            if response.get("id") != payload["id"] or not response.get("ok"):
                raise CompressorUnavailable(str(response.get("error") or "mismatched response"))
            return response

    async def version(self) -> str:
        if self._version is None:
            self._version = str((await self._request({"op": "version"}))["version"])
        return self._version

    async def compress(self, mode: str, text: str) -> str:
        response = await self._request({"op": "compress", "mode": mode, "text": text})
        if response.get("version") != self._version:
            # The sidecar was replaced; memo keys must not mix versions. The
            # next call re-reads the version.
            self._version = None
            raise CompressorUnavailable("compressor version changed")
        return str(response["text"])


class SidecarBackend(_JsonLineBackend):
    name = "sidecar"

    def __init__(self, socket_path: str, timeout_ms: int):
        super().__init__(timeout_ms)
        self.socket_path = socket_path

    async def _open(self):
        return await asyncio.open_unix_connection(self.socket_path, limit=32 * 1024 * 1024)


class LocalBackend(_JsonLineBackend):
    """Runs bridge.mjs as a child process under Node's permission model."""

    name = "local"

    def __init__(self, bridge_dir: str, timeout_ms: int):
        super().__init__(timeout_ms)
        self.bridge_dir = bridge_dir
        self._process: asyncio.subprocess.Process | None = None

    async def _open(self):
        bridge = Path(self.bridge_dir)
        if not (bridge / "node_modules" / "context-compress").is_dir():
            raise OSError(f"run `npm ci --ignore-scripts` in {bridge}")
        self._process = await asyncio.create_subprocess_exec(
            "node", "--permission", f"--allow-fs-read={bridge}", str(bridge / "bridge.mjs"),
            "--stdio", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=32 * 1024 * 1024,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, cwd=str(bridge))
        return self._process.stdout, self._process.stdin

    async def _close(self):
        await super()._close()
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()


def make_backend(config: CompressionConfig):
    if config.backend == "sidecar":
        return SidecarBackend(config.socket, config.timeout_ms)
    if config.backend == "local":
        return LocalBackend(config.bridge_dir, config.timeout_ms)
    return NoopBackend()


@dataclass
class ContextCompressionService:
    config: CompressionConfig
    backend: object = None
    _memo: OrderedDict = field(default_factory=OrderedDict)

    def __post_init__(self):
        if self.backend is None:
            self.backend = make_backend(self.config)

    def eligible(self, category: str) -> bool:
        return self.config.enabled and category in self.config.categories

    def unchanged(self, text: str, reason: str, mode: str | None = None,
                   version: str | None = None) -> CompressionResult:
        size = len(text.encode("utf-8"))
        tokens = estimate_tokens(text)
        return CompressionResult(text, False, reason, mode, version, size, size,
                                 tokens, tokens, sha256(text))

    def choose_mode(self, text: str) -> str:
        return "conservative" if is_error_output(strip_ansi(text)) else self.config.mode

    async def compress(self, category: str, text: str, *, failed: bool = False) -> CompressionResult:
        if not self.eligible(category):
            return self.unchanged(text, "not_eligible")
        if failed:
            return self.unchanged(text, "tool_failed")
        if len(text.encode("utf-8")) < self.config.min_bytes:
            return self.unchanged(text, "below_threshold")
        mode = self.choose_mode(text)
        try:
            version = await self.backend.version()
        except Exception as exc:  # any backend fault fails open
            log.warning("compression unavailable: %s", exc)
            return self.unchanged(text, "backend_error", mode)
        key = "|".join((sha256(text), getattr(self.backend, "name", "?"), version, mode,
                        "", category, self.config.fingerprint()))
        cached = self._memo.get(key)
        if cached is not None:
            self._memo.move_to_end(key)
            return cached
        try:
            candidate = await self.backend.compress(mode, text)
        except Exception as exc:  # any backend fault fails open
            log.warning("compression failed open: %s", exc)
            return self.unchanged(text, "backend_error", mode, version)
        candidate = _UPSTREAM_HINT.sub("use codepartner_get_observation for the original", candidate)
        reason = self.rejection(text, candidate)
        result = (self.unchanged(text, reason, mode, version) if reason else CompressionResult(
            candidate, True, "applied", mode, version, len(text.encode("utf-8")),
            len(candidate.encode("utf-8")), estimate_tokens(text), estimate_tokens(candidate),
            sha256(text)))
        self._memo[key] = result
        if len(self._memo) > _MEMO_ENTRIES:
            self._memo.popitem(last=False)
        return result

    def rejection(self, original: str, candidate: str) -> str | None:
        """Why `candidate` may not replace `original`, or None when it may."""
        if not candidate.strip():
            return "empty_output"
        plain = strip_ansi(original)
        value = _json_value(plain)
        if value is not None:
            if _json_value(candidate) != value:
                return "json_changed"
        else:
            if identifiers(plain) - identifiers(candidate):
                return "identifier_lost"
            kept = {line.rstrip() for line in candidate.splitlines()}
            if error_lines(plain) - kept:
                return "error_line_lost"
        if len(candidate.encode("utf-8")) > len(original.encode("utf-8")) * (1 - self.config.min_saving_ratio):
            return "insufficient_saving"
        return None


_shared: ContextCompressionService | None = None


def shared_service() -> ContextCompressionService | None:
    """Process-wide service (one backend connection, one memo), or None when off."""
    global _shared
    config = CompressionConfig.from_env()
    if not config.enabled:
        return None
    if _shared is None or _shared.config != config:
        _shared = ContextCompressionService(config)
    return _shared
