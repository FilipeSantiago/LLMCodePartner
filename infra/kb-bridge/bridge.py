"""Private Streamable HTTP proxy for a host-installed Codebase Memory binary."""

import asyncio
import logging
import os
import stat
import sys
from pathlib import Path

import httpx
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.server import create_proxy
from hypercorn.asyncio import serve
from hypercorn.config import Config
from starlette.responses import JSONResponse

BIN = Path("/usr/local/bin/codebase-memory-mcp")
DATA = Path("/var/lib/codebase-memory-mcp")
SOCKET_DIR = Path("/run/kb-bridge")
SOCKET = SOCKET_DIR / "mcp.sock"
REQUIRED = {"list_projects", "index_status", "get_architecture", "search_graph",
            "trace_path", "get_file_outline", "get_code_snippet"}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s kb-bridge %(message)s")
log = logging.getLogger("kb_bridge")


def validate_paths() -> tuple[int, int]:
    uid, gid = int(os.environ["KB_UID"]), int(os.environ["KB_GID"])
    if uid <= 0 or gid <= 0:
        raise RuntimeError("KB_UID and KB_GID must be positive host IDs")
    if not BIN.is_file() or not os.access(BIN, os.X_OK):
        raise RuntimeError(f"KB_BINARY_PATH is not an executable file: {BIN}")
    if not DATA.is_dir() or not any(DATA.glob("*.db")):
        raise RuntimeError("KB_DATA_PATH must be an existing indexed storage directory")
    if DATA.stat().st_uid != uid or DATA.stat().st_gid != gid:
        raise RuntimeError("KB_UID/KB_GID must match the storage directory owner and group")
    mode = DATA.stat().st_mode
    if not mode & stat.S_IWUSR:
        raise RuntimeError("Codebase Memory v0.11.0 needs writable KB storage")
    return uid, gid


def upstream() -> StdioTransport:
    return StdioTransport(
        command=str(BIN), args=["--tool-profile=scout"],
        env={"CBM_CACHE_DIR": str(DATA), "CBM_LOG_LEVEL": "info"},
    )


async def check_upstream() -> None:
    async with Client(upstream()) as client:
        names = {tool.name for tool in await client.list_tools()}
        missing = REQUIRED - names
        if missing:
            raise RuntimeError("Codebase Memory is missing tools: " + ", ".join(sorted(missing)))
        await client.call_tool("list_projects", {"format": "json", "limit": 1})


async def health() -> int:
    transport = httpx.AsyncHTTPTransport(uds=str(SOCKET))
    try:
        async with httpx.AsyncClient(transport=transport, timeout=6) as client:
            response = await client.get("http://kb-bridge/health/ready")
            return 0 if response.status_code == 200 else 1
    except Exception as exc:
        log.error("health transport error: %s", exc)
        return 1


async def run() -> None:
    uid, gid = validate_paths()
    version = await asyncio.create_subprocess_exec(
        str(BIN), "--version", stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await version.communicate()
    if version.returncode or stdout.strip() != b"codebase-memory-mcp 0.11.0":
        raise RuntimeError("KB_BINARY_PATH must provide Codebase Memory v0.11.0: " +
                           (stderr or stdout).decode(errors="replace"))

    SOCKET_DIR.mkdir(exist_ok=True)
    os.chown(SOCKET_DIR, uid, gid)
    os.chmod(SOCKET_DIR, 0o770)
    if SOCKET.exists():
        if not stat.S_ISSOCK(SOCKET.stat().st_mode):
            raise RuntimeError(f"refusing to replace non-socket at {SOCKET}")
        SOCKET.unlink()
    os.setgroups([gid])
    os.setgid(gid)
    os.setuid(uid)
    os.umask(0o007)

    await check_upstream()
    log.info("Codebase Memory stdio connection ready; serving HTTP at %s", SOCKET)
    proxy = create_proxy(upstream(), name="codebase-memory-bridge")

    @proxy.custom_route("/health/live", methods=["GET"])
    async def live(_request):
        return JSONResponse({"status": "alive"})

    @proxy.custom_route("/health/ready", methods=["GET"])
    async def ready(_request):
        try:
            await check_upstream()
            return JSONResponse({"status": "ready"})
        except Exception as exc:
            log.error("upstream readiness failed: %s", exc)
            return JSONResponse({"status": "unavailable"}, status_code=503)

    app = proxy.http_app(path="/mcp", stateless_http=True)
    config = Config()
    config.bind = [f"unix:{SOCKET}"]
    config.accesslog = "-"
    config.errorlog = "-"
    await serve(app, config)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--health":
        raise SystemExit(asyncio.run(health()))
    try:
        asyncio.run(run())
    except Exception:
        log.exception("bridge startup or transport failure")
        raise
