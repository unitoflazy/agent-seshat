"""`sesh mcp`: zero-downtime stdio entrypoint.

1. Ping the warm HTTP daemon (``/health``, 50 ms timeout).
2. Alive  -> transparently pump newline-delimited JSON-RPC from stdin to the daemon's
   streamable-HTTP endpoint (stateless + JSON responses) and write replies to stdout.
   If the daemon dies mid-session, a local stdio server is spawned, the cached
   ``initialize`` handshake is replayed, and traffic continues without the client noticing.
3. Dead   -> run the SQLite + ONNX engine in-process over stdio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import httpx

from .config import Settings, get_settings

log = logging.getLogger(__name__)

PING_TIMEOUT_S = 0.05
REPLAY_INIT_ID = "seshat-proxy-replay-init"


def daemon_alive(base_url: str, timeout: float = PING_TIMEOUT_S) -> bool:
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def daemon_health(base_url: str, timeout: float = 0.5) -> dict | None:
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _parse_sse(text: str) -> list[dict]:
    out: list[dict] = []
    data: list[str] = []
    for line in text.splitlines() + [""]:
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line.strip() and data:
            try:
                payload = json.loads("\n".join(data))
                out.extend(payload if isinstance(payload, list) else [payload])
            except json.JSONDecodeError:
                pass
            data = []
    return out


class DaemonUnavailable(Exception):
    pass


class LocalFallback:
    """A local `sesh serve --transport stdio` child that takes over if the daemon disappears."""

    def __init__(self, write, project: str):
        self._write = write
        self._project = project
        self.proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None

    async def start(self, init_msg: dict | None, initialized: bool) -> None:
        env = dict(os.environ)
        if self._project:
            env.setdefault("AGENT_SESHAT_PROJECT", self._project)
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "agent_seshat", "serve", "--transport", "stdio",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, env=env,
            limit=16 * 1024 * 1024,
        )
        if init_msg is not None:
            replay = dict(init_msg, id=REPLAY_INIT_ID)
            await self._send_raw(replay)
            assert self.proc.stdout is not None
            while True:  # swallow the replayed initialize response
                line = await self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("local fallback server exited during initialize")
                try:
                    if json.loads(line).get("id") == REPLAY_INIT_ID:
                        break
                except json.JSONDecodeError:
                    continue
            if initialized:
                await self._send_raw({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._reader = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            await self._write(line.rstrip(b"\r\n"))

    async def _send_raw(self, msg: Any) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(msg).encode() + b"\n")
        await self.proc.stdin.drain()

    async def send(self, msg: Any) -> None:
        await self._send_raw(msg)

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), timeout=3)
            except Exception:
                self.proc.kill()


class StdioHttpProxy:
    def __init__(self, base_url: str, project: str = "", mcp_path: str = "/mcp"):
        self.endpoint = base_url.rstrip("/") + mcp_path
        self.project = project
        self.protocol_version: str | None = None
        self.init_msg: dict | None = None
        self.initialized = False
        self.fallback: LocalFallback | None = None
        self._fallback_lock = asyncio.Lock()
        self._out_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self.client: httpx.AsyncClient | None = None

    # -- io ---------------------------------------------------------------------------

    async def write(self, payload: bytes | dict | list) -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload, separators=(",", ":")).encode()
        async with self._out_lock:
            sys.stdout.buffer.write(data + b"\n")
            sys.stdout.buffer.flush()

    def inject_project(self, msg: dict) -> dict:
        """Clients spawn `sesh mcp` inside the workspace; tell the daemon which repo we're in."""
        if not self.project or msg.get("method") != "tools/call":
            return msg
        params = msg.get("params") or {}
        if params.get("name") != "search_skills":
            return msg
        args = dict(params.get("arguments") or {})
        if not args.get("project"):
            args["project"] = self.project
            msg = dict(msg, params=dict(params, arguments=args))
        return msg

    # -- transport --------------------------------------------------------------------

    async def post(self, msg: Any) -> list[dict]:
        assert self.client is not None
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        try:
            r = await self.client.post(self.endpoint, content=json.dumps(msg), headers=headers)
        except httpx.TransportError as exc:
            raise DaemonUnavailable(str(exc)) from exc
        if r.status_code in (502, 503, 504):
            raise DaemonUnavailable(f"HTTP {r.status_code}")
        if r.status_code == 202 or not r.content:
            return []
        ctype = r.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            return _parse_sse(r.text)
        try:
            payload = r.json()
        except json.JSONDecodeError:
            if isinstance(msg, dict) and "id" in msg:
                return [{
                    "jsonrpc": "2.0", "id": msg["id"],
                    "error": {"code": -32603, "message": f"agent-seshat daemon returned HTTP {r.status_code}"},
                }]
            return []
        return payload if isinstance(payload, list) else [payload]

    async def activate_fallback(self, replay_init: bool = True) -> LocalFallback:
        async with self._fallback_lock:
            if self.fallback is None:
                log.warning("agent-seshat daemon unreachable; switching to in-process stdio server")
                fb = LocalFallback(self.write, self.project)
                await fb.start(self.init_msg if replay_init else None, self.initialized)
                self.fallback = fb
            return self.fallback

    async def handle(self, msg: Any) -> None:
        is_init = isinstance(msg, dict) and msg.get("method") == "initialize"
        if isinstance(msg, dict):
            msg = self.inject_project(msg)
            if msg.get("method") == "notifications/initialized":
                self.initialized = True
        if self.fallback is not None:
            await self.fallback.send(msg)
            return
        try:
            responses = await self.post(msg)
        except DaemonUnavailable:
            # If the handshake itself failed, the child must answer it for real (no replay).
            fb = await self.activate_fallback(replay_init=not is_init)
            await fb.send(msg)
            if is_init:
                self.init_msg = msg
            return
        if is_init:
            self.init_msg = msg
        for resp in responses:
            if is_init and isinstance(resp, dict) and isinstance(resp.get("result"), dict):
                self.protocol_version = resp["result"].get("protocolVersion") or self.protocol_version
            await self.write(resp)

    async def run(self) -> None:
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=2.0))
        try:
            while True:
                line = await asyncio.to_thread(sys.stdin.buffer.readline)
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    await self.write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                    continue
                if isinstance(msg, dict) and msg.get("method") == "initialize":
                    await self.handle(msg)  # handshake must complete before anything else
                    continue
                task = asyncio.create_task(self.handle(msg))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            if self._tasks:
                await asyncio.wait(self._tasks, timeout=30)
        finally:
            await self.client.aclose()
            if self.fallback:
                await self.fallback.close()


def run_mcp_entrypoint(settings: Settings | None = None, force_local: bool = False) -> None:
    """Entry point for `sesh mcp`."""
    from .server import detect_project, run_server

    settings = settings or get_settings()
    if not force_local and os.environ.get("AGENT_SESHAT_NO_PROXY") != "1" and daemon_alive(settings.daemon_url):
        log.info("Proxying stdio to warm daemon at %s", settings.daemon_url)
        asyncio.run(StdioHttpProxy(settings.daemon_url, project=detect_project()).run())
        return
    run_server("stdio")
