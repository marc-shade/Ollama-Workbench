"""
MCP tool discovery and execution.

Discovers MCP (Model Context Protocol) servers from the configuration files
of the coding CLIs installed on this system:

- Claude Code:  ~/.claude.json  (top-level "mcpServers", plus the current
                project's entry under "projects") and ./.mcp.json
                (project-scoped, git-shared)
- Codex CLI:    ~/.codex/config.toml  ([mcp_servers.<name>] tables)
- Gemini CLI:   ~/.gemini/settings.json  ("mcpServers")

For each discovered server it speaks real MCP: JSON-RPC 2.0 over stdio
(newline-delimited) or streamable HTTP - initialize, tools/list, tools/call.
No SDK dependency; the protocol subset used here is small.

Public interface (consumed by ui/tool_playground.py):
    get_available_mcp_tools() -> (tools_dict, schemas_list)
    execute_mcp_tool(tool_name, tools_dict, arguments) -> str (raises on error)
    get_mcp_discovery_report() -> list of per-server status dicts
"""

import json
import logging
import os
import re
import subprocess
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "ollama-workbench", "version": "1.0"}

# Per-server budget for spawn + initialize + tools/list. npx-based servers
# can cold-start slowly the first time.
SERVER_TIMEOUT_SECONDS = 20
# Budget for a single tools/call.
CALL_TIMEOUT_SECONDS = 60

# Written by the last get_available_mcp_tools() run; read by the UI.
_last_discovery_report: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def _expand(value: Any) -> Any:
    """Expand ~ and $VARS in strings, recursively for lists/dicts."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _normalize_server(name: str, entry: Any, source: str) -> Optional[Dict[str, Any]]:
    """Normalize one config entry to a common server description."""
    if not isinstance(entry, dict):
        return None
    if entry.get("disabled") is True or entry.get("enabled") is False:
        return None

    entry = _expand(entry)
    transport = entry.get("type") or entry.get("transport")
    url = entry.get("url") or entry.get("httpUrl") or entry.get("serverUrl")
    command = entry.get("command")

    if not transport:
        transport = "stdio" if command else ("http" if url else None)
    if transport in ("streamable-http", "streamable_http"):
        transport = "http"
    if transport not in ("stdio", "http", "sse"):
        return None
    if transport == "stdio" and not command:
        return None
    if transport in ("http", "sse") and not url:
        return None

    return {
        "name": name,
        "source": source,
        "transport": transport,
        "command": command,
        "args": entry.get("args") or [],
        "env": entry.get("env") or {},
        "cwd": entry.get("cwd"),
        "url": url,
        "headers": entry.get("headers") or {},
    }


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Could not parse {path}: {e}")
        return None


def _claude_servers(home: Path, cwd: Path) -> List[Dict[str, Any]]:
    servers: List[Dict[str, Any]] = []

    # Project-scoped, git-shared config
    project_cfg = _read_json(cwd / ".mcp.json")
    if project_cfg:
        for name, entry in (project_cfg.get("mcpServers") or {}).items():
            s = _normalize_server(name, entry, "claude:.mcp.json")
            if s:
                servers.append(s)

    cfg = _read_json(home / ".claude.json")
    if cfg:
        for name, entry in (cfg.get("mcpServers") or {}).items():
            s = _normalize_server(name, entry, "claude")
            if s:
                servers.append(s)
        # Per-project servers registered for the current directory
        project = (cfg.get("projects") or {}).get(str(cwd)) or {}
        for name, entry in (project.get("mcpServers") or {}).items():
            s = _normalize_server(name, entry, "claude:project")
            if s:
                servers.append(s)
    return servers


def _codex_servers(home: Path) -> List[Dict[str, Any]]:
    path = home / ".codex" / "config.toml"
    if not path.exists():
        return []
    try:
        import tomllib
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Could not parse {path}: {e}")
        return []

    servers = []
    for name, entry in (cfg.get("mcp_servers") or {}).items():
        s = _normalize_server(name, entry, "codex")
        if s:
            servers.append(s)
    return servers


def _gemini_servers(home: Path) -> List[Dict[str, Any]]:
    cfg = _read_json(home / ".gemini" / "settings.json")
    if not cfg:
        return []
    servers = []
    for name, entry in (cfg.get("mcpServers") or {}).items():
        s = _normalize_server(name, entry, "gemini")
        if s:
            servers.append(s)
    return servers


def discover_mcp_servers(home: Optional[Path] = None, cwd: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Discover MCP servers from Claude Code, Codex, and Gemini configs.

    On duplicate names the first source wins, in this precedence order:
    project .mcp.json, Claude global, Claude per-project, Codex, Gemini.
    """
    home = home or Path.home()
    cwd = cwd or Path.cwd()

    merged: Dict[str, Dict[str, Any]] = {}
    for server in _claude_servers(home, cwd) + _codex_servers(home) + _gemini_servers(home):
        if server["name"] not in merged:
            merged[server["name"]] = server
        else:
            logger.debug(
                f"MCP server '{server['name']}' from {server['source']} shadowed by "
                f"{merged[server['name']]['source']}"
            )
    return list(merged.values())


# ---------------------------------------------------------------------------
# MCP protocol clients
# ---------------------------------------------------------------------------

class MCPError(RuntimeError):
    """Raised when an MCP server cannot be reached or returns an error."""


class _StdioClient:
    """Minimal MCP client over stdio (newline-delimited JSON-RPC 2.0)."""

    def __init__(self, server: Dict[str, Any]):
        self.server = server
        self.proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._next_id = 0

    def __enter__(self):
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (self.server.get("env") or {}).items()})
        cmd = [self.server["command"]] + [str(a) for a in self.server.get("args") or []]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                cwd=self.server.get("cwd") or None,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise MCPError(f"command not found: {self.server['command']}")
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def _reader(self):
        try:
            assert self.proc and self.proc.stdout
            for line in self.proc.stdout:
                self._lines.put(line)
        except Exception as e:
            logger.debug(f"MCP stdio reader for '{self.server['name']}' ended: {e}")
        finally:
            # Sentinel so request() can distinguish "server exited" from timeout
            self._lines.put(None)

    def __exit__(self, *exc):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception as e:
                logger.debug(f"Terminating MCP server '{self.server['name']}' escalated to kill: {e}")
                try:
                    self.proc.kill()
                except Exception as kill_error:
                    logger.debug(f"Kill failed for MCP server '{self.server['name']}': {kill_error}")

    def _send(self, message: Dict[str, Any]):
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: float = SERVER_TIMEOUT_SECONDS) -> Dict[str, Any]:
        self._next_id += 1
        req_id = self._next_id
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

        import time
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"timeout waiting for {method} response")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise MCPError(f"timeout waiting for {method} response")
            if line is None:
                raise MCPError(f"server exited during {method}")
            line = line.strip()
            if not line:
                continue
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-protocol output on stdout
            if reply.get("id") != req_id:
                continue  # notification or unrelated message
            if "error" in reply:
                raise MCPError(f"{method}: {reply['error'].get('message', reply['error'])}")
            return reply.get("result", {})

    def notify(self, method: str):
        self._send({"jsonrpc": "2.0", "method": method})

    def initialize(self):
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments},
                            timeout=CALL_TIMEOUT_SECONDS)


class _HttpClient:
    """Minimal MCP client over streamable HTTP."""

    def __init__(self, server: Dict[str, Any]):
        self.server = server
        self.session_id: Optional[str] = None
        self._next_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Stateless per-request HTTP client - no connection to tear down
        return None

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: float = SERVER_TIMEOUT_SECONDS) -> Dict[str, Any]:
        import requests
        self._next_id += 1
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            **(self.server.get("headers") or {}),
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        resp = requests.post(self.server["url"], json=msg, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise MCPError(f"{method}: HTTP {resp.status_code}")
        if "Mcp-Session-Id" in resp.headers:
            self.session_id = resp.headers["Mcp-Session-Id"]

        body = resp.text
        if resp.headers.get("Content-Type", "").startswith("text/event-stream"):
            # Take the last data: payload in the stream
            payloads = [ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")]
            if not payloads:
                raise MCPError(f"{method}: empty SSE response")
            body = payloads[-1]
        reply = json.loads(body)
        if "error" in reply:
            raise MCPError(f"{method}: {reply['error'].get('message', reply['error'])}")
        return reply.get("result", {})

    def notify(self, method: str):
        try:
            import requests
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": PROTOCOL_VERSION,
                **(self.server.get("headers") or {}),
            }
            if self.session_id:
                headers["Mcp-Session-Id"] = self.session_id
            requests.post(self.server["url"],
                          json={"jsonrpc": "2.0", "method": method},
                          headers=headers, timeout=10)
        except Exception as e:
            # Notifications are fire-and-forget per the MCP spec
            logger.debug(f"MCP notify {method} to '{self.server['name']}' failed: {e}")

    def initialize(self):
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments},
                            timeout=CALL_TIMEOUT_SECONDS)


class _SseClient:
    """MCP client for the HTTP+SSE transport.

    The server keeps an SSE stream open on GET <url>; its first event is
    'endpoint' with the POST target for JSON-RPC messages. Responses to
    posted requests arrive as 'message' events on the SSE stream.
    """

    def __init__(self, server: Dict[str, Any]):
        self.server = server
        self.endpoint: Optional[str] = None
        self._events: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._conn = None
        self._stream = None
        self._next_id = 0

    def __enter__(self):
        # http.client instead of requests: it exposes the raw socket, which
        # __exit__ must shut down to unblock the reader thread (closing a
        # requests stream deadlocks on the buffered-reader lock the blocked
        # recv() holds), and its readline() delivers SSE lines unbuffered.
        import http.client
        from urllib.parse import urlparse, urljoin

        parsed = urlparse(self.server["url"])
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self._conn = conn_cls(parsed.hostname, parsed.port, timeout=SERVER_TIMEOUT_SECONDS)
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        headers = {"Accept": "text/event-stream", **(self.server.get("headers") or {})}
        try:
            self._conn.request("GET", path or "/", headers=headers)
            self._stream = self._conn.getresponse()
        except (ConnectionError, OSError) as e:
            raise MCPError(f"SSE connect: {e}")
        if self._stream.status >= 400:
            raise MCPError(f"SSE connect: HTTP {self._stream.status}")
        # The stream idles between events; only connect/handshake should be
        # bounded by the socket timeout. request() enforces its own deadlines.
        if self._conn.sock is not None:
            self._conn.sock.settimeout(None)

        threading.Thread(target=self._reader, daemon=True).start()

        # First event must announce the message endpoint
        import time
        deadline = time.monotonic() + SERVER_TIMEOUT_SECONDS
        while self.endpoint is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError("SSE connect: no endpoint event received")
            try:
                event = self._events.get(timeout=remaining)
            except queue.Empty:
                raise MCPError("SSE connect: no endpoint event received")
            if event is None:
                raise MCPError("SSE connect: stream closed before endpoint event")
            if event.get("event") == "endpoint":
                self.endpoint = urljoin(self.server["url"], event.get("data", ""))
        return self

    def _reader(self):
        try:
            event_type = "message"
            data_lines: List[str] = []
            assert self._stream is not None
            while True:
                raw_bytes = self._stream.readline()
                if not raw_bytes:
                    break
                raw = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                if raw == "":
                    if data_lines:
                        self._events.put({"event": event_type, "data": "\n".join(data_lines)})
                    event_type = "message"
                    data_lines = []
                elif raw.startswith("event:"):
                    event_type = raw[6:].strip()
                elif raw.startswith("data:"):
                    data_lines.append(raw[5:].strip())
        except Exception as e:
            logger.debug(f"SSE reader for '{self.server['name']}' ended: {e}")
        finally:
            self._events.put(None)

    def __exit__(self, *exc):
        if self._conn is None:
            return
        import socket as socket_mod
        try:
            if self._conn.sock is not None:
                self._conn.sock.shutdown(socket_mod.SHUT_RDWR)
        except OSError as e:
            logger.debug(f"SSE socket shutdown for '{self.server['name']}': {e}")
        try:
            self._conn.close()
        except Exception as e:
            logger.debug(f"Closing SSE connection for '{self.server['name']}': {e}")

    def _post(self, message: Dict[str, Any]):
        import requests
        assert self.endpoint
        headers = {"Content-Type": "application/json", **(self.server.get("headers") or {})}
        resp = requests.post(self.endpoint, json=message, headers=headers, timeout=10)
        if resp.status_code >= 400:
            raise MCPError(f"POST {message.get('method')}: HTTP {resp.status_code}")

    def request(self, method: str, params: Optional[Dict[str, Any]] = None,
                timeout: float = SERVER_TIMEOUT_SECONDS) -> Dict[str, Any]:
        self._next_id += 1
        req_id = self._next_id
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._post(msg)

        import time
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"timeout waiting for {method} response")
            try:
                event = self._events.get(timeout=remaining)
            except queue.Empty:
                raise MCPError(f"timeout waiting for {method} response")
            if event is None:
                raise MCPError(f"SSE stream closed during {method}")
            if event.get("event") != "message":
                continue
            try:
                reply = json.loads(event["data"])
            except json.JSONDecodeError:
                continue
            if reply.get("id") != req_id:
                continue
            if "error" in reply:
                raise MCPError(f"{method}: {reply['error'].get('message', reply['error'])}")
            return reply.get("result", {})

    def notify(self, method: str):
        try:
            self._post({"jsonrpc": "2.0", "method": method})
        except Exception as e:
            logger.debug(f"MCP notify {method} to '{self.server['name']}' failed: {e}")

    def initialize(self):
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments},
                            timeout=CALL_TIMEOUT_SECONDS)


def _client_for(server: Dict[str, Any]):
    if server["transport"] == "stdio":
        return _StdioClient(server)
    if server["transport"] == "http":
        return _HttpClient(server)
    if server["transport"] == "sse":
        return _SseClient(server)
    raise MCPError(f"transport '{server['transport']}' not supported")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _list_server_tools(server: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with _client_for(server) as client:
        client.initialize()
        return server, client.list_tools()


def get_available_mcp_tools() -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Discover MCP servers and list their tools.

    Returns (tools_dict, schemas_list). tools_dict maps a unique sanitized
    tool key to metadata; schemas_list holds OpenAI-style function schemas
    whose names are "mcp__<tool key>" (the format tool_playground feeds to
    the model).
    """
    global _last_discovery_report
    servers = discover_mcp_servers()
    tools: Dict[str, Dict[str, Any]] = {}
    schemas: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []

    if not servers:
        _last_discovery_report = []
        return tools, schemas

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_list_server_tools, s): s for s in servers}
        for future in as_completed(futures):
            server = futures[future]
            try:
                _, server_tools = future.result()
            except Exception as e:
                logger.warning(f"MCP server '{server['name']}' ({server['source']}): {e}")
                report.append({"name": server["name"], "source": server["source"],
                               "transport": server["transport"], "tools": 0, "error": str(e)})
                continue

            count = 0
            for tool in server_tools:
                tool_name = tool.get("name")
                if not tool_name:
                    continue
                key = _sanitize(f"{server['name']}__{tool_name}")
                if key in tools:
                    continue
                description = (tool.get("description") or "").strip()
                input_schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
                tools[key] = {
                    # Keys the existing playground UI displays
                    "type": "mcp",
                    "path": f"{server['source']}:{server['name']}",
                    # Execution metadata
                    "server": server["name"],
                    "server_config": server,
                    "tool": tool_name,
                    "description": description,
                    "input_schema": input_schema,
                    "source": server["source"],
                }
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": f"mcp__{key}",
                        "description": description or f"{tool_name} (MCP server {server['name']})",
                        "parameters": input_schema,
                    },
                })
                count += 1
            report.append({"name": server["name"], "source": server["source"],
                           "transport": server["transport"], "tools": count, "error": None})

    _last_discovery_report = report
    return tools, schemas


def get_mcp_discovery_report() -> List[Dict[str, Any]]:
    """Per-server status of the last get_available_mcp_tools() run."""
    return list(_last_discovery_report)


def execute_mcp_tool(tool_name: str, mcp_tools: Dict[str, Dict[str, Any]],
                     arguments: Any) -> str:
    """Execute a discovered MCP tool and return its text result.

    Raises MCPError (or ValueError for bad input) on failure; the caller in
    tool_playground has an explicit error-display path for exceptions.
    """
    info = mcp_tools.get(tool_name)
    if info is None:
        raise ValueError(f"Unknown MCP tool: {tool_name}")

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Tool arguments are not valid JSON: {e}")
    if arguments is None:
        arguments = {}

    with _client_for(info["server_config"]) as client:
        client.initialize()
        result = client.call_tool(info["tool"], arguments)

    if result.get("isError"):
        texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        raise MCPError("; ".join(t for t in texts if t) or "tool reported an error")

    parts: List[str] = []
    for content in result.get("content", []):
        if content.get("type") == "text":
            parts.append(content.get("text", ""))
        elif content.get("type") == "resource":
            resource = content.get("resource", {})
            parts.append(resource.get("text") or f"[resource: {resource.get('uri', 'unknown')}]")
        else:
            parts.append(f"[{content.get('type', 'unknown')} content]")
    if not parts and "structuredContent" in result:
        parts.append(json.dumps(result["structuredContent"]))
    return "\n".join(parts) if parts else "(empty result)"
