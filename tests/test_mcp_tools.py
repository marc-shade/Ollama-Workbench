"""
Tests for ollama_workbench.ui.mcp_tools - MCP server discovery from
Claude Code / Codex / Gemini configs and JSON-RPC stdio protocol handling.

The protocol tests run against a real subprocess speaking newline-delimited
JSON-RPC 2.0 (the MCP stdio transport), so the client path is exercised
end-to-end without any network or external MCP servers.
"""

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from ollama_workbench.ui.mcp_tools import (
    MCPError,
    _normalize_server,
    _sanitize,
    discover_mcp_servers,
    execute_mcp_tool,
    get_available_mcp_tools,
    get_mcp_discovery_report,
)


# ---------------------------------------------------------------------------
# Config discovery
# ---------------------------------------------------------------------------

def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


class TestConfigDiscovery:
    def test_claude_global_servers(self, tmp_path):
        _write(tmp_path / "home" / ".claude.json", """
            {"mcpServers": {"alpha": {"command": "alpha-server", "args": ["--x"],
                                       "env": {"KEY": "v"}}}}
        """)
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=tmp_path / "proj")
        assert len(servers) == 1
        s = servers[0]
        assert s["name"] == "alpha"
        assert s["source"] == "claude"
        assert s["transport"] == "stdio"
        assert s["command"] == "alpha-server"
        assert s["args"] == ["--x"]
        assert s["env"] == {"KEY": "v"}

    def test_project_mcp_json_discovered_and_wins_over_global(self, tmp_path):
        _write(tmp_path / "home" / ".claude.json", """
            {"mcpServers": {"dup": {"command": "global-version"}}}
        """)
        _write(tmp_path / "proj" / ".mcp.json", """
            {"mcpServers": {"dup": {"command": "project-version"}}}
        """)
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=tmp_path / "proj")
        assert len(servers) == 1
        assert servers[0]["command"] == "project-version"
        assert servers[0]["source"] == "claude:.mcp.json"

    def test_claude_per_project_servers_for_cwd_only(self, tmp_path):
        cwd = tmp_path / "proj"
        _write(tmp_path / "home" / ".claude.json", json.dumps({
            "projects": {
                str(cwd): {"mcpServers": {"projserv": {"command": "p"}}},
                "/somewhere/else": {"mcpServers": {"other": {"command": "o"}}},
            }
        }))
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=cwd)
        names = {s["name"] for s in servers}
        assert names == {"projserv"}
        assert servers[0]["source"] == "claude:project"

    def test_codex_toml_servers(self, tmp_path):
        _write(tmp_path / "home" / ".codex" / "config.toml", """
            model = "gpt-5"

            [mcp_servers.beta]
            command = "beta-server"
            args = ["serve"]

            [mcp_servers.beta.env]
            TOKEN = "t"
        """)
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=tmp_path / "proj")
        assert len(servers) == 1
        s = servers[0]
        assert (s["name"], s["source"], s["command"]) == ("beta", "codex", "beta-server")
        assert s["env"] == {"TOKEN": "t"}

    def test_gemini_settings_servers(self, tmp_path):
        _write(tmp_path / "home" / ".gemini" / "settings.json", """
            {"mcpServers": {"gamma": {"command": "gamma-server"},
                            "gweb": {"httpUrl": "http://localhost:9999/mcp"}}}
        """)
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=tmp_path / "proj")
        by_name = {s["name"]: s for s in servers}
        assert by_name["gamma"]["transport"] == "stdio"
        assert by_name["gweb"]["transport"] == "http"
        assert by_name["gweb"]["url"] == "http://localhost:9999/mcp"

    def test_all_three_sources_merge(self, tmp_path):
        _write(tmp_path / "home" / ".claude.json",
               '{"mcpServers": {"a": {"command": "a"}}}')
        _write(tmp_path / "home" / ".codex" / "config.toml",
               '[mcp_servers.b]\ncommand = "b"\n')
        _write(tmp_path / "home" / ".gemini" / "settings.json",
               '{"mcpServers": {"c": {"command": "c"}}}')
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=tmp_path / "proj")
        assert {s["name"] for s in servers} == {"a", "b", "c"}
        assert {s["source"] for s in servers} == {"claude", "codex", "gemini"}

    def test_disabled_entries_skipped(self, tmp_path):
        _write(tmp_path / "home" / ".claude.json", """
            {"mcpServers": {"off1": {"command": "x", "disabled": true},
                            "off2": {"command": "y", "enabled": false},
                            "on":   {"command": "z"}}}
        """)
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=tmp_path / "proj")
        assert {s["name"] for s in servers} == {"on"}

    def test_missing_configs_yield_empty_list(self, tmp_path):
        assert discover_mcp_servers(home=tmp_path / "nohome", cwd=tmp_path / "noproj") == []

    def test_malformed_config_survives(self, tmp_path):
        _write(tmp_path / "home" / ".claude.json", "{not json")
        _write(tmp_path / "home" / ".gemini" / "settings.json",
               '{"mcpServers": {"ok": {"command": "ok"}}}')
        servers = discover_mcp_servers(home=tmp_path / "home", cwd=tmp_path / "proj")
        assert {s["name"] for s in servers} == {"ok"}


class TestNormalization:
    def test_non_dict_entry_rejected(self):
        assert _normalize_server("x", "not-a-dict", "claude") is None

    def test_entry_without_command_or_url_rejected(self):
        assert _normalize_server("x", {"args": ["a"]}, "claude") is None

    def test_env_and_home_expansion(self, monkeypatch):
        monkeypatch.setenv("MCP_TEST_TOKEN", "sekrit")
        s = _normalize_server("x", {
            "command": "~/bin/server",
            "env": {"TOKEN": "$MCP_TEST_TOKEN"},
        }, "claude")
        assert s["command"].startswith(str(Path.home()))
        assert s["env"]["TOKEN"] == "sekrit"

    def test_streamable_http_alias(self):
        s = _normalize_server("x", {"type": "streamable-http", "url": "http://h/mcp"}, "claude")
        assert s["transport"] == "http"

    def test_sanitize(self):
        assert _sanitize("srv.name__tool.op") == "srv_name__tool_op"


# ---------------------------------------------------------------------------
# Protocol tests against a real fake MCP server subprocess
# ---------------------------------------------------------------------------

FAKE_SERVER = r'''
import json, sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    if "id" not in msg:
        continue  # notification
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
            {"name": "echo", "description": "Echo text back",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string"}},
                             "required": ["text"]}},
            {"name": "boom", "description": "Always fails",
             "inputSchema": {"type": "object", "properties": {}}}]}})
    elif method == "tools/call":
        name = msg["params"]["name"]
        if name == "echo":
            text = msg["params"].get("arguments", {}).get("text", "")
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": "echo: " + text}]}})
        else:
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "isError": True,
                "content": [{"type": "text", "text": "boom failed"}]}})
    else:
        send({"jsonrpc": "2.0", "id": msg["id"],
              "error": {"code": -32601, "message": "unknown method"}})
'''


@pytest.fixture()
def fake_server(tmp_path):
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(FAKE_SERVER)
    return {
        "name": "fake",
        "source": "claude",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(script)],
        "env": {},
        "cwd": None,
        "url": None,
        "headers": {},
    }


class TestProtocol:
    def test_list_tools_and_schemas(self, fake_server):
        with patch("ollama_workbench.ui.mcp_tools.discover_mcp_servers",
                   return_value=[fake_server]):
            tools, schemas = get_available_mcp_tools()

        assert set(tools) == {"fake__echo", "fake__boom"}
        info = tools["fake__echo"]
        assert info["type"] == "mcp"
        assert info["server"] == "fake"
        assert info["tool"] == "echo"
        assert info["description"] == "Echo text back"
        assert info["input_schema"]["required"] == ["text"]

        by_name = {s["function"]["name"]: s for s in schemas}
        assert set(by_name) == {"mcp__fake__echo", "mcp__fake__boom"}
        assert by_name["mcp__fake__echo"]["type"] == "function"
        assert by_name["mcp__fake__echo"]["function"]["parameters"]["required"] == ["text"]

        report = get_mcp_discovery_report()
        assert report == [{"name": "fake", "source": "claude", "transport": "stdio",
                           "tools": 2, "error": None}]

    def test_execute_tool_returns_text(self, fake_server):
        with patch("ollama_workbench.ui.mcp_tools.discover_mcp_servers",
                   return_value=[fake_server]):
            tools, _ = get_available_mcp_tools()
        result = execute_mcp_tool("fake__echo", tools, {"text": "hello"})
        assert result == "echo: hello"

    def test_execute_tool_accepts_json_string_arguments(self, fake_server):
        with patch("ollama_workbench.ui.mcp_tools.discover_mcp_servers",
                   return_value=[fake_server]):
            tools, _ = get_available_mcp_tools()
        result = execute_mcp_tool("fake__echo", tools, '{"text": "from json"}')
        assert result == "echo: from json"

    def test_execute_tool_error_result_raises(self, fake_server):
        with patch("ollama_workbench.ui.mcp_tools.discover_mcp_servers",
                   return_value=[fake_server]):
            tools, _ = get_available_mcp_tools()
        with pytest.raises(MCPError, match="boom failed"):
            execute_mcp_tool("fake__boom", tools, {})

    def test_execute_unknown_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown MCP tool"):
            execute_mcp_tool("nope", {}, {})

    def test_execute_bad_json_arguments_raise(self, fake_server):
        tools = {"fake__echo": {"server_config": fake_server, "tool": "echo"}}
        with pytest.raises(ValueError, match="not valid JSON"):
            execute_mcp_tool("fake__echo", tools, "{broken")

    def test_missing_command_reported_not_raised(self, tmp_path):
        broken = {
            "name": "ghost", "source": "codex", "transport": "stdio",
            "command": str(tmp_path / "does-not-exist"), "args": [],
            "env": {}, "cwd": None, "url": None, "headers": {},
        }
        with patch("ollama_workbench.ui.mcp_tools.discover_mcp_servers",
                   return_value=[broken]):
            tools, schemas = get_available_mcp_tools()
        assert tools == {} and schemas == []
        report = get_mcp_discovery_report()
        assert len(report) == 1
        assert report[0]["name"] == "ghost"
        assert report[0]["error"]

    def test_unknown_transport_reported(self):
        weird = {"name": "w", "source": "claude", "transport": "carrier-pigeon",
                 "command": None, "args": [], "env": {}, "cwd": None,
                 "url": "http://h", "headers": {}}
        with patch("ollama_workbench.ui.mcp_tools.discover_mcp_servers",
                   return_value=[weird]):
            tools, _ = get_available_mcp_tools()
        assert tools == {}
        assert "not supported" in get_mcp_discovery_report()[0]["error"]


# ---------------------------------------------------------------------------
# SSE transport against a real in-process HTTP server
# ---------------------------------------------------------------------------

class _FakeSseMcpServer:
    """Minimal HTTP+SSE MCP server: GET /sse streams events, POST /messages
    accepts JSON-RPC and answers over the SSE stream."""

    def __init__(self):
        import http.server
        import socketserver

        outer = self
        # One queue per SSE connection; the newest connection receives
        # responses (clients here connect sequentially).
        self.current_queue = None

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                my_queue = queue.Queue()
                outer.current_queue = my_queue
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"event: endpoint\ndata: /messages?session=1\n\n")
                self.wfile.flush()
                while True:
                    payload = my_queue.get()
                    if payload is None:
                        return
                    try:
                        self.wfile.write(
                            f"event: message\ndata: {payload}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                msg = json.loads(self.rfile.read(length))
                self.send_response(202)
                self.end_headers()
                if "id" not in msg:
                    return
                method = msg.get("method")
                if method == "initialize":
                    result = {"protocolVersion": "2025-06-18", "capabilities": {},
                              "serverInfo": {"name": "fake-sse", "version": "1.0"}}
                elif method == "tools/list":
                    result = {"tools": [{"name": "ping", "description": "Ping",
                                         "inputSchema": {"type": "object", "properties": {}}}]}
                elif method == "tools/call":
                    result = {"content": [{"type": "text", "text": "pong"}]}
                else:
                    result = {}
                if outer.current_queue is not None:
                    outer.current_queue.put(json.dumps(
                        {"jsonrpc": "2.0", "id": msg["id"], "result": result}))

        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        import threading
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        if self.current_queue is not None:
            self.current_queue.put(None)
        self.httpd.shutdown()


import queue  # noqa: E402  (used by the fake SSE server)


class TestSseTransport:
    def test_sse_list_and_call(self):
        server = _FakeSseMcpServer()
        try:
            cfg = {"name": "ssefake", "source": "claude", "transport": "sse",
                   "command": None, "args": [], "env": {}, "cwd": None,
                   "url": f"http://127.0.0.1:{server.port}/sse", "headers": {}}
            with patch("ollama_workbench.ui.mcp_tools.discover_mcp_servers",
                       return_value=[cfg]):
                tools, schemas = get_available_mcp_tools()
            assert set(tools) == {"ssefake__ping"}
            assert schemas[0]["function"]["name"] == "mcp__ssefake__ping"

            result = execute_mcp_tool("ssefake__ping", tools, {})
            assert result == "pong"
        finally:
            server.stop()
