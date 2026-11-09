"""Tests for v2.0 Centralized MCP Aggregator, Tool Indexing & Meta-Broker (`mcp_aggregator.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_seshat.config import Settings
from agent_seshat.indexer import SkillIndex
from agent_seshat.mcp_aggregator import (
    call_mcp_tool,
    consolidate_mcp_configs,
    harvest_mcp_configs,
    load_mcp_registry,
    search_mcp_tools,
    sync_mcp_tools,
)

MOCK_DOWNSTREAM_SERVER_CODE = '''"""Lightweight stdio MCP server used to test Agent-Seshat MCP Hub aggregation."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("deploy-pilot")

@mcp.tool()
def get_deployment_status(service_name: str, environment: str = "prod") -> str:
    """Check Kubernetes deployment rollout status and healthy replica counts for a microservice."""
    return f"service={service_name} env={environment} status=HEALTHY replicas=4/4"

@mcp.tool()
def trigger_canary_rollout(service_name: str, traffic_percent: int = 10) -> str:
    """Shift a percentage of production ingress traffic to the canary deployment."""
    return f"canary_started service={service_name} weight={traffic_percent}%"

if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def test_mcp_aggregator_harvest_dedup_sync_search_and_call(tmp_path: Path) -> None:
    seshat_home = tmp_path / ".agent-seshat"
    settings = Settings(home=seshat_home)

    fake_home = tmp_path / "user_home"
    fake_home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Write lightweight mock stdio MCP server script
    mock_server_script = tmp_path / "mock_deploy_mcp.py"
    mock_server_script.write_text(MOCK_DOWNSTREAM_SERVER_CODE, encoding="utf-8")

    # 1. Claude config (~/.claude.json): contains `deploy-pilot` AND a self-reference to `agent-seshat`
    claude_cfg = fake_home / ".claude.json"
    claude_cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "deploy-pilot": {
                        "command": sys.executable,
                        "args": [str(mock_server_script)],
                    },
                    "agent-seshat": {
                        "command": "sesh",
                        "args": ["mcp"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    # 2. Cursor config (~/.cursor/mcp.json): duplicates `deploy-pilot` and adds `docs-http`
    cursor_cfg = fake_home / ".cursor" / "mcp.json"
    cursor_cfg.parent.mkdir(parents=True)
    cursor_cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "deploy-pilot": {
                        "command": sys.executable,
                        "args": [str(mock_server_script)],
                    },
                    "docs-http": {
                        "url": "http://127.0.0.1:9999/mcp",
                        "disabled": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    # 3. Copilot / VS Code config (.vscode/mcp.json): also references `deploy-pilot`
    vscode_cfg = workspace / ".vscode" / "mcp.json"
    vscode_cfg.parent.mkdir(parents=True)
    vscode_cfg.write_text(
        json.dumps(
            {
                "servers": {
                    "deploy-pilot": {
                        "command": sys.executable,
                        "args": [str(mock_server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    # 4. OpenAI Codex config (~/.codex/config.toml): references `deploy-pilot` and `sesh`
    codex_cfg = fake_home / ".codex" / "config.toml"
    codex_cfg.parent.mkdir(parents=True)
    codex_cfg.write_text(
        f'[mcp_servers.deploy-pilot]\ncommand = "{sys.executable}"\nargs = ["{mock_server_script}"]\n\n'
        '[mcp_servers.agent-seshat]\ncommand = "sesh"\nargs = ["mcp"]\n',
        encoding="utf-8",
    )

    # Step 1: Harvest MCP configs across all 4 agents
    harvest_report = harvest_mcp_configs(
        workspace=workspace,
        user_home=fake_home,
        settings=settings,
    )

    assert harvest_report.skipped_self == 2
    registry = load_mcp_registry(settings)
    assert sorted(registry.keys()) == ["deploy-pilot", "docs-http"]

    deploy_entry = registry["deploy-pilot"]
    assert deploy_entry.transport == "stdio"
    assert set(deploy_entry.source_agents) == {"claude", "cursor", "copilot", "codex"}
    assert registry["docs-http"].enabled is False

    # Step 2: Sync downstream tool schemas from `deploy-pilot` into SQLite + ONNX
    idx = SkillIndex(settings)
    sync_results = sync_mcp_tools(idx, settings=settings)
    assert len(sync_results) == 1
    assert sync_results[0].server_name == "deploy-pilot"
    assert sync_results[0].error == ""
    assert sync_results[0].tools_count == 2

    # Step 3: Search upstream MCP tools using Hybrid Search (ONNX + BM25)
    search_output = search_mcp_tools(
        idx,
        query="Kubernetes deployment rollout status and healthy replicas",
        limit=5,
    )
    assert "deploy-pilot" in search_output
    assert "get_deployment_status" in search_output
    assert '"service_name"' in search_output

    # Step 4: Proxy `call_mcp_tool` to the downstream stdio MCP server
    call_output = call_mcp_tool(
        idx,
        server_name="deploy-pilot",
        tool_name="get_deployment_status",
        arguments={"service_name": "checkout-api", "environment": "staging"},
        settings=settings,
    )
    assert call_output == "service=checkout-api env=staging status=HEALTHY replicas=4/4"

    canary_output = call_mcp_tool(
        idx,
        server_name="deploy-pilot",
        tool_name="trigger_canary_rollout",
        arguments={"service_name": "checkout-api", "traffic_percent": 25},
        settings=settings,
    )
    assert canary_output == "canary_started service=checkout-api weight=25%"

    # Step 5: Consolidate scattered configs so all agents route exclusively through `agent-seshat`
    consolidated = consolidate_mcp_configs(
        workspace=workspace,
        user_home=fake_home,
        settings=settings,
    )
    assert len(consolidated) == 4
    assert (fake_home / ".claude.json.pre-seshat.bak").is_file()
    new_claude = json.loads(claude_cfg.read_text(encoding="utf-8"))
    assert list(new_claude["mcpServers"].keys()) == ["agent-seshat"]

    idx.close()
