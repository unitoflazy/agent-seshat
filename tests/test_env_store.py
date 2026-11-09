"""Tests for scoped Environment Variable & Secret Store (`env_store.py`), MCP tools, and downstream MCP injection."""

from __future__ import annotations

import sys
from pathlib import Path

from agent_seshat.config import Settings
from agent_seshat.env_store import EnvStore
from agent_seshat.indexer import SkillIndex
from agent_seshat.mcp_aggregator import MCPServerEntry, call_mcp_tool, save_mcp_registry
from agent_seshat.server import SeshatService

MOCK_ENV_ECHO_SERVER = '''"""Mock stdio MCP server that echoes environment variables injected by Agent-Seshat."""
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("env-echo")

@mcp.tool()
def read_injected_env(var_name: str) -> str:
    """Return the runtime value of `var_name` inside the downstream MCP server process."""
    return os.environ.get(var_name, "<MISSING>")

if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def test_env_store_scoping_masking_and_persistence(tmp_path: Path) -> None:
    home = tmp_path / ".agent-seshat"
    settings = Settings(home=home)
    store = EnvStore(settings=settings)

    # 1. Standard non-secret variable (global)
    r_url = store.set(
        "API_BASE_URL",
        "https://api.global.example.com",
        scope="global",
        description="Default API endpoint",
    )
    assert r_url.is_secret is False
    assert r_url.display_value(reveal_secret=False) == "https://api.global.example.com"

    # 2. Project-scoped override for `payments-api`
    store.set(
        "API_BASE_URL",
        "https://api.payments.internal",
        scope="project:payments-api",
        description="Payments internal endpoint",
    )

    # 3. Secret variable (auto-detected from key/value)
    r_sec = store.set(
        "STRIPE_API_KEY",
        "sk-live-1234567890abcdefghijklmnop9876",
        scope="project:payments-api",
        description="Production Stripe secret key",
    )
    assert r_sec.is_secret is True
    assert "1234567890" not in r_sec.masked_value
    assert r_sec.display_value(reveal_secret=False) == "sk-l...9876"
    assert r_sec.display_value(reveal_secret=True) == "sk-live-1234567890abcdefghijklmnop9876"

    # 4. Scope resolution: project:payments-api overrides global, other projects fall back to global
    resolved_payments = store.get("API_BASE_URL", project="payments-api")
    assert resolved_payments is not None
    assert resolved_payments.value == "https://api.payments.internal"
    assert resolved_payments.scope == "project:payments-api"

    resolved_other = store.get("API_BASE_URL", project="other-repo")
    assert resolved_other is not None
    assert resolved_other.value == "https://api.global.example.com"
    assert resolved_other.scope == "global"

    # 5. MCP service tools: list_env_vars masks secrets, get_env_var reveals on request
    service = SeshatService(store.index, default_project="payments-api")
    listing_md = service.list_env_vars()
    assert "https://api.payments.internal" in listing_md
    assert "sk-l...9876" in listing_md
    assert "sk-live-1234567890abcdefghijklmnop9876" not in listing_md

    fetched_md = service.get_env_var("STRIPE_API_KEY", reveal_secret=True)
    assert "sk-live-1234567890abcdefghijklmnop9876" in fetched_md

    # 6. Cache DB rebuild resiliency: deleting index.db restores variables from env_store.json
    store.index.close()
    for p in settings.cache_dir.glob("index.db*"):
        p.unlink()

    rebuilt_store = EnvStore(settings=settings)
    restored = rebuilt_store.get("STRIPE_API_KEY", project="payments-api")
    assert restored is not None
    assert restored.value == "sk-live-1234567890abcdefghijklmnop9876"
    assert restored.is_secret is True

    # 7. Unset removes variable
    assert rebuilt_store.unset("API_BASE_URL", scope="project:payments-api") is True
    after_unset = rebuilt_store.get("API_BASE_URL", project="payments-api")
    assert after_unset is not None
    assert after_unset.scope == "global"
    rebuilt_store.index.close()


def test_dotenv_import_and_export(tmp_path: Path) -> None:
    home = tmp_path / ".agent-seshat"
    store = EnvStore(settings=Settings(home=home))

    dotenv_file = tmp_path / ".env.staging"
    dotenv_file.write_text(
        "# Staging service port\n"
        "PORT=8080\n"
        'export SERVICE_TITLE="Seshat Payments Gateway" # Display title\n'
        "GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD\n",
        encoding="utf-8",
    )

    imported = store.import_dotenv(dotenv_file, scope="project:staging-app")
    assert len(imported) == 3

    by_key = {r.key: r for r in imported}
    assert by_key["PORT"].value == "8080"
    assert by_key["PORT"].description == "Staging service port"
    assert by_key["PORT"].is_secret is False

    assert by_key["SERVICE_TITLE"].value == "Seshat Payments Gateway"
    assert by_key["SERVICE_TITLE"].description == "Display title"

    assert by_key["GITHUB_TOKEN"].is_secret is True

    exported_masked = store.export_dotenv(scope="project:staging-app", mask_secrets=True)
    assert 'SERVICE_TITLE="Seshat Payments Gateway"' in exported_masked
    assert "ghp_...ABCD" in exported_masked
    assert "ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD" not in exported_masked

    exported_raw = store.export_dotenv(scope="project:staging-app", mask_secrets=False)
    assert "ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD" in exported_raw
    store.index.close()


def test_downstream_mcp_server_receives_injected_env_vars(tmp_path: Path) -> None:
    home = tmp_path / ".agent-seshat"
    settings = Settings(home=home)
    idx = SkillIndex(settings)
    store = EnvStore(idx, settings=settings)

    store.set("SHARED_REGION", "ap-southeast-1", scope="global")
    store.set("MCP_API_TOKEN", "tok_secret_mcp_9999999999", scope="mcp:env-echo", is_secret=True)

    script = tmp_path / "mock_env_echo.py"
    script.write_text(MOCK_ENV_ECHO_SERVER, encoding="utf-8")

    save_mcp_registry(
        [
            MCPServerEntry(
                name="env-echo",
                transport="stdio",
                command=sys.executable,
                args=[str(script)],
                enabled=True,
                source_agents=["claude"],
            )
        ],
        settings=settings,
    )

    region_val = call_mcp_tool(
        idx,
        server_name="env-echo",
        tool_name="read_injected_env",
        arguments={"var_name": "SHARED_REGION"},
        settings=settings,
    )
    assert region_val == "ap-southeast-1"

    token_val = call_mcp_tool(
        idx,
        server_name="env-echo",
        tool_name="read_injected_env",
        arguments={"var_name": "MCP_API_TOKEN"},
        settings=settings,
    )
    assert token_val == "tok_secret_mcp_9999999999"
    idx.close()
