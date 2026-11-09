#!/usr/bin/env bash
# ==============================================================================
# Agent-Seshat (sesh) — Uninstaller
#
# Stops the Docker daemon and removes the `sesh` & `agent-seshat` CLI symlinks.
# Preserves ~/.agent-seshat data directory unless `--purge` is passed.
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESH_HOME="${AGENT_SESHAT_HOME:-$HOME/.agent-seshat}"
PURGE=0

for arg in "$@"; do
    case "$arg" in
        --purge)
            PURGE=1
            ;;
        --help|-h)
            echo "Usage: ./uninstall.sh [--purge]"
            echo ""
            echo "Options:"
            echo "  --purge   Also delete ${SESH_HOME} (skills, SQLite index, env store)"
            exit 0
            ;;
    esac
done

echo "==> Stopping Agent-Seshat Docker container..."
if command -v docker >/dev/null 2>&1 && [ -f "${REPO_DIR}/docker-compose.yml" ]; then
    docker compose -f "${REPO_DIR}/docker-compose.yml" down 2>/dev/null || true
fi

echo "==> Removing CLI symlinks..."
for bin_dir in "${HOME}/.local/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
    for cmd in "sesh" "agent-seshat"; do
        target="${bin_dir}/${cmd}"
        if [ -L "$target" ]; then
            rm -f "$target"
            echo "    Removed ${target}"
        fi
    done
done

if [ "$PURGE" -eq 1 ]; then
    echo "==> Purging data directory ${SESH_HOME}..."
    rm -rf "${SESH_HOME}"
else
    echo "==> Preserved data directory at ${SESH_HOME} (pass --purge to delete)."
fi

echo "Agent-Seshat uninstalled."
