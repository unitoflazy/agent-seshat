#!/usr/bin/env bash
# ==============================================================================
# Agent-Seshat (sesh) — One-Command Docker Installer
#
# 1. Verifies Docker & Docker Compose availability
# 2. Initializes ~/.agent-seshat and seeds sample skills if empty
# 3. Builds (if needed) and starts the warm `agent-seshat` Docker daemon
# 4. Installs the zero-host-dependency `sesh` and `agent-seshat` CLI launchers
#    onto the host PATH (~/.local/bin and /opt/homebrew/bin if writable)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${AGENT_SESHAT_PORT:-8765}"
SESH_HOME="${AGENT_SESHAT_HOME:-$HOME/.agent-seshat}"
FORCE_BUILD=0

for arg in "$@"; do
    case "$arg" in
        --build|-b)
            FORCE_BUILD=1
            ;;
        --help|-h)
            echo "Usage: ./install.sh [--build]"
            echo ""
            echo "Options:"
            echo "  --build, -b   Force rebuild of the agent-seshat Docker image"
            exit 0
            ;;
    esac
done

echo "==> [1/5] Checking Docker prerequisites..."
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: 'docker' is not installed or not on PATH." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker daemon is not running. Please start Docker Desktop first." >&2
    exit 1
fi

echo "==> [2/5] Initializing data directory at ${SESH_HOME}..."
mkdir -p "${SESH_HOME}/skills" "${SESH_HOME}/cache"
if [ -d "${REPO_DIR}/sample_skills" ] && [ -z "$(ls -A "${SESH_HOME}/skills" 2>/dev/null || true)" ]; then
    cp -R "${REPO_DIR}/sample_skills/"* "${SESH_HOME}/skills/"
    echo "    Seeded starter skills into ${SESH_HOME}/skills/"
fi

echo "==> [3/5] Starting warm Agent-Seshat Docker daemon..."
chmod +x "${REPO_DIR}/bin/sesh" "${REPO_DIR}/bin/agent-seshat"

if [ "$FORCE_BUILD" -eq 1 ]; then
    docker compose -f "${REPO_DIR}/docker-compose.yml" up -d --build
else
    docker compose -f "${REPO_DIR}/docker-compose.yml" up -d
fi

echo "==> [4/5] Waiting for daemon health check at http://127.0.0.1:${PORT}/health..."
HEALTHY=0
for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 0.5
done

if [ "$HEALTHY" -eq 1 ]; then
    echo "    Daemon is healthy: $(curl -fsS "http://127.0.0.1:${PORT}/health")"
else
    echo "    Warning: Daemon is still warming up. Check 'sesh daemon logs' for progress."
fi

echo "==> [5/5] Installing 'sesh' & 'agent-seshat' CLI onto host PATH..."
mkdir -p "${HOME}/.local/bin"
ln -sf "${REPO_DIR}/bin/sesh" "${HOME}/.local/bin/sesh"
ln -sf "${REPO_DIR}/bin/agent-seshat" "${HOME}/.local/bin/agent-seshat"
INSTALLED_PATHS=("${HOME}/.local/bin/sesh")

# Also link into /opt/homebrew/bin or /usr/local/bin if writable so terminals/agents find it immediately
for sys_bin in "/opt/homebrew/bin" "/usr/local/bin"; do
    if [ -d "$sys_bin" ] && [ -w "$sys_bin" ]; then
        ln -sf "${REPO_DIR}/bin/sesh" "${sys_bin}/sesh"
        ln -sf "${REPO_DIR}/bin/agent-seshat" "${sys_bin}/agent-seshat"
        INSTALLED_PATHS+=("${sys_bin}/sesh")
    fi
done

# Ensure ~/.local/bin is in user's shell RC if not already on PATH
if [[ ":$PATH:" != *":${HOME}/.local/bin:"* ]]; then
    for rc_file in "${HOME}/.zshrc" "${HOME}/.bashrc"; do
        if [ -f "$rc_file" ] && ! grep -q '.local/bin' "$rc_file" 2>/dev/null; then
            echo '' >> "$rc_file"
            echo '# Agent-Seshat CLI' >> "$rc_file"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc_file"
        fi
    done
fi

echo ""
echo "=================================================================="
echo " Agent-Seshat (sesh) installed successfully!"
echo "=================================================================="
echo "  CLI Installed : ${INSTALLED_PATHS[*]}"
echo "  Docker Daemon : http://127.0.0.1:${PORT}/mcp"
echo "  Data Dir      : ${SESH_HOME}"
echo ""
echo "Quick Start Commands:"
echo "  sesh status                        # View indexed skills, rules & daemon status"
echo "  sesh search \"worker pool\"          # Run hybrid semantic + BM25 search"
echo "  sesh deep-sync                     # Scan machine for CLAUDE.md, .cursorrules, etc."
echo "  sesh rules list                    # List all rules & pinned status"
echo "  sesh env set API_KEY val           # Store scoped environment variables"
echo "  sesh connect --all                 # Connect Claude, Cursor, Codex & Copilot"
echo "  sesh daemon [up|down|restart|logs] # Manage the Docker container lifecycle"
echo "=================================================================="
