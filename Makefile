.PHONY: help install uninstall up down restart logs status build test

help: ## Show available make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Start Docker daemon and install `sesh` CLI onto host PATH
	@chmod +x install.sh bin/sesh bin/agent-seshat
	@./install.sh

uninstall: ## Stop Docker daemon and remove `sesh` CLI symlinks
	@chmod +x uninstall.sh
	@./uninstall.sh

up: ## Start the warm Agent-Seshat Docker daemon
	docker compose up -d

down: ## Stop the Agent-Seshat Docker daemon
	docker compose down

restart: ## Restart the Agent-Seshat Docker daemon
	docker compose restart

logs: ## Tail logs from the Agent-Seshat Docker daemon
	docker compose logs -f

status: ## Check Docker daemon health and indexed skills status
	@./bin/sesh daemon status
	@./bin/sesh status

build: ## Rebuild the Agent-Seshat Docker image
	docker compose build

test: ## Run the pytest suite
	uv run pytest -v
