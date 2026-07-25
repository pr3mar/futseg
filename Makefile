# Developer entry points for futseg.
#
# Every target invokes uv with UV_PROJECT_ENVIRONMENT set *here* rather than relying
# on shell configuration. That is the point of this file, not a convenience: the
# source tree lives on /mnt/c (docs/design/2026-07-25-linux-first-platform.md), and
# when uv is not told where the environment belongs it silently builds a multi-GB
# .venv in-tree on drvfs. An export in ~/.bashrc does not survive a terminal that was
# opened before the edit, a non-interactive shell, or an IDE run configuration. A
# variable set in the Makefile survives all three.

SHELL := /bin/bash
.DEFAULT_GOAL := help

export UV_PROJECT_ENVIRONMENT := $(HOME)/.venvs/futseg

# uv installs to ~/.local/bin, which only reaches PATH via ~/.profile in a login
# shell. Same reasoning as above: a non-interactive `make` must not fail merely
# because the caller's shell was configured differently.
export PATH := $(HOME)/.local/bin:$(PATH)

UV := uv

.PHONY: help guard sync lint test check cuda env clean-venv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

guard: ## Check the platform, uv, and that no stray in-tree .venv exists
	@if [ "$$(uname -s)" != "Linux" ]; then \
	  echo "ERROR: futseg targets Linux; detected $$(uname -s)." >&2; \
	  echo "       Run make from WSL2, not Git Bash or PowerShell." >&2; \
	  echo "       See docs/design/2026-07-25-linux-first-platform.md" >&2; \
	  exit 1; \
	fi
	@command -v $(UV) >/dev/null || { echo "ERROR: uv is not on PATH." >&2; exit 1; }
	@if [ -d .venv ]; then \
	  echo "WARNING: a stray in-tree .venv exists." >&2; \
	  echo "         Something ran uv without UV_PROJECT_ENVIRONMENT set, so it built" >&2; \
	  echo "         the environment on drvfs. Remove it with: make clean-venv" >&2; \
	fi

sync: guard ## Create or refresh the virtualenv from uv.lock
	$(UV) sync

lint: guard ## Run ruff
	$(UV) run ruff check .

test: guard ## Run the fast test suite
	$(UV) run pytest

check: lint test ## Run lint and tests

cuda: guard ## Verify CUDA does real work on this machine
	$(UV) run python scripts/cuda_check.py

env: guard ## Print the resolved environment
	@echo "UV_PROJECT_ENVIRONMENT = $(UV_PROJECT_ENVIRONMENT)"
	@$(UV) run python -c 'import sys; print("sys.prefix             =", sys.prefix)'

clean-venv: ## Remove a stray in-tree .venv (never the real environment)
	@if [ -d .venv ]; then \
	  echo "removing in-tree .venv ($$(du -sh .venv | cut -f1))"; \
	  rm -rf .venv; \
	else \
	  echo "no in-tree .venv present"; \
	fi
