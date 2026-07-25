# Developer entry points for futseg.
#
# Everything runs inside the development container (Dockerfile, compose.yaml).
# The container *is* the environment: there is no host virtualenv to create,
# activate or keep in sync, and no host Python at all. The only host requirement
# is a working docker with GPU support.

COMPOSE := docker compose
SERVICE := dev
RUN := $(COMPOSE) run --rm $(SERVICE)

.DEFAULT_GOAL := help
.PHONY: help build shell sync lint test check cuda gpu clean clean-cache

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

build: ## Build the development image
	$(COMPOSE) build

shell: ## Open an interactive shell in the container
	$(RUN) bash

sync: ## Refresh dependencies from uv.lock inside the container
	$(RUN) uv sync --frozen

lint: ## Run ruff
	$(RUN) uv run ruff check .

test: ## Run the fast test suite
	$(RUN) uv run pytest

check: lint test ## Run lint and tests

cuda: ## Verify CUDA does real work inside the container
	$(RUN) uv run python scripts/cuda_check.py

gpu: ## Show the GPU as the container sees it
	$(RUN) nvidia-smi

clean: ## Remove containers and the built image; keeps the weights cache
	$(COMPOSE) down --rmi local --remove-orphans

clean-cache: ## Also delete the cache volume -- forces a multi-GB re-download
	$(COMPOSE) down --volumes --remove-orphans
