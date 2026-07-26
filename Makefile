# Developer entry points for futseg.
#
# Everything runs inside the development container (Dockerfile, compose.yaml).
# The container *is* the environment: there is no host virtualenv to create,
# activate or keep in sync, and no host Python at all. The only host requirement
# is a working docker with GPU support.
#
# Two ways to work, and the task targets below suit either:
#
#   one-off   run a target, get a fresh container, throw it away
#   resident  `make up` once, then targets reuse that container (faster, and
#             `make shell` attaches to the same one an IDE is using)

COMPOSE := docker compose
SERVICE := dev

.DEFAULT_GOAL := help
.PHONY: help build rebuild up down restart ps logs shell exec sync lint test \
        check cuda gpu clean clean-cache

# Run a command in the container: exec into the resident one when it is up,
# otherwise spin up a one-off and discard it. Keeps every task target working
# identically whether or not `make up` has been run.
define in_container
@if [ -n "$$($(COMPOSE) ps -q $(SERVICE) 2>/dev/null)" ]; then \
  $(COMPOSE) exec $(SERVICE) $(1); \
else \
  $(COMPOSE) run --rm $(SERVICE) $(1); \
fi
endef

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} \
	     /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next} \
	     /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' \
	     $(MAKEFILE_LIST)
	@echo

##@ Image

build: ## Build the image; the dependency layer stays cached
	$(COMPOSE) build

rebuild: ## Rebuild from scratch, ignoring every cached layer
	$(COMPOSE) build --no-cache

##@ Container lifecycle

up: ## Start the container in the background
	$(COMPOSE) up -d

down: ## Stop and remove the container
	$(COMPOSE) down

restart: ## Stop, then start again
	$(COMPOSE) down
	$(COMPOSE) up -d

ps: ## Show whether the container is running
	$(COMPOSE) ps

logs: ## Follow the container's logs
	$(COMPOSE) logs -f

##@ Working inside

shell: ## Open an interactive shell in the container
	@if [ -n "$$($(COMPOSE) ps -q $(SERVICE) 2>/dev/null)" ]; then \
	  $(COMPOSE) exec $(SERVICE) bash; \
	else \
	  $(COMPOSE) run --rm $(SERVICE) bash; \
	fi

exec: ## Run one command inside: make exec CMD="python -V"
	@test -n "$(CMD)" || { echo 'usage: make exec CMD="python -V"' >&2; exit 2; }
	$(call in_container,$(CMD))

sync: ## Refresh dependencies from uv.lock
	$(call in_container,uv sync --frozen)

##@ Checks

lint: ## Run ruff
	$(call in_container,uv run ruff check .)

test: ## Run the fast test suite
	$(call in_container,uv run pytest)

check: lint test ## Run lint and tests

cuda: ## Verify CUDA does real work inside the container
	$(call in_container,uv run python scripts/cuda_check.py)

gpu: ## Show the GPU as the container sees it
	$(call in_container,nvidia-smi)

##@ Cleanup

clean: ## Remove containers and the built image; keeps the cache volume
	$(COMPOSE) down --rmi local --remove-orphans

clean-cache: ## Also delete the cache volume -- forces a multi-GB re-download
	$(COMPOSE) down --volumes --remove-orphans
