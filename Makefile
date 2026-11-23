# All commands run from the repo root, as modules (-m), so `server` and `client` import each other.
# Commands that call the server load VLLM_API_KEY from .env.
APP := $(shell grep '^modal_app_name:' config/vllm-inference.yaml | cut -d' ' -f2)
SERVER := -m server.vllm_inference
RUN := uv run --env-file .env

.PHONY: api-key download deploy serve remove chat ui test check

# Create VLLM_API_KEY in .env (once) and push it to the `vllm-api-key` Modal Secret.
api-key:
	@grep -q '^VLLM_API_KEY=' .env 2>/dev/null || printf '\nVLLM_API_KEY=%s\n' "$$(openssl rand -hex 32)" >> .env
	@uv run modal secret create vllm-api-key VLLM_API_KEY=$$(grep '^VLLM_API_KEY=' .env | cut -d= -f2) --force

download:
	uv run modal run $(SERVER)::download

deploy:
	uv run modal deploy $(SERVER)

serve:
	uv run modal serve $(SERVER)

remove:
	uv run modal app stop $(APP)

chat:
	$(RUN) python -m client.chat

ui:
	$(RUN) python -m client.ui

test:
	$(RUN) modal run -m tests.smoke_test

# Lint, format check and type check. Runs locally, no Modal or GPU needed.
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
