# All commands run from the repo root, as modules (-m), so `server` and `client` import each other.
# Commands that call the servers load MODAL_PROXY_TOKEN from .env.
APP := $(shell uv run python -c "from server.config import CONFIG; print(CONFIG.modal_app_name)")
SGLANG_APP := $(shell uv run python -c "from server.config import CONFIG; print(CONFIG.sglang_modal_app_name)")
SERVER := -m server.vllm_inference
SGLANG := -m server.sglang_inference
RUN := uv run --env-file .env
# Server for `make chat` / `make ui`: vllm (default) or sglang, e.g. `make chat ENGINE=sglang`.
ENGINE ?= vllm

.PHONY: proxy-token download deploy-vllm serve-vllm deploy-sglang serve-sglang remove chat ui test check compare compare-quick

# Fill the weights Volume shared by both servers (CPU container, ~50 GB, once).
download:
	uv run modal run -m server.weights::download

deploy-vllm:
	uv run modal deploy $(SERVER)

serve-vllm:
	uv run modal serve $(SERVER)

deploy-sglang:
	uv run modal deploy $(SGLANG)

serve-sglang:
	uv run modal serve $(SGLANG)

# Stop both apps and their containers, so GPUs stop billing now. Stopping only
# the containers does not work: the autoscaler starts new ones until the
# scaledown window ends. `-`: keep going if one app is not deployed.
# Run `make deploy-vllm` / `make deploy-sglang` before the next use.
remove:
	-uv run modal app stop $(APP)
	-uv run modal app stop $(SGLANG_APP)

chat:
	$(RUN) python -m client.chat --engine $(ENGINE)

ui:
	$(RUN) python -m client.ui --engine $(ENGINE)

test:
	$(RUN) modal run -m tests.smoke_test

# Benchmark vLLM vs SGLang and write reports/<timestamp>/report.md.
# Both servers stay up (and billed) 15 min after; `make remove` stops them.
compare: deploy-vllm deploy-sglang
	$(RUN) modal run -m bench.compare

compare-quick: deploy-vllm deploy-sglang
	$(RUN) modal run -m bench.compare --quick

# Lint, format check and type check. Runs locally, no Modal or GPU needed.
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
