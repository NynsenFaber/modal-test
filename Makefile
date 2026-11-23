APP := example-vllm-inference
SCRIPT := scripts/vllm-inference.py

.PHONY: api-key download deploy serve test remove chat

# Create VLLM_API_KEY in .env (once) and push it to the `vllm-api-key` Modal Secret.
api-key:
	@grep -q '^VLLM_API_KEY=' .env 2>/dev/null || printf '\nVLLM_API_KEY=%s\n' "$$(openssl rand -hex 32)" >> .env
	@uv run modal secret create vllm-api-key VLLM_API_KEY=$$(grep '^VLLM_API_KEY=' .env | cut -d= -f2) --force

download:
	uv run modal run $(SCRIPT)::download

deploy:
	uv run modal deploy $(SCRIPT)

serve:
	uv run modal serve $(SCRIPT)

test:
	uv run --env-file .env modal run $(SCRIPT)

remove:
	uv run modal app stop $(APP)

chat:
	uv run --env-file .env python scripts/chat.py
