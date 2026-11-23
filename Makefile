APP := example-vllm-inference
SCRIPT := scripts/vllm-inference.py

.PHONY: download deploy serve remove

download:
	uv run modal run $(SCRIPT)::download

deploy:
	uv run modal deploy $(SCRIPT)

serve:
	uv run modal serve $(SCRIPT)

remove:
	uv run modal app stop $(APP)