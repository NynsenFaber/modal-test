APP := example-vllm-inference
SCRIPT := scripts/vllm-inference.py

.PHONY: deploy serve remove

deploy:
	uv run modal deploy $(SCRIPT)

serve:
	uv run modal serve $(SCRIPT)

remove:
	uv run modal app stop $(APP)