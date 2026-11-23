# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An OpenAI-compatible vLLM server (Gemma 4 26B-A4B + MTP speculative decoding) deployed on Modal, plus local clients (CLI chat, Gradio UI) and a smoke test. Python >=3.12, managed with `uv`.

# What to do at first start
Download locally the modal skills
```bash
modal skills install --claude
```

## Commands

All go through the Makefile (`uv run ...` under the hood):

- `make api-key` — generate `VLLM_API_KEY` into `.env` (once) and push it to the `vllm-api-key` Modal Secret. Run before first deploy.
- `make download` — pre-fetch model + speculative model weights into the `huggingface-cache` Volume (CPU function, needs the `huggingface-secret` Modal Secret).
- `make deploy` / `make serve` — deploy, or run a hot-reloading dev copy (`modal serve`).
- `make remove` — stop the deployed app.
- `make chat` — CLI chat (`client/chat.py`; accepts a one-shot prompt, `--url`, `--system`, `--no-thinking`).
- `make ui` — Gradio UI on http://127.0.0.1:7860 (`--url`, `--share`).
- `make test` — smoke test. Spins up a *temporary* GPU copy of the server via `modal run`, so expect a multi-minute cold start and GPU cost. Options: `uv run --env-file .env modal run -m tests.smoke_test --content "Hi" --no-twice`.

- `make check` — `ruff check`, `ruff format --check` and `mypy` (strict; config in `pyproject.toml`). Needs no Modal or GPU.

No unit-test suite is configured. `make test` is the only test and it requires Modal credentials and a GPU.
Run always `ruff format` after any change, this is our linter. Then run `make check`; it must pass.
`server/config.py` types the YAML as the `Config` TypedDict and fails fast on missing keys, so add any new YAML key there too.

## Architecture

- **Single source of config:** `config/vllm-inference.yaml` (app name, model + pinned revisions, GPU type/count, port, region, `fast_boot`). The server reads it locally and also bakes it into the image at `/root/vllm-inference.yaml`, choosing the path via `modal.is_local()`. Clients read the same file for the app name and model name. Change models/GPUs here, not in code.
- **Server (`server/vllm_inference.py`):** a Modal `@app.server` class whose `@modal.enter()` launches `vllm serve` as a subprocess. vLLM CLI flags (tool/reasoning parsers `gemma4`, speculative config, multimodal disabled, `--enforce-eager` when `fast_boot`) are built there. Two Modal Volumes cache HF weights and vLLM compile artifacts.
- **Auth:** the endpoint is `unauthenticated=True` at the Modal layer; vLLM itself enforces `Authorization: Bearer $VLLM_API_KEY` on `/v1/*` (`/health` stays open). Local commands get the key through `uv run --env-file .env`.
- **Clients:** `client/chat.py` holds the shared logic (`connect`, `server_url`, `wait_until_ready`, `stream_reply`, `ask`). `client/ui.py` and `tests/smoke_test.py` import from it, so run them as modules (`python -m client.ui`, `modal run -m tests.smoke_test`) to keep the repo root on `sys.path`.
- **Server discovery:** without `--url`, clients look up the deployed app with `modal.Server.from_name(app_name, "Server").get_url()`. A `modal serve` dev copy has a different URL and must be passed with `--url`.
- **Reasoning stream:** vLLM puts thinking in a non-standard `delta.reasoning` / `reasoning_content` field. `stream_reply` yields `("reasoning" | "content", text)`. Thinking is toggled via `extra_body={"chat_template_kwargs": {"enable_thinking": ...}}` and is never sent back in chat history.
- `model=` in requests must equal `--served-model-name`, which is the config's `model_name`.
