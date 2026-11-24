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

- `make proxy-token` — create a Modal proxy token (once) and save it to `.env` as `MODAL_PROXY_TOKEN=wk-....ws-...`. Run before first use.
- `make download` — pre-fetch model + speculative model weights into the `huggingface-cache` Volume (`server/weights.py`, own small Modal app `gemma-weights`, CPU, needs the `huggingface-secret` Modal Secret). Shared by both servers.
- `make deploy-vllm` / `make serve-vllm` — deploy, or run a hot-reloading dev copy (`modal serve`).
- `make deploy-sglang` / `make serve-sglang` — same for the SGLang server.
- `make remove` — `modal app stop` on both apps. Stopping only containers does not work: the autoscaler restarts them until the scaledown window ends. Redeploy before next use.
- `make compare` / `make compare-quick` — deploy both servers, benchmark vLLM vs SGLang (`bench/compare.py`), write `reports/<timestamp>/report.md` + PNG charts + `results.json`. GPU cost: both servers run during the sweep and 15 min after.
- `make chat` — CLI chat (`client/chat.py`; accepts a one-shot prompt, `--url`, `--engine`, `--system`, `--no-thinking`). `ENGINE=sglang` picks the SGLang server (default `vllm`).
- `make ui` — Gradio UI on http://127.0.0.1:7860 (`--url`, `--engine`, `--share`). Same `ENGINE=` variable.
- `make test` — smoke test. Spins up a *temporary* GPU copy of the server via `modal run`, so expect a multi-minute cold start and GPU cost. Options: `uv run --env-file .env modal run -m tests.smoke_test --content "Hi" --no-twice`.

- `make check` — `ruff check`, `ruff format --check` and `mypy` (strict; config in `pyproject.toml`). Needs no Modal or GPU.

No unit-test suite is configured. `make test` is the only test and it requires Modal credentials and a GPU.
Run always `ruff format` after any change, this is our linter. Then run `make check`; it must pass.
Settings live in `server/config.py` as a frozen dataclass `Config` and one `CONFIG` instance; mypy checks their types. Read them as attributes: `CONFIG.model_name`.

## Architecture

- **Single source of config:** `server/config.py` (app name, model + pinned revisions, GPU type/count, port, region, `fast_boot`). The container imports the same file, because Modal ships the `server` package. Change models/GPUs there.
- **Server (`server/vllm_inference.py`):** a Modal `@app.server` class whose `@modal.enter()` launches `vllm serve` as a subprocess. vLLM CLI flags (tool/reasoning parsers `gemma4`, speculative config, multimodal disabled, `--enforce-eager` when `fast_boot`) are built there. It mounts the shared weights Volume plus its own `vllm-cache` Volume for compile artifacts.
- **Weights (`server/weights.py`):** `hf_cache` Volume, `HF_CACHE_DIR`, `hf_secret`, `MINUTES`, and the `download` function. Both servers import these from here; neither server imports the other.
- **SGLang server (`server/sglang_inference.py`):** a second Modal app (`sglang_modal_app_name`) with the same model, GPU, port, HF cache Volume and proxy auth; its class is also named `Server`. Speculative decoding uses `--speculative-algorithm NEXTN` with the same draft model (SGLang promotes it to its Gemma 4 MTP mode); `sglang_mtp` in the config turns it off.
- **Benchmark (`bench/compare.py`):** `run_load` runs on a Modal CPU container in `routing_region`; `main` (local entrypoint) looks up both deployed apps with `deployed_url(app_name)`, sweeps concurrency × scenarios, and writes the report. matplotlib is a dev dependency, imported only on the local side.
- **Auth:** Modal proxy auth only (`unauthenticated=False` on both servers); vLLM and SGLang have no API key of their own. Clients send `Authorization: Bearer $MODAL_PROXY_TOKEN` (`client.chat.proxy_token`), on every route including `/health`. Modal rejects requests without it before a container starts. The token is workspace-wide, not per app. `client.chat.ENGINES` maps engine → app name; chat/UI take `--engine vllm|sglang`. Local commands get the token through `uv run --env-file .env`; the benchmark passes it to its container with `Secret.from_local_environ`.
- **Clients:** `client/chat.py` holds the shared logic (`connect`, `server_url`, `wait_until_ready`, `stream_reply`, `ask`). `client/ui.py` and `tests/smoke_test.py` import from it, so run them as modules (`python -m client.ui`, `modal run -m tests.smoke_test`) to keep the repo root on `sys.path`.
- **Server discovery:** without `--url`, clients look up the deployed app with `modal.Server.from_name(app_name, "Server").get_url()`. A `modal serve` dev copy has a different URL and must be passed with `--url`.
- **Reasoning stream:** vLLM puts thinking in a non-standard `delta.reasoning` / `reasoning_content` field. `stream_reply` yields `("reasoning" | "content", text)`. Thinking is toggled via `extra_body={"chat_template_kwargs": {"enable_thinking": ...}}` and is never sent back in chat history.
- `model=` in requests must equal `--served-model-name`, which is the config's `model_name`.
