# vLLM on Modal

Serve Gemma 4 (26B-A4B) as an OpenAI-compatible API on a Modal GPU, and chat with it from the terminal or a browser.

```
config/vllm-inference.yaml   all settings: model, GPU, app name, region
server/config.py             loads the YAML (locally and inside the container)
server/vllm_inference.py     Modal app: image, weight download, vLLM server
client/chat.py               terminal chat + shared client helpers
client/ui.py                 Gradio browser chat
tests/smoke_test.py          starts a temporary server and checks that it answers
```

## Setup (once)

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync` and `uv run modal setup`.
2. Accept the Gemma license on Hugging Face, then store your token:
   `uv run modal secret create huggingface-secret HF_TOKEN=hf_...`
3. `make api-key` creates `VLLM_API_KEY` in `.env` and stores it in Modal.
4. `make download` fetches the model weights into a Modal Volume (CPU, ~50 GB).
5. `make deploy`.

## Use

| Command | What it does |
|---|---|
| `make chat` | Terminal chat. `python -m client.chat --help` for options. |
| `make ui` | Browser chat at http://127.0.0.1:7860. |
| `make test` | Smoke test on a temporary GPU server. |
| `make serve` | Dev copy that reloads on code changes. Pass its URL with `--url`. |
| `make remove` | Stop the deployed app. |
| `make check` | Lint (ruff), format check and type check (mypy, strict). No GPU needed. |

The first request after 15 idle minutes waits for a cold start (about 3 minutes).
The GPU is billed while a container runs, including those 15 idle minutes.
