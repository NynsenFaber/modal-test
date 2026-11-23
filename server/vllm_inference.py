"""OpenAI-compatible vLLM server for Gemma 4 on Modal.

Usage::

    make download   # once: fetch model weights into a Modal Volume (CPU only)
    make deploy     # deploy the server; containers start on the first request

How it works:

1. ``image`` defines the container: CUDA and vLLM.
2. ``download`` fills the ``huggingface-cache`` Volume with the model weights,
   so that server boots read the weights from the Volume and download nothing.
3. ``Server`` starts ``vllm serve`` as a subprocess when a container boots.
   Modal forwards HTTP traffic to vLLM's port once the port is open. Until then
   Modal answers 503.

Settings come from ``server/config.py``.
Based on Modal's vLLM example: https://modal.com/docs/examples/vllm_inference
"""

import json
import subprocess

import modal

from server.config import CONFIG

MINUTES = 60  # seconds; Modal timeouts are given in seconds

app = modal.App(CONFIG.modal_app_name)

# --- Container image ----------------------------------------------------------
# The CUDA "devel" image provides the CUDA toolkit that vLLM's kernels need.
# Modal installs the GPU driver itself. `.entrypoint([])` removes the base
# image's entrypoint script, which only prints a banner.
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    # Pinned versions: a vLLM upgrade can change flags and model support.
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env(
        {
            # Faster Hugging Face downloads through the Xet storage backend.
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Log throughput and KV cache usage every second.
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)

# --- Storage ------------------------------------------------------------------
# Volumes persist between container runs. Without them, every cold start would
# download ~50 GB of weights and recompile the model.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
# Default cache locations of Hugging Face and vLLM inside the container.
HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

# --- Secrets ------------------------------------------------------------------
# Modal Secrets become environment variables inside the container.
# HF_TOKEN: needed because Gemma is a gated model on Hugging Face.
hf_secret = modal.Secret.from_name("huggingface-secret")
# VLLM_API_KEY: created by `make api-key`. vLLM reads it from the environment.
api_key_secret = modal.Secret.from_name("vllm-api-key")


@app.function(
    image=image,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[hf_secret],
    # Downloading ~50 GB can take a while; the default timeout is 5 minutes.
    timeout=60 * MINUTES,
)
def download() -> None:
    """Download the model and draft model weights into the ``huggingface-cache`` Volume.

    Runs on CPU, so it costs much less than downloading during a GPU boot.
    Files that are already complete are skipped, so running it again is safe.
    """
    # Imported here because `huggingface_hub` is installed in the image only,
    # not on your machine.
    from huggingface_hub import snapshot_download

    snapshot_download(CONFIG.model_name, revision=CONFIG.model_revision)
    snapshot_download(
        CONFIG.speculative_model_name,
        revision=CONFIG.speculative_model_revision,
    )
    # Save the new files now. Otherwise the Volume is only saved at container exit.
    hf_cache.commit()


def vllm_command() -> list[str]:
    """Build the ``vllm serve`` command line from the config.

    Kept separate from ``Server`` so the command can be printed and checked
    locally without starting a GPU.

    Returns:
        The command and its arguments, ready for ``subprocess.Popen``.
    """
    speculative_config = {
        "model": CONFIG.speculative_model_name,
        "revision": CONFIG.speculative_model_revision,
        # Tokens the draft model guesses per step. Later guesses are accepted
        # less often, so higher values give diminishing returns.
        "num_speculative_tokens": 4,
    }
    return [
        "vllm", "serve", CONFIG.model_name,
        "--revision", CONFIG.model_revision,
        # Clients must send this exact name as `model` in their requests.
        "--served-model-name", CONFIG.model_name,
        # Listen on all interfaces so Modal's proxy can reach the server.
        "--host", "0.0.0.0",
        "--port", str(CONFIG.vllm_port),
        "--uvicorn-log-level=info",
        # Prepare the next batch on the CPU while the GPU runs the current one.
        "--async-scheduling",
        # Eager mode skips torch.compile and CUDA graphs: faster boot,
        # slower generation. Controlled by `fast_boot` in the config.
        "--enforce-eager" if CONFIG.fast_boot else "--no-enforce-eager",
        # Split each layer across the GPUs of one container.
        "--tensor-parallel-size", str(CONFIG.n_gpu),
        # Text only: no image, video or audio input, which saves GPU memory.
        "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0, "audio": 0}),
        # Let the model decide when to call tools, and parse Gemma 4's
        # tool-call and reasoning ("thinking") output into OpenAI fields.
        "--enable-auto-tool-choice",
        "--reasoning-parser", "gemma4",
        "--tool-call-parser", "gemma4",
        # Speculative decoding: the draft model guesses several tokens, and the
        # big model checks them all in one step. Same output, fewer big steps.
        "--speculative-config", json.dumps(speculative_config),
    ]  # fmt: skip


@app.server(
    image=image,
    gpu=f"{CONFIG.gpu_type}:{CONFIG.n_gpu}",
    port=CONFIG.vllm_port,
    routing_region=CONFIG.routing_region,
    volumes={HF_CACHE_DIR: hf_cache, VLLM_CACHE_DIR: vllm_cache},
    secrets=[hf_secret, api_key_secret],
    # Modal's own auth is off. Instead, VLLM_API_KEY makes vLLM require
    # `Authorization: Bearer <key>` on /v1/* routes. /health stays open.
    unauthenticated=True,
    # Idle time before the container stops. The GPU is billed until then.
    scaledown_window=15 * MINUTES,
    # Max boot time: load weights, compile, capture CUDA graphs.
    # Modal kills and restarts a container that takes longer.
    startup_timeout=10 * MINUTES,
    # Requests per container before Modal starts another one.
    # Tune it with a load test.
    target_concurrency=100,
)
class Server:
    """One vLLM process per container, started and stopped with the container."""

    # Set in `start`. Declared here for type checkers.
    process: subprocess.Popen[bytes]

    @modal.enter()
    def start(self) -> None:
        """Launch vLLM. Runs once when a container boots, before it gets traffic."""
        cmd = vllm_command()
        print(*cmd)  # visible in the container logs, useful for debugging
        # Non-blocking: Modal itself waits for the port to open before routing.
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        """Stop vLLM cleanly when the container shuts down."""
        self.process.terminate()
