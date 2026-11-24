"""OpenAI-compatible vLLM server for Gemma 4 on Modal.

Usage::

    make download   # once: fetch model weights into a Modal Volume (server/weights.py)
    make deploy-vllm   # deploy the server; containers start on the first request

How it works:

1. ``image`` defines the container: CUDA and vLLM.
2. ``Server`` starts ``vllm serve`` as a subprocess when a container boots.
   Modal forwards HTTP traffic to vLLM's port once the port is open. Until then
   Modal answers 503.
3. Weights come from the ``huggingface-cache`` Volume, filled once by
   ``make download`` (``server/weights.py``), so boots download nothing.

Settings come from ``server/config.py``.
Based on Modal's vLLM example: https://modal.com/docs/examples/vllm_inference
"""

import json
import subprocess

import modal

from server.config import CONFIG
from server.weights import HF_CACHE_DIR, MINUTES, hf_cache, hf_secret

app = modal.App(CONFIG.modal_app_name)

# --- Container image ----------------------------------------------------------
# The CUDA "devel" image provides the CUDA toolkit that vLLM's kernels need. This
# setup comes from Modal's vLLM example: https://modal.com/docs/examples/vllm_inference
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            "VLLM_LOG_STATS_INTERVAL": "1",  # more frequent metrics logging
        }
    )
)

# --- Storage ------------------------------------------------------------------
# vLLM's compile artifacts persist between container runs, so a cold start does
# not recompile the model. The weights Volume is shared, see server/weights.py.
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
# Default cache location of vLLM inside the container, so no extra config is needed.
VLLM_CACHE_DIR = "/root/.cache/vllm"


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
        "--uvicorn-log-level=info", # vLLM logs go to the container logs, visible in Modal's dashboard
        # Prepare the next batch on the CPU while the GPU runs the current one.
        "--async-scheduling",
        # Eager mode skips torch.compile and CUDA graphs: faster boot,
        # slower generation. Controlled by `fast_boot` in the config.
        "--enforce-eager" if CONFIG.fast_boot else "--no-enforce-eager",
        # Number of GPUs to use. vLLM splits the model weights across them.
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
    secrets=[hf_secret],
    # Modal proxy auth
    unauthenticated=False,
    # Idle time before the container stops. The GPU is billed until then.
    scaledown_window=15 * MINUTES,
    # Max boot time: load weights, compile, capture CUDA graphs.
    # Modal kills and restarts a container that takes longer.
    startup_timeout=10 * MINUTES,
    # At most one GPU container. The autoscaler treats target_concurrency as a
    # soft target and started a 2nd (cold) container at 64 concurrent requests,
    # which splits benchmark load across GPUs. One container also caps the bill.
    max_containers=1,
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
