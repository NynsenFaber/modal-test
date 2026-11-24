"""OpenAI-compatible SGLang server for Gemma 4 on Modal, the twin of ``vllm_inference``.

Usage::

    make download        # once, shared with vLLM: weights go to the same Volume (server/weights.py)
    make deploy-sglang   # deploy; containers start on the first request

Same model, GPU, port, auth and speculative draft model as the vLLM server,
so ``make compare`` measures the engines and nothing else. It is a separate
Modal app (``CONFIG.sglang_modal_app_name``) whose class is also named
``Server``, so clients find it the same way.

Settings come from ``server/config.py``.
Based on Modal's SGLang example: https://modal.com/docs/examples/sglang_low_latency
and SGLang's Gemma 4 cookbook: https://docs.sglang.io/cookbook/autoregressive/Google/Gemma4
"""

import subprocess

import modal

from server.config import CONFIG
from server.weights import HF_CACHE_DIR, MINUTES, hf_cache, hf_secret

app = modal.App(CONFIG.sglang_modal_app_name)

# Official SGLang image, pinned: an upgrade can change flags and model support.
# v0.5.23 (latest release, Nov 2026) runs Gemma 4 with its "assistant" draft
# model. "runtime" is the slim variant without build tools.
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.23-runtime")
    .entrypoint([])
    # The Volume is mounted here; an empty directory avoids a mount conflict.
    .run_commands(f"rm -rf {HF_CACHE_DIR}")
    .env(
        {
            # Same cache layout as the vLLM image, so the weights that
            # `make download` stored are found without a second download.
            "HF_HOME": HF_CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)


def sglang_command() -> list[str]:
    """Build the ``sglang.launch_server`` command line from the config.

    Returns:
        The command and its arguments, ready for ``subprocess.Popen``.
    """
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", CONFIG.model_name,
        "--revision", CONFIG.model_revision,
        # Clients must send this exact name as `model` in their requests.
        "--served-model-name", CONFIG.model_name,
        "--host", "0.0.0.0",
        "--port", str(CONFIG.vllm_port),
        "--tp-size", str(CONFIG.n_gpu),
        # Same Gemma 4 parsers as vLLM: thinking goes to `reasoning_content`.
        "--reasoning-parser", "gemma4",
        "--tool-call-parser", "gemma4",
        # No text-only flag here: SGLang v0.5.23's --language-model-only rejects
        # Gemma4ForConditionalGeneration, so the vision/audio towers stay loaded.
        # SGLang defaults to 48 parallel requests with speculative decoding.
        # Raise it to the requests Modal sends per container (target_concurrency),
        # so requests queue in neither engine before Modal scales out.
        "--max-running-requests", "100",
        # Prometheus metrics on /metrics.
        "--enable-metrics",
        # Share of GPU memory for weights + KV cache. Leaves room for the draft
        # model's activations and CUDA graphs on a single 80 GB GPU.
        "--mem-fraction-static", "0.85",
    ]  # fmt: skip
    if CONFIG.fast_boot:
        # No CUDA graph capture: faster boot, slower generation.
        cmd += ["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"]
    if CONFIG.sglang_mtp:
        # SGLang turns NEXTN + a Gemma 4 "assistant" draft into its MTP mode.
        # 4 draft steps, one candidate per step: the same as vLLM's
        # `num_speculative_tokens: 4`, so both engines speculate equally far.
        cmd += [
            "--speculative-algorithm", "NEXTN",
            "--speculative-draft-model-path", CONFIG.speculative_model_name,
            "--speculative-draft-model-revision", CONFIG.speculative_model_revision,
            "--speculative-num-steps", "4",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "5",
        ]  # fmt: skip
    return cmd


@app.server(
    image=image,
    gpu=f"{CONFIG.gpu_type}:{CONFIG.n_gpu}",
    port=CONFIG.vllm_port,
    routing_region=CONFIG.routing_region,
    volumes={HF_CACHE_DIR: hf_cache},
    secrets=[hf_secret],
    # Modal proxy auth, as for vLLM: no proxy token, no access, no GPU start.
    unauthenticated=False,
    scaledown_window=15 * MINUTES,
    # SGLang captures CUDA graphs for the main and the draft model at boot.
    startup_timeout=20 * MINUTES,
    target_concurrency=100,
    # At most one GPU container. The autoscaler treats target_concurrency as a
    # soft target and started a 2nd (cold) container at 64 concurrent requests,
    # which splits benchmark load across GPUs. One container also caps the bill.
    max_containers=1,
)
class Server:
    """One SGLang process per container, started and stopped with the container."""

    process: subprocess.Popen[bytes]

    @modal.enter()
    def start(self) -> None:
        """Launch SGLang. Modal routes traffic once its port is open."""
        cmd = sglang_command()
        print(*cmd)  # visible in the container logs, useful for debugging
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self) -> None:
        """Stop SGLang cleanly when the container shuts down."""
        self.process.terminate()
