"""Model weights shared by both servers: the cache Volume and the command that fills it.

Usage::

    make download   # once, before the first deploy of either server

Both ``vllm_inference`` and ``sglang_inference`` mount the same
``huggingface-cache`` Volume, so the weights are downloaded once, on a cheap CPU
container, and every GPU boot reads them from the Volume.
"""

import modal

from server.config import CONFIG

MINUTES = 60  # seconds; Modal timeouts are given in seconds

# Volumes persist between container runs. Without this one, every cold start
# would download ~50 GB of weights.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
# Default Hugging Face cache location in both server images. Mounting the Volume
# here means no extra configuration in the servers.
HF_CACHE_DIR = "/root/.cache/huggingface"

# HF_TOKEN: needed because Gemma is a gated model on Hugging Face.
hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App("gemma-weights")

# Only the Hugging Face client: no CUDA, no inference engine.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]")
    .env({"HF_HOME": HF_CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
)


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
    # Imported here because `huggingface_hub` is installed in the image only.
    from huggingface_hub import snapshot_download

    snapshot_download(CONFIG.model_name, revision=CONFIG.model_revision)
    snapshot_download(
        CONFIG.speculative_model_name, revision=CONFIG.speculative_model_revision
    )
    # Save the new files now. Otherwise the Volume is only saved at container exit.
    hf_cache.commit()
