# # Run OpenAI-compatible LLM inference with Gemma and vLLM

# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.

# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# This has complicated their interface far beyond "text-in, text-out".
# OpenAI's API has emerged as a standard for that interface,
# and it is supported by open source LLM serving frameworks like [vLLM](https://docs.vllm.ai/en/latest/).

# This example is intended to demonstrate the basics of deploying LLM inference on Modal.
# For more on how to optimize performance, see
# [this guide](https://modal.com/docs/guide/high-performance-llm-inference)
# and check out our
# [LLM Engineer's Almanac](https://modal.com/llm-almanac).

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# vLLM can be installed with `uv pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

import json
from pathlib import Path

import modal
import yaml

# Configuration lives in config/vllm-inference.yaml.
# Locally we read it from the repo; in the container we read the copy baked into the image.
LOCAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "vllm-inference.yaml"
REMOTE_CONFIG_PATH = "/root/vllm-inference.yaml"

with open(LOCAL_CONFIG_PATH if modal.is_local() else REMOTE_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            "VLLM_LOG_STATS_INTERVAL": "1",  # more frequent metrics logging
        }
    )
    .add_local_file(LOCAL_CONFIG_PATH, REMOTE_CONFIG_PATH)
)

# ## Download the model weights

# We'll be running a pretrained foundation model --
# [Google's Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/).
# It can also take images, video, and audio as inputs,
# though we won't use that here.

# We'll use the 26BA4B variant, [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).
# This variant is trained with reasoning capabilities, which allow it to
# enhance the quality of its generated responses.
# It has `26B`illion parameters, of which `4B`illion are `A`ctive
# in processing of each token.

# You can swap this model out for another by changing `model_name` in the config file,
# though you might also need to adjust some of the server configuration as well.
# A single H200 GPU has enough VRAM to store this 26,000,000,000 parameter model
# along with a large KV cache.


MODEL_NAME = config["model_name"]
MODEL_REVISION = config["model_revision"]

# Although vLLM will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk.
# For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# We'll also cache some of vLLM's JIT compilation artifacts in a Modal Volume.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Configuring vLLM

# ### Trading off fast boots and token generation performance

# vLLM has embraced dynamic and just-in-time compilation to eke out additional performance without having to write too many custom kernels,
# e.g. via the Torch compiler and CUDA graph capture.
# These compilation features incur latency in exchange for lowered latency and higher throughput during generation.
# This latency is typically tens of seconds to a few minutes, reduced to about ten seconds when loaded from the cache.
# We make this trade-off controllable with the `fast_boot` config value.

FAST_BOOT = config["fast_boot"]

# If you're running an LLM service that frequently scales from 0 (frequent ["cold starts"](https://modal.com/docs/guide/cold-start))
# you might want to set this to `True`, or consider [GPU memory snapshots](https://modal.com/docs/guide/memory-snapshots).
# It's also useful to set this when you're iterating on the server configuration.

# If you're running an LLM service that usually has multiple replicas running, then set this to `False` for improved performance.

# See the code below for details on the parameters that `FAST_BOOT` controls.

# ### Model-specific configuration

# Almost all models require some amount of configuration via command-line flags,
# especially to achieve optimal performance.

# We set these flags in the code below, roughly following the
# [usage guide from the vLLM docs](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html).

# For instance, we turn off multimodal features to save on [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram),
# and we activate the [built-in multi-token prediction (MTP)](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
# speculative decoding for improved throughput at lower concurrencies.

SPECULATIVE_MODEL_NAME = config["speculative_model_name"]
SPECULATIVE_MODEL_REVISION = config["speculative_model_revision"]

# For more on the performance you can expect when serving your own LLMs, see
# [our LLM engine performance benchmarks](https://modal.com/llm-almanac).

# ## Build a vLLM engine and serve it

# The class below spawns a vLLM instance listening at port 8000, serving requests to our model.
# We wrap it in the [`@app.server`](https://modal.com/docs/guide/servers) decorator
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`, and only starts accepting requests
# once the model is spun up and the process is ready to listen on the configured port.


app = modal.App(config["modal_app_name"])

N_GPU = config["n_gpu"]
GPU = config["gpu_type"]
MINUTES = 60  # seconds
VLLM_PORT = config["vllm_port"]
ROUTING_REGION = config["routing_region"]


# Download the weights once, on CPU and with a long timeout, so server boots
# only read them from the Volume. Run with `make download`.
@app.function(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * MINUTES,
)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME, revision=MODEL_REVISION)
    snapshot_download(SPECULATIVE_MODEL_NAME, revision=SPECULATIVE_MODEL_REVISION)
    hf_cache_vol.commit()


@app.server(
    image=vllm_image,
    gpu=f"{GPU}:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    startup_timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    routing_region=ROUTING_REGION,
    target_concurrency=100,  # how many requests can one replica handle? tune carefully!
    unauthenticated=True,  # no Modal auth; vLLM checks the API key instead
    # vLLM reads VLLM_API_KEY from the environment and then requires
    # `Authorization: Bearer <key>` on /v1/* routes (/health stays open).
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("vllm-api-key"),
    ],
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=info",
            "--async-scheduling",
        ]

        # enforce-eager disables both Torch compilation and CUDA graph capture
        # default is no-enforce-eager. see the --compilation-config flag for tighter control
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

        # assume multiple GPUs are for splitting up large matrix multiplications
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        # add model-specific configuration
        cmd += [
            # skip multimedia support, just language
            "--limit-mm-per-prompt",
            json.dumps({"image": 0, "video": 0, "audio": 0}),
            # enable reasoning and tool use
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "gemma4",
            "--tool-call-parser",
            "gemma4",
        ]

        # add speculative decoding
        cmd += [
            "--speculative-config",
            json.dumps(
                {
                    "model": SPECULATIVE_MODEL_NAME,
                    "revision": SPECULATIVE_MODEL_REVISION,
                    "num_speculative_tokens": 4,
                }
            ),
        ]

        print(*cmd)

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Deploy the server

# ```bash
# make deploy        # = modal deploy server/vllm_inference.py
# ```

# ## Interact with the server

# Chat with it through the OpenAI SDK with `make chat` (see `client/chat.py`).
# Smoke-test a temporary copy of the server with `make test` (see `tests/smoke_test.py`).
