"""All settings for the server, the clients and the smoke test. Edit them here."""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: settings cannot be changed while the program runs
class Config:
    # Name of the Modal app. Clients also use it to find the deployed server.
    modal_app_name: str
    # Hugging Face repo id of the model, and the exact commit to load.
    # Pinning the revision keeps a repo update from changing the server silently.
    model_name: str
    model_revision: str
    # Small "draft" model used for speculative decoding, also pinned.
    speculative_model_name: str
    speculative_model_revision: str
    # True: skip torch.compile and CUDA graph capture. Faster boot, slower generation.
    fast_boot: bool
    # Modal GPU type (e.g. "H200", "A100-80GB") and how many per container.
    gpu_type: str
    n_gpu: int
    # Port vLLM listens on inside the container. Modal forwards traffic to it.
    vllm_port: int
    # Modal region that receives requests (e.g. "us-east").
    routing_region: str


CONFIG = Config(
    modal_app_name="example-vllm-inference",
    model_name="google/gemma-4-26B-A4B-it",
    model_revision="47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301",
    speculative_model_name="google/gemma-4-26B-A4B-it-assistant",
    speculative_model_revision="f188f476dc11dd5bb3014dc861529d316bce49d3",
    fast_boot=False,
    gpu_type="A100-80GB",
    n_gpu=1,
    vllm_port=8000,
    routing_region="us-east",
)
