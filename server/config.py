"""Load ``config/vllm-inference.yaml``, the single source of settings.

The server, the clients and the smoke test all import ``CONFIG`` from here, so a
setting is changed in one place only.

The file is read from two places:

* Locally (``modal deploy``, ``make chat``, ...) it is read from the repo.
* Inside the Modal container it is read from the copy that
  ``server/vllm_inference.py`` bakes into the image at ``REMOTE_PATH``.
  The container has no access to your local disk, so it needs its own copy.
"""

from pathlib import Path
from typing import TypedDict, cast

import modal
import yaml


class Config(TypedDict):
    """Shape of the YAML file. Keep in sync with ``config/vllm-inference.yaml``."""

    # Name of the Modal app. Also used by clients to find the deployed server.
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


LOCAL_PATH = Path(__file__).resolve().parents[1] / "config" / "vllm-inference.yaml"
REMOTE_PATH = "/root/config/vllm-inference.yaml"


def load_config(path: Path) -> Config:
    """Read the YAML file and check that every key in ``Config`` is present.

    Args:
        path: Location of the YAML file.

    Returns:
        The parsed settings.

    Raises:
        KeyError: If the file is missing one or more required keys. Failing
            here gives a clear message, instead of a ``KeyError`` deep inside
            the server after a multi-minute container boot.
    """
    data = yaml.safe_load(path.read_text())
    missing = Config.__required_keys__ - data.keys()
    if missing:
        raise KeyError(f"{path} is missing keys: {sorted(missing)}")
    # The key check above makes this cast safe for presence. Value types are
    # trusted to match the YAML; mypy cannot check the contents of a file.
    return cast(Config, data)


# `modal.is_local()` is True on your machine and False inside a Modal container.
CONFIG = load_config(LOCAL_PATH if modal.is_local() else Path(REMOTE_PATH))
