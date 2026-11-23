"""Chat with the deployed vLLM server through the OpenAI SDK.

    uv run python scripts/chat.py                      # interactive chat
    uv run python scripts/chat.py "Explain SVD"        # one question, then exit
    uv run python scripts/chat.py --url https://...-dev.us-east.modal.direct   # e.g. a `modal serve` URL
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal
import yaml
from openai import OpenAI

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "vllm-inference.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
GRAY, RESET = "\033[90m", "\033[0m"


def server_url(url: str | None) -> str:
    if url:
        return url.rstrip("/")
    app_name = CONFIG["modal_app_name"]
    found = modal.Server.from_name(app_name, "Server").get_url()
    if not found:
        sys.exit(f"No deployed server found for app '{app_name}'. Run `make deploy` or pass --url.")
    return found.rstrip("/")


def wait_until_ready(url: str, timeout: float = 15 * 60) -> None:
    # Modal answers 503 until a container has booted (a cold start takes a few minutes).
    deadline = time.time() + timeout
    print(f"Waiting for {url} ...", end="", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=60):
                print(" ready.")  # urlopen raises on any non-2xx status
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    sys.exit("\nServer did not become ready in time.")


def ask(client: OpenAI, messages: list[dict], thinking: bool) -> str:
    stream = client.chat.completions.create(
        model=CONFIG["model_name"],  # must match --served-model-name
        messages=messages,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    answer = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # vLLM sends the model's thinking in a non-standard `reasoning` field
        reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning:
            print(f"{GRAY}{reasoning}{RESET}", end="", flush=True)
        if delta.content:
            answer.append(delta.content)
            print(delta.content, end="", flush=True)
    print()
    return "".join(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?", help="ask one question and exit; omit for interactive chat") # postionally required argument
    parser.add_argument("--url", help="server URL (default: look up the deployed app)")
    parser.add_argument("--system", default="You are a helpful assistant.", help="system prompt")
    parser.add_argument("--no-thinking", action="store_true", help="skip the model's reasoning step")
    args = parser.parse_args()

    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        sys.exit("VLLM_API_KEY is not set. Run `make api-key`, then use `make chat`.")

    url = server_url(args.url)
    wait_until_ready(url)
    client = OpenAI(base_url=f"{url}/v1", api_key=api_key)  # sent as `Authorization: Bearer <key>`

    messages = [{"role": "system", "content": args.system}]
    thinking = not args.no_thinking

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        ask(client, messages, thinking)
        return

    print("Chat started. Empty line or Ctrl+D to quit. Thinking shown in gray.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        messages.append({"role": "user", "content": user})
        print("llm> ", end="")
        # keep only the final answer in history; the thinking is not sent back
        messages.append({"role": "assistant", "content": ask(client, messages, thinking)})


if __name__ == "__main__":
    main()
