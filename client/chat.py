"""Terminal chat with the vLLM server, plus the client helpers shared by the UI and the test.

Usage::

    make chat                                                    # interactive chat
    uv run --env-file .env python -m client.chat "Explain SVD"   # one question, then exit
    uv run --env-file .env python -m client.chat --url https://...-dev.us-east.modal.direct
    uv run --env-file .env python -m client.chat --engine sglang     # the SGLang server

The server speaks the OpenAI API, so this client uses the official ``openai``
SDK with ``base_url`` pointed at the server instead of at api.openai.com.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Literal

import modal
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from server.config import CONFIG

# Message list in the OpenAI chat format: [{"role": ..., "content": ...}, ...].
Messages = list[ChatCompletionMessageParam]

# Engine -> name of the Modal app that serves it.
ENGINES = {"vllm": CONFIG.modal_app_name, "sglang": CONFIG.sglang_modal_app_name}

# ANSI escape codes: gray text for the model's thinking, then back to normal.
GRAY, RESET = "\033[90m", "\033[0m"


def connect(url: str | None = None, engine: str = "vllm") -> OpenAI:
    """Find the server, wait until it has booted, and return an authenticated client.

    Args:
        url: Server base URL, e.g. from ``modal serve``. If ``None``, the URL of
            the deployed app of ``engine`` is looked up on Modal.
        engine: ``"vllm"`` or ``"sglang"``: picks the app to look up.

    Returns:
        An ``OpenAI`` client whose requests go to the server.

    Exits:
        If ``MODAL_PROXY_TOKEN`` is not set, if no deployed app exists, or if the
        server does not boot in time. These are setup errors for a person to
        fix, so a short message is clearer than a traceback.
    """
    token = proxy_token()
    url = (url or deployed_url(ENGINES[engine])).rstrip("/")
    wait_until_ready(url, token)
    # The SDK sends `Authorization: Bearer <token>` on every request, which is
    # how Modal's proxy accepts a proxy token. Both engines serve /v1 routes.
    return OpenAI(base_url=f"{url}/v1", api_key=token)


def proxy_token() -> str:
    """Return the Modal proxy token (``wk-....ws-...``) from the environment, or exit.

    The servers run with Modal proxy auth: Modal rejects requests without a
    valid token with 401 before they reach a container, so strangers who know
    the URL can neither use the model nor start a GPU.
    """
    token = os.environ.get("MODAL_PROXY_TOKEN")
    if not token:
        sys.exit(
            "MODAL_PROXY_TOKEN is not set. Run `make proxy-token` and use the make targets."
        )
    return token


def deployed_url(app_name: str = CONFIG.modal_app_name) -> str:
    """Return the public URL of the deployed ``Server`` of ``app_name``, or exit if there is none."""
    server = modal.Server.from_name(app_name, "Server")
    url = server.get_url()
    if not url:
        sys.exit(
            "No deployed server found. Run `make deploy-vllm` / `make deploy-sglang` or pass --url."
        )
    return url


def wait_until_ready(url: str, token: str, timeout: float = 15 * 60) -> None:
    """Block until ``GET /health`` returns 200, printing a dot per attempt.

    Any request to the URL makes Modal start a container if none is running.
    Until vLLM listens on its port, Modal answers 503. A cold start takes a
    few minutes: load the weights, compile, capture CUDA graphs.

    Args:
        url: Server base URL, without a trailing slash.
        token: Modal proxy token. Without it Modal answers 401 to every request.
        timeout: Seconds to wait before giving up. Must cover a cold start.
    """
    deadline = time.time() + timeout
    print(f"Waiting for {url} ...", end="", flush=True)
    while time.time() < deadline:
        try:
            # urlopen raises HTTPError (a URLError subclass) on any non-2xx status.
            request = urllib.request.Request(
                f"{url}/health", headers={"Authorization": f"Bearer {token}"}
            )
            with urllib.request.urlopen(request, timeout=60):
                print(" ready.")
                return
        except urllib.error.HTTPError as e:
            # 401: Modal rejected the token. Retrying cannot fix that.
            if e.code == 401:
                sys.exit("\nModal rejected MODAL_PROXY_TOKEN (401). Check it in .env.")
            print(".", end="", flush=True)
            time.sleep(2)
        except (urllib.error.URLError, TimeoutError):
            print(".", end="", flush=True)
            time.sleep(2)
    sys.exit("\nServer did not become ready in time.")


def stream_reply(
    client: OpenAI, messages: Messages, thinking: bool
) -> Iterator[tuple[Literal["reasoning", "content"], str]]:
    """Send a chat request and yield the reply piece by piece as it streams in.

    Args:
        client: Client returned by ``connect``.
        messages: Conversation so far, ending with the user's message.
        thinking: Whether the model reasons before it answers. Off is faster.

    Yields:
        ``("reasoning", text)`` for the model's thinking and ``("content", text)``
        for the answer. Each piece is a few tokens long.
    """
    stream = client.chat.completions.create(
        # Must match `--served-model-name` on the server.
        model=CONFIG.model_name,
        messages=messages,
        stream=True,
        # Fields outside the OpenAI API go in `extra_body`. vLLM passes
        # `chat_template_kwargs` to Gemma's chat template, which reads
        # `enable_thinking`.
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    for chunk in stream:
        # Some chunks (e.g. a final usage report) carry no choices.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # vLLM sends the thinking in a field the OpenAI SDK does not declare.
        # Its name depends on the vLLM version, so check both.
        reasoning = getattr(delta, "reasoning", None) or getattr(
            delta, "reasoning_content", None
        )
        if reasoning:
            yield "reasoning", reasoning
        if delta.content:
            yield "content", delta.content


def ask(client: OpenAI, messages: Messages, thinking: bool = True) -> str:
    """Print the reply as it streams, thinking in gray, and return only the answer.

    Args:
        client: Client returned by ``connect``.
        messages: Conversation so far, ending with the user's message.
        thinking: Whether the model reasons before it answers.

    Returns:
        The answer text, without the thinking. This is what goes back into the
        chat history.
    """
    answer: list[str] = []
    for kind, text in stream_reply(client, messages, thinking):
        if kind == "reasoning":
            print(f"{GRAY}{text}{RESET}", end="", flush=True)
        else:
            answer.append(text)
            print(text, end="", flush=True)
    print()
    return "".join(answer)


def main() -> None:
    """Parse the command line, then ask one question or run an interactive chat."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "prompt", nargs="?", help="ask one question and exit; omit for interactive chat"
    )
    parser.add_argument("--url", help="server URL (default: look up the deployed app)")
    parser.add_argument(
        "--engine", choices=ENGINES, default="vllm", help="which server to talk to"
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.", help="system prompt"
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="let the model reason before it answers",
    )
    args = parser.parse_args()

    client = connect(args.url, args.engine)
    messages: Messages = [{"role": "system", "content": args.system}]

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        ask(client, messages, args.thinking)
        return

    print("Chat started. Empty line or Ctrl+D to quit. Thinking is shown in gray.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        messages.append({"role": "user", "content": user})
        print("llm> ", end="")
        # The model is stateless: every request resends the whole conversation.
        # Only the answer goes into the history; the thinking is not sent back.
        answer = ask(client, messages, args.thinking)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
