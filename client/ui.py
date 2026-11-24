"""Browser chat UI for the vLLM server, built with Gradio.

Usage::

    make ui
    uv run --env-file .env python -m client.ui --url https://...-dev.us-east.modal.direct

Waits for the server to boot (a few minutes when cold), then opens
http://127.0.0.1:7860. The UI runs on your machine; only the model runs on Modal.
"""

import argparse
from collections.abc import Iterator
from typing import Any

import gradio as gr
from openai import OpenAI

from client.chat import ENGINES, Messages, connect, stream_reply
from server.config import CONFIG

# Title of the collapsible bubble that shows the model's thinking. Also used to
# recognize those bubbles in the history, so they are not sent back to the model.
THINKING_TITLE = "Thinking"


def text_of(content: Any) -> str:
    """Extract plain text from a Gradio message's ``content``.

    Gradio stores content either as a string or as a list of blocks such as
    ``{"type": "text", "text": "..."}``. Non-text blocks (files, images) are
    ignored, because the server is text-only.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(text_of(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return ""


def to_openai(history: list[dict[str, Any]], message: str, system: str) -> Messages:
    """Convert Gradio's chat history plus the new message to OpenAI messages.

    Args:
        history: Previous messages, as Gradio passes them to the chat function.
        message: The text the user just sent.
        system: The system prompt from the UI.

    Returns:
        ``[system, ...previous user/assistant messages..., new user message]``.
        Thinking bubbles are left out: they are for display only, and sending
        them back would fill the context window for no benefit.
    """
    messages: Messages = [{"role": "system", "content": system}]
    for msg in history:
        if (msg.get("metadata") or {}).get("title") == THINKING_TITLE:
            continue
        text = text_of(msg["content"])
        # Build each role explicitly, so the type checker can verify the shape.
        if msg["role"] == "user":
            messages.append({"role": "user", "content": text})
        elif msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": text})
    messages.append({"role": "user", "content": message})
    return messages


def build_ui(client: OpenAI) -> gr.ChatInterface:
    """Create the chat UI. ``client`` is passed in so tests can use a fake server."""

    def respond(
        message: str, history: list[dict[str, Any]], system: str, thinking: bool
    ) -> Iterator[list[gr.ChatMessage]]:
        """Stream the reply into the chat window.

        Gradio calls this for every user message. Each ``yield`` replaces the
        reply shown so far, so the text appears to grow as tokens arrive.
        ``system`` and ``thinking`` come from ``additional_inputs`` below.
        """
        reasoning, answer = "", ""
        messages = to_openai(history, message, system)
        for kind, text in stream_reply(client, messages, thinking):
            if kind == "reasoning":
                reasoning += text
            else:
                answer += text
            reply: list[gr.ChatMessage] = []
            if reasoning:
                # A `title` in metadata makes Gradio show a collapsible bubble.
                reply.append(
                    gr.ChatMessage(
                        content=reasoning, metadata={"title": THINKING_TITLE}
                    )
                )
            if answer:
                reply.append(gr.ChatMessage(content=answer))
            yield reply

    return gr.ChatInterface(
        respond,
        title=CONFIG.model_name,
        description="Served by vLLM on Modal.",
        # Extra controls, passed to `respond` in this order after `history`.
        additional_inputs=[
            gr.Textbox("You are a helpful assistant.", label="System prompt"),
            gr.Checkbox(True, label="Show model thinking (slower)"),
        ],
        examples=[
            ["Explain the singular value decomposition."],
            ["Write a haiku about GPUs."],
        ],
        # Clicking an example only fills the input box. Running it would wake
        # the GPU server just because someone browsed the examples.
        run_examples_on_click=False,
    )


def main() -> None:
    """Connect to the server, then start the UI and open it in the browser."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", help="server URL (default: look up the deployed app)")
    parser.add_argument(
        "--engine", choices=ENGINES, default="vllm", help="which server to talk to"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        # The UI holds the proxy token, so a share link gives GPU access without it.
        help="create a public gradio.live link (anyone with it can use your GPU)",
    )
    args = parser.parse_args()

    build_ui(connect(args.url, args.engine)).launch(share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
