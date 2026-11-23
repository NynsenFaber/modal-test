"""Browser chat UI for the vLLM server, built with Gradio.

    make ui                                                   # = uv run --env-file .env python -m client.ui
    uv run --env-file .env python -m client.ui --url https://...-dev.us-east.modal.direct

Waits for the server to boot (cold start: a few minutes), then opens http://127.0.0.1:7860.
"""

import argparse
from collections.abc import Iterator

import gradio as gr
from openai import OpenAI

from client.chat import CONFIG, connect, stream_reply

THINKING_TITLE = "Thinking"


def text_of(content) -> str:
    """Gradio may store message content as a string or as a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(text_of(part) for part in content)
    if isinstance(content, dict):
        return content.get("text", "")
    return ""


def to_openai(history: list[dict], message: str, system: str) -> list[dict]:
    """Gradio history -> OpenAI messages. Thinking bubbles are UI-only and not sent back."""
    messages = [{"role": "system", "content": system}]
    for msg in history:
        if (msg.get("metadata") or {}).get("title") == THINKING_TITLE:
            continue
        messages.append({"role": msg["role"], "content": text_of(msg["content"])})
    messages.append({"role": "user", "content": message})
    return messages


def build_ui(client: OpenAI) -> gr.ChatInterface:
    def respond(message: str, history: list[dict], system: str, thinking: bool) -> Iterator[list]:
        reasoning, answer = "", ""
        for kind, text in stream_reply(client, to_openai(history, message, system), thinking):
            if kind == "reasoning":
                reasoning += text
            else:
                answer += text
            reply = []
            if reasoning:
                # Shown as a collapsible "Thinking" bubble above the answer.
                reply.append(gr.ChatMessage(content=reasoning, metadata={"title": THINKING_TITLE}))
            if answer:
                reply.append(gr.ChatMessage(content=answer))
            yield reply

    return gr.ChatInterface(
        respond,
        title=CONFIG["model_name"],
        description="Served by vLLM on Modal.",
        additional_inputs=[
            gr.Textbox("You are a helpful assistant.", label="System prompt"),
            gr.Checkbox(True, label="Show model thinking (slower)"),
        ],
        examples=[["Explain the singular value decomposition."], ["Write a haiku about GPUs."]],
        run_examples_on_click=False,  # don't call the GPU server just to render examples
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="server URL (default: look up the deployed app)")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link (anyone with it can use your GPU)")
    args = parser.parse_args()

    build_ui(connect(args.url)).launch(share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
