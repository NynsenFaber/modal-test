"""Smoke test: start a temporary copy of the server on Modal and check that it answers.

Usage::

    make test
    uv run --env-file .env modal run -m tests.smoke_test --content "Hi" --no-twice

``modal run`` creates a temporary app with its own URL, separate from the
deployed one. The app stops when this function returns. Expect a cold start of
a few minutes and the GPU cost of that time.

What a pass proves: the image builds, vLLM boots with the configured flags,
the proxy token is accepted, and the model streams a non-empty answer.
"""

from client.chat import ask, connect
from server.vllm_inference import Server, app

# Two different personas make it easy to see that the system prompt is applied.
SYSTEM_PROMPTS = [
    "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    "You are Jar Jar Binks.",
]


@app.local_entrypoint()
def main(
    content: str = "Explain the singular value decomposition.", twice: bool = True
) -> None:
    """Ask ``content`` once per system prompt and fail if any answer is empty.

    ``@app.local_entrypoint`` runs this function on your machine while the
    ``Server`` runs on Modal. Modal turns the parameters into CLI flags:
    ``--content``, and ``--twice`` / ``--no-twice``.
    """
    # Server.get_url() is the temporary app's URL. If the proxy token is wrong,
    # the first request below fails with 401 AuthenticationError.
    # `@app.server` adds `get_url` at runtime, but Modal's type hints (1.6.4)
    # still describe the plain class, hence the ignore.
    client = connect(Server.get_url())  # type: ignore[attr-defined]

    for system in SYSTEM_PROMPTS[: 2 if twice else 1]:
        print(f"\n--- system: {system}\n--- user: {content}\n")
        answer = ask(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        assert answer.strip(), "server returned an empty answer"

    print("\nSmoke test passed.")
