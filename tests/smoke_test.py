"""Smoke test: start a temporary copy of the server on Modal, wait for it, ask a question.

    make test                                                    # = uv run --env-file .env modal run -m tests.smoke_test
    uv run --env-file .env modal run -m tests.smoke_test --content "Hi" --no-twice

The temporary app stops when the test ends. Expect a cold start of a few minutes.
"""

import os

from openai import OpenAI

from client.chat import ask, wait_until_ready
from server.vllm_inference import Server, app

SYSTEM_PROMPTS = [
    "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    "You are Jar Jar Binks.",
]


@app.local_entrypoint()
def main(content: str = "Explain the singular value decomposition.", twice: bool = True):
    url = Server.get_url().rstrip("/")
    wait_until_ready(url)
    # A wrong or missing key makes the request below fail with 401.
    client = OpenAI(base_url=f"{url}/v1", api_key=os.environ["VLLM_API_KEY"])

    for system in SYSTEM_PROMPTS[: 2 if twice else 1]:
        print(f"\n--- system: {system}\n--- user: {content}\n")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        answer = ask(client, messages, thinking=True)
        assert answer.strip(), "server returned an empty answer"

    print("\nSmoke test passed.")
