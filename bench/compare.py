"""Benchmark the vLLM and the SGLang server on the same load and write a report.

Usage::

    make compare         # deploy both servers, full sweep (about 15-25 minutes)
    make compare-quick   # 2 concurrency levels, chat scenario only
    uv run --env-file .env modal run -m bench.compare --quick

How it works:

1. Both servers are looked up as deployed apps (``make deploy-vllm``, ``make deploy-sglang``).
2. ``run_load`` runs on a Modal CPU container in the servers' region, so home
   network latency does not end up in the numbers. It sends ``n`` streaming
   requests with at most ``concurrency`` in flight and times every chunk.
3. This machine turns the timings into metrics and writes
   ``reports/<timestamp>/report.md`` with charts and ``results.json`` with the raw data.

Every request asks for exactly ``output_tokens`` tokens (``ignore_eos``) with
thinking off, and starts with a random id so no request reuses another's
prefix cache. Both engines therefore do the same work.
"""

import asyncio
import json
import math
import os
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import modal
import openai

from client.chat import ENGINES, deployed_url, proxy_token, wait_until_ready
from server.config import CONFIG

MINUTES = 60

# Modal's on-demand price for the GPU in `CONFIG.gpu_type` (A100-80GB), USD per
# hour. Check https://modal.com/pricing and update it when the GPU changes.
GPU_PRICE_PER_HOUR = 2.50

# A request is "good" if it meets both limits: the reply starts within 1 s and
# streams faster than 20 tokens/s, which is faster than people read.
SLO_TTFT_S = 1.0
SLO_TPOT_S = 0.050

# Scenario name -> (prompt length in words, output tokens).
# "chat" is decode-heavy; "long-prompt" stresses prefill, so it shows in TTFT.
SCENARIOS = {"chat": (200, 256), "long-prompt": (3000, 256)}
CONCURRENCY = [1, 4, 16, 64]  # below target_concurrency=100: one container only

# Report label -> engine key in client.chat.ENGINES (Modal app name).
LABELS = {
    "vLLM": "vllm",
    "SGLang" + ("" if CONFIG.sglang_mtp else " (no MTP)"): "sglang",
}

app = modal.App("gemma-bench")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("openai>=2.0")
    .add_local_python_source("client", "server")
)

# Filler for prompts. The content does not matter, only its length.
WORDS = ["the", "of", "and", "to", "in", "is", "was", "for", "on", "that", "with"]


async def one_request(
    client: openai.AsyncOpenAI, sem: asyncio.Semaphore, words: int, output_tokens: int
) -> dict[str, Any]:
    """Send one streaming request and return its timings, in seconds from its start."""
    prompt = f"{uuid.uuid4()} " + " ".join(random.choices(WORDS, k=words))
    prompt += "\n\nWrite a long story inspired by the words above."
    chunks: list[float] = []  # arrival time of every chunk that carries text
    usage = None
    error = None
    async with sem:
        start = time.perf_counter()
        try:
            stream = await client.chat.completions.create(
                model=CONFIG.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_tokens,
                stream=True,
                # The last chunk reports the exact token counts.
                stream_options={"include_usage": True},
                extra_body={
                    # Generate exactly max_tokens on both engines.
                    "ignore_eos": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    # Thinking is off, but count it if an engine sends some anyway.
                    reasoning = getattr(delta, "reasoning", None) or getattr(
                        delta, "reasoning_content", None
                    )
                    if delta.content or reasoning:
                        chunks.append(time.perf_counter() - start)
        except openai.APIError as e:
            error = repr(e)
        end = time.perf_counter() - start
    return {
        "chunks": chunks,
        "end": end,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "error": error,
    }


@app.function(
    image=image,
    region=CONFIG.routing_region,
    # Copies MODAL_PROXY_TOKEN from this machine's environment into the container.
    secrets=[modal.Secret.from_local_environ(["MODAL_PROXY_TOKEN"])],
    timeout=60 * MINUTES,
)
async def run_load(
    url: str,
    concurrency: int,
    n_requests: int,
    words: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Send ``n_requests`` requests, ``concurrency`` at a time, and return all timings."""
    client = openai.AsyncOpenAI(
        base_url=f"{url}/v1",
        api_key=os.environ["MODAL_PROXY_TOKEN"],  # sent as the Bearer token
        timeout=10 * MINUTES,
        max_retries=0,  # a retry would hide an error and distort the timings
    )
    sem = asyncio.Semaphore(concurrency)
    start = time.perf_counter()
    requests = await asyncio.gather(
        *(one_request(client, sem, words, output_tokens) for _ in range(n_requests))
    )
    return {"wall": time.perf_counter() - start, "requests": requests}


# --- Metrics -----------------------------------------------------------------


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile; NaN for an empty list."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def summarize(run: dict[str, Any]) -> dict[str, float]:
    """Turn the raw timings of one load level into the metrics of the report."""
    reqs = run["requests"]
    ok = [r for r in reqs if not r["error"] and r["chunks"] and r["output_tokens"] > 1]
    ttft = [r["chunks"][0] for r in ok]
    ttlt = [r["end"] for r in ok]
    # Per token, not per chunk: with speculative decoding a chunk holds several tokens.
    tpot = [(r["end"] - r["chunks"][0]) / (r["output_tokens"] - 1) for r in ok]
    gaps = [
        b - a for r in ok for a, b in zip(r["chunks"], r["chunks"][1:], strict=False)
    ]
    good = sum(
        t < SLO_TTFT_S and p < SLO_TPOT_S for t, p in zip(ttft, tpot, strict=True)
    )
    tps = sum(r["output_tokens"] for r in ok) / run["wall"]
    return {
        "ttft_p50": pct(ttft, 50) * 1000,
        "ttft_p95": pct(ttft, 95) * 1000,
        "tpot_p50": pct(tpot, 50) * 1000,
        "tpot_p95": pct(tpot, 95) * 1000,
        "ttlt_p50": pct(ttlt, 50),
        "ttlt_p95": pct(ttlt, 95),
        "itl_p99": pct(gaps, 99) * 1000,
        "tps": tps,
        "user_tps": 1 / pct(tpot, 50) if tpot else float("nan"),
        "rps": len(ok) / run["wall"],
        "goodput": 100 * good / len(reqs),
        "errors": 100 * (len(reqs) - len(ok)) / len(reqs),
        "cost_per_mtok": GPU_PRICE_PER_HOUR / (tps * 3600) * 1e6
        if tps
        else float("nan"),
        "input_tokens": pct([r["input_tokens"] for r in ok], 50),
        "output_tokens": pct([r["output_tokens"] for r in ok], 50),
    }


# key -> (label, higher is better)
METRICS = {
    "ttft_p50": ("TTFT p50 (ms)", False),
    "ttft_p95": ("TTFT p95 (ms)", False),
    "tpot_p50": ("TPOT p50 (ms)", False),
    "tpot_p95": ("TPOT p95 (ms)", False),
    "ttlt_p50": ("TTLT p50 (s)", False),
    "ttlt_p95": ("TTLT p95 (s)", False),
    "itl_p99": ("ITL p99, gap between chunks (ms)", False),
    "tps": ("TPS, total (tok/s)", True),
    "user_tps": ("TPS, per user (tok/s)", True),
    "rps": ("Requests/s", True),
    "goodput": ("Goodput (% in SLO)", True),
    "errors": ("Errors (%)", False),
    "cost_per_mtok": ("Cost per 1M output tokens ($)", False),
}

# Summary of one engine: scenario -> concurrency -> metrics.
Summary = dict[str, dict[int, dict[str, float]]]


def better(key: str, a: float, b: float, names: list[str]) -> str:
    """Name of the engine with the better value, or "tie" within 3 %."""
    if math.isnan(a) or math.isnan(b):
        return "-"
    if abs(a - b) <= 0.03 * max(abs(a), abs(b)):
        return "tie"
    a_wins = (a > b) == METRICS[key][1]
    return names[0] if a_wins else names[1]


# --- Report ------------------------------------------------------------------


def vs(
    key: str, a: dict[str, float], b: dict[str, float], names: list[str], unit: str
) -> str:
    """'winner (value a vs value b unit)' for one metric at one load level."""
    return (
        f"{better(key, a[key], b[key], names)} ({fmt(a[key])} vs {fmt(b[key])}{unit})"
    )


def plot(summaries: dict[str, Summary], scenario: str, path: Path) -> None:
    """Draw one scenario as six grouped bar charts: one group per load level."""
    import matplotlib

    matplotlib.use("Agg")  # no window, just a file
    import matplotlib.pyplot as plt

    panels = ["tps", "user_tps", "ttft_p50", "tpot_p50", "ttlt_p50", "goodput"]
    engines = list(summaries)
    cs = sorted(summaries[engines[0]][scenario])
    width = 0.8 / len(engines)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, key in zip(axes.flat, panels, strict=True):
        label, higher_is_better = METRICS[key]
        for i, engine in enumerate(engines):
            levels = summaries[engine][scenario]
            xs = [x + (i - (len(engines) - 1) / 2) * width for x in range(len(cs))]
            bars = ax.bar(xs, [levels[c][key] for c in cs], width, label=engine)
            ax.bar_label(bars, labels=[fmt(levels[c][key]) for c in cs], fontsize=8)
        ax.set_title(f"{label}: {'higher' if higher_is_better else 'lower'} is better")
        ax.set_xticks(range(len(cs)), [f"c={c}" for c in cs])
        ax.set_xlabel("concurrent requests")
        ax.margins(y=0.15)  # room for the value labels above the bars
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Gemma 4 on {CONFIG.n_gpu}x {CONFIG.gpu_type}: {scenario}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt(v: float) -> str:
    """Short number for tables: 3 significant digits, no decimals from 100 up."""
    if math.isnan(v):
        return "-"
    return f"{v:,.3g}" if abs(v) < 100 else f"{v:,.0f}"


def write_report(
    summaries: dict[str, Summary], ready_s: dict[str, float], out: Path
) -> None:
    """Write ``report.md`` with the setup, one chart and one table per scenario."""
    names = list(summaries)
    a, b = names
    mtp = "on" if CONFIG.sglang_mtp else "off"
    ready = ", ".join(f"{e} {s:.0f} s" for e, s in ready_s.items())
    lines = [
        f"# Gemma 4: {a} vs {b}",
        "",
        (
            f"- Model: `{CONFIG.model_name}` on {CONFIG.n_gpu}x {CONFIG.gpu_type}"
            f" (${GPU_PRICE_PER_HOUR}/h), region {CONFIG.routing_region}"
        ),
        (
            f"- Speculative decoding (`{CONFIG.speculative_model_name}`):"
            f" vLLM on, SGLang {mtp}"
        ),
        (
            "- Thinking off, fixed output length (`ignore_eos`), unique prompts"
            " (no prefix cache)"
        ),
        (
            f"- SLO for goodput: TTFT < {SLO_TTFT_S * 1000:.0f} ms"
            f" and TPOT < {SLO_TPOT_S * 1000:.0f} ms"
        ),
        (
            f"- Time until /health answered: {ready}"
            " (minutes = cold start, seconds = already warm)"
        ),
        "",
    ]
    for scenario in summaries[a]:
        levels_a, levels_b = summaries[a][scenario], summaries[b][scenario]
        cs = sorted(levels_a)
        lo, hi = cs[0], cs[-1]

        tokens_in = levels_a[lo]["input_tokens"]
        tokens_out = levels_a[lo]["output_tokens"]
        lines += [
            f"## Scenario: {scenario} (~{tokens_in:.0f} tokens in, {tokens_out:.0f} out)",
            "",
            f"![{scenario}]({scenario}.png)",
            "",
            f"**Verdict** ({a} vs {b})",
            "",
            (
                f"- One user (c={lo}): TTFT {vs('ttft_p50', levels_a[lo], levels_b[lo], names, ' ms')},"
                f" TPOT {vs('tpot_p50', levels_a[lo], levels_b[lo], names, ' ms')}"
            ),
            (
                f"- High load (c={hi}): throughput {vs('tps', levels_a[hi], levels_b[hi], names, ' tok/s')},"
                f" cost per 1M tokens {vs('cost_per_mtok', levels_a[hi], levels_b[hi], names, ' $')}"
            ),
            f"- Goodput at c={hi}: {vs('goodput', levels_a[hi], levels_b[hi], names, ' %')}",
            "",
            f"| c | metric | {a} | {b} | better |",
            "|---|---|---|---|---|",
        ]
        for c in cs:
            for key, (label, _) in METRICS.items():
                va, vb = levels_a[c][key], levels_b[c][key]
                lines.append(
                    f"| {c} | {label} | {fmt(va)} | {fmt(vb)}"
                    f" | {better(key, va, vb, names)} |"
                )
        lines.append("")
    lines += [
        "## Metrics",
        "",
        "- **TTFT**: time to first token. What a user waits before text appears.",
        (
            "- **TPOT**: time per output token after the first:"
            " (TTLT - TTFT) / (tokens - 1)."
        ),
        "- **TTLT**: time to last token, the whole request.",
        (
            "- **TPS, total**: output tokens per second of the whole server"
            " (throughput, cost)."
        ),
        "- **TPS, per user**: 1 / TPOT, the streaming speed one user sees.",
        "- **ITL p99**: 99th percentile pause between streamed chunks (stutter).",
        "- **Goodput**: share of requests that met the SLO above.",
        (
            "- **Cost per 1M output tokens**: GPU price / total TPS,"
            " with the GPU fully used."
        ),
        "",
        "c = concurrent requests. Raw timings: `results.json`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


# Measurements of one load level are repeated up to this many times if any
# request fails, e.g. because Modal preempted the GPU container mid-run.
ATTEMPTS = 3


def prepare(url: str, token: str, levels: list[int]) -> None:
    """Wait until the server answers, then warm it up at every load level.

    Each new batch size can trigger a one-time kernel compile (seconds), which
    would otherwise land in the measured tail (TTFT p95, TTLT p95).
    """
    wait_until_ready(url, token)
    for c in levels:
        run_load.remote(url, c, c, 50, 32)


@app.local_entrypoint()
def main(quick: bool = False) -> None:
    """Run the sweep on both engines, then write the report."""
    scenarios = {"chat": SCENARIOS["chat"]} if quick else SCENARIOS
    levels = CONCURRENCY[:2] if quick else CONCURRENCY
    raw: dict[str, Any] = {}
    summaries: dict[str, Summary] = {}
    ready_s: dict[str, float] = {}

    token = proxy_token()
    for engine, key in LABELS.items():
        url = deployed_url(ENGINES[key]).rstrip("/")
        t = time.perf_counter()
        prepare(url, token, levels)
        ready_s[engine] = time.perf_counter() - t
        raw[engine], summaries[engine] = {}, {}
        for scenario, (words, output_tokens) in scenarios.items():
            raw[engine][scenario], summaries[engine][scenario] = {}, {}
            for c in levels:
                n = max(8, 4 * c)
                for attempt in range(1, ATTEMPTS + 1):
                    print(f"{engine} | {scenario} | concurrency {c} | {n} requests")
                    run = run_load.remote(url, c, n, words, output_tokens)
                    s = summarize(run)
                    print(
                        f"  TTFT p50 {fmt(s['ttft_p50'])} ms, TPOT p50 {fmt(s['tpot_p50'])} ms,"
                        f" {fmt(s['tps'])} tok/s, errors {fmt(s['errors'])} %"
                    )
                    if s["errors"] == 0 or attempt == ATTEMPTS:
                        break
                    # Usually the GPU container was preempted (Modal then answers
                    # 503 until a new one boots). Wait for it, warm it, measure again.
                    print(
                        f"  errors: retry {attempt}/{ATTEMPTS - 1} after the server is back"
                    )
                    prepare(url, token, levels)
                raw[engine][scenario][c] = run
                summaries[engine][scenario][c] = s

    out = Path("reports") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(raw, indent=1))
    for scenario in scenarios:
        plot(summaries, scenario, out / f"{scenario}.png")
    write_report(summaries, ready_s, out)
    print(f"\nReport: {out / 'report.md'}")
