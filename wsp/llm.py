"""All LLM calls go through call_llm(). Each call is logged to logs/<tag>.json;
if that log already exists it is replayed instead of calling the model.

Calls run through headless Claude Code (`claude -p`) on a Claude subscription,
isolated: empty temp working dir (no repo files or CLAUDE.md), no tools, no MCP
servers, our own system prompt, no saved session. cost_usd is Claude Code's
list-price equivalent, not an amount billed.
"""
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

MODEL = "claude-opus-5-5"
EFFORT = "high"
SYSTEM = "You are an expert in machine learning."
LOG_DIR = Path("logs")


def call_llm(prompt: str, tag: str) -> dict:
    path = LOG_DIR / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # use subscription login
    t0 = time.time()
    with tempfile.TemporaryDirectory() as empty_dir:
        out = subprocess.run(
            ["claude", "-p", prompt, "--model", MODEL, "--effort", EFFORT,
             "--tools", "", "--strict-mcp-config", "--no-session-persistence",
             "--system-prompt", SYSTEM, "--output-format", "json"],
            cwd=empty_dir, env=env, capture_output=True, text=True,
        )
    if out.returncode != 0:
        raise RuntimeError(f"claude -p failed for {tag}: {(out.stderr or out.stdout)[-500:]}")
    res = json.loads(out.stdout)
    u = res["usage"]

    rec = {
        "tag": tag, "model": MODEL, "effort": EFFORT, "system": SYSTEM, "prompt": prompt,
        "text": res["result"],
        "stop_reason": res["stop_reason"],
        "input_tokens": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
        "output_tokens": u["output_tokens"],
        "thinking_tokens": u["output_tokens_details"]["thinking_tokens"],
        "cost_usd": res["total_cost_usd"],
        "wall_s": time.time() - t0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


def extract_json(text: str) -> dict:
    """Parse the last ```json block in a reply."""
    return json.loads(re.findall(r"```json\s*\n(.*?)```", text, flags=re.S)[-1])
