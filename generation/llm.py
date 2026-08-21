"""Shared OpenRouter plumbing for the generation modules.

Both news.py (article rewrite) and keywords_category.py (keywords + category)
talk to the same model through the same client and parse responses the same way;
that shared part lives here so neither module imports the other.

System prompts are plain text under prompts/, loaded by name.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite")
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.5"))
OPENROUTER_REASONING_EFFORT = os.getenv("OPENROUTER_REASONING_EFFORT", "medium")

# System prompts live as plain text next to the code so they can be edited and
# diffed without touching Python.
PROMPTS_DIR = Path(__file__).parent / "prompts"


def client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


@lru_cache(maxsize=None)
def load_prompt(name):
    """The system prompt stored in prompts/<name>.txt."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def parse_response(response):
    """(parsed dict, token usage) from a client.responses.parse() result."""
    parsed = response.output_parsed
    response_output = json.loads(json.dumps(parsed.model_dump()))

    # OpenRouter reports generation cost inside `usage.cost`, which isn't part of
    # the SDK's typed Usage model, so pull it from the raw response payload.
    raw_usage = response.model_dump(mode="json").get("usage") or {}
    token_usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost": raw_usage.get("cost"),
    }
    return response_output, token_usage
