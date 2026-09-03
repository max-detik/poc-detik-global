"""Shared OpenRouter plumbing for the generation modules.

Both news.py (article rewrite) and keywords_category.py (keywords + category)
talk to the same model through the same client and parse responses the same way;
that shared part lives here so neither module imports the other.

System prompts are plain text under prompts/, loaded by name. The keyword
metadata shape both generators emit lives here too, for the same reason.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

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


# ---------- keyword metadata ----------

# Both generators return keywords in the same shape: a scored array plus the flat
# pipe-joined string the CMS stores. Kept here so news.py and
# keywords_category.py share it without importing each other.
KEYWORD_SEPARATOR = "|"
MIN_KEYWORDS, MAX_KEYWORDS = 5, 10


class KeywordAuto(BaseModel):
    """One scored keyword. `label` is filled downstream, so the model returns null."""

    score: float = Field(..., ge=0, le=1, description="relevance 0-1, descending across the array")
    label: Optional[str] = Field(..., description="always null")
    keyword: str = Field(..., description="the keyword phrase, lowercase")


def keywords_field():
    """The (type, Field) pair for a `keywords_auto` array on a response schema."""
    return (
        List[KeywordAuto],
        Field(
            min_length=MIN_KEYWORDS,
            max_length=MAX_KEYWORDS,
            description="keywords describing the article, most central first",
        ),
    )


def normalize_keywords(result):
    """Rebuild `keywordauto` from `keywords_auto` so the two can't disagree.

    The model is asked for both in the prompt, but the flat string is a pure
    function of the array, so it is recomputed rather than trusted.
    """
    keywords = [
        keyword for keyword in (result.get("keywords_auto") or [])
        if (keyword.get("keyword") or "").strip()
    ]
    for keyword in keywords:
        keyword["keyword"] = keyword["keyword"].strip()
    result["keywords_auto"] = keywords
    result["keywordauto"] = KEYWORD_SEPARATOR.join(k["keyword"] for k in keywords)
    return result
