"""Keyword and category generation from a piece of content.

generate_keywords_category() reads one article body and returns the search
keywords it should carry plus its ranked `categoryauto` labels, chosen from the
taxonomy in input/categoryauto_labelling.csv.

Separate from news.py, which rewrites articles: different prompt, different
schema, different input. The shared OpenRouter plumbing is in llm.py.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, create_model

from generation.llm import (
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
    OPENROUTER_TEMPERATURE,
    client,
    load_prompt,
    parse_response,
)
from openai.types.shared_params import Reasoning

# How many ranked category labels generate_keywords_category() returns.
CATEGORY_CHOICES = 3

# The allowed `categoryauto` labels and what each one covers, maintained by the
# desk as a CSV of "level 1, level 2, Leaf, Deskripsi" rows. Generation uses the
# leaves; the two parent levels are there for reporting and evaluation.
ROOT = Path(__file__).resolve().parents[1]
CATEGORY_LABELS_PATH = ROOT / "input/categoryauto_labelling.csv"


def _clean_description(text):
    """One-line description; the CSV lists sub-items on their own lines."""
    lines = [" ".join(line.split()) for line in (text or "").splitlines()]
    return "; ".join(line for line in lines if line)


def _column(row, *names):
    """Row value by header name, tolerant of case and spacing ("level 1"/"Level1")."""
    keys = {"".join((k or "").split()).lower(): k for k in row}
    for name in names:
        key = keys.get("".join(name.split()).lower())
        if key is not None:
            return (row[key] or "").strip()
    return ""


@lru_cache(maxsize=1)
def category_taxonomy():
    """[(level1, level2, leaf, description)] from the labelling CSV, in file order."""
    with open(CATEGORY_LABELS_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = [
            (
                _column(row, "level 1"),
                _column(row, "level 2"),
                _column(row, "Leaf"),
                _clean_description(_column(row, "Deskripsi")),
            )
            for row in csv.DictReader(f)
        ]
    rows = [r for r in rows if r[2]]
    if not rows:
        raise ValueError(f"no category labels found in {CATEGORY_LABELS_PATH}")
    return rows


@lru_cache(maxsize=1)
def _category_labels():
    """[(leaf, description)] — what the prompt offers the model to choose from."""
    return [(leaf, description) for _, _, leaf, description in category_taxonomy()]


@lru_cache(maxsize=1)
def _keyword_category_schema():
    """KeywordCategorySchema with `categoryauto` restricted to the CSV's leaves.

    `categoryauto` is a ranked list of CATEGORY_CHOICES labels, best fit first —
    the same shape the production categoriser stores (rank 1..3), so a story that
    genuinely spans several desks is not forced into one.

    Built at call time so the allowed values always come from the CSV on disk,
    and so importing this module doesn't require the file to be present.
    """
    leaves = tuple(leaf for leaf, _ in _category_labels())
    return create_model(
        "KeywordCategorySchema",
        keywordauto=(
            List[str],
            Field(min_length=5, max_length=10, description="search keywords describing the content"),
        ),
        categoryauto=(
            List[Literal[leaves]],  # type: ignore[valid-type]
            Field(
                min_length=CATEGORY_CHOICES,
                max_length=CATEGORY_CHOICES,
                description=(
                    f"{CATEGORY_CHOICES} distinct news categories from the taxonomy, "
                    "ranked best fit first"
                ),
            ),
        ),
    )


def _format_category_labels():
    return "\n".join(f"- {leaf} — {desc}" for leaf, desc in _category_labels())


def generate_keywords_category(content):
    """Derive `keywordauto` (5-10 terms) and `categoryauto` from raw content text.

    `content` is a plain string — article body, HTML or plain text, in any
    language. `categoryauto` is a ranked list of CATEGORY_CHOICES labels, best fit
    first, each one from input/categoryauto_labelling.csv — enforced both in the
    prompt and by the response schema. Returns (result_dict, token_usage).
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("content must not be empty")

    api = client()

    system_instruction = (
        load_prompt("keywords_category")
        .replace("{category_list}", _format_category_labels())
        .replace("{category_choices}", str(CATEGORY_CHOICES))
    )

    prompt_input = f"""
    Below is the content. Output only its keywords and its ranked categories.

    Content: {content}"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_input},
    ]
    response = api.responses.parse(
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        input=messages,
        text_format=_keyword_category_schema(),
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )

    return parse_response(response)