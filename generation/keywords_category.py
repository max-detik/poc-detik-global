"""Keyword and category generation from a piece of content.

generate_keywords_category() reads one article body and returns the scored
search keywords it should be indexed under plus its ranked category labels,
chosen from the taxonomy in input/categoryauto_labelling.csv. The keywords serve
both search and content-based recommendation, so they favour named entities and
topics that recur across articles.

Separate from news.py, which rewrites articles: different prompt, different
schema, different input. The shared OpenRouter plumbing is in llm.py.
"""

import csv
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, create_model

from generation.llm import (
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
    OPENROUTER_TEMPERATURE,
    client,
    keywords_field,
    load_prompt,
    normalize_keywords,
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
def _category_auto_model():
    """CategoryAuto with `leaf` restricted to the CSV's leaves.

    The tree levels and the id are null here: this call only picks the leaf, and
    the parents are resolved downstream from the same CSV.
    """
    leaves = tuple(leaf for leaf, _ in _category_labels())
    return create_model(
        "CategoryAuto",
        score=(float, Field(..., ge=0, le=1, description="confidence 0-1, descending across the array")),
        tree_level2=(Optional[str], Field(..., description="always null")),
        tree_level1=(Optional[str], Field(..., description="always null")),
        rank=(int, Field(..., ge=1, description="1-based position, ascending across the array")),
        id=(Optional[str], Field(..., description="always null")),
        leaf=(Literal[leaves], Field(..., description="most specific category label")),  # type: ignore[valid-type]
    )


@lru_cache(maxsize=1)
def _keyword_category_schema():
    """The response schema: ranked categories and scored keywords.

    `categories_auto` is a ranked list of CATEGORY_CHOICES labels, best fit first
    — the same shape the production categoriser stores (rank 1..3), so a story
    that genuinely spans several desks is not forced into one. `categoryauto` and
    `keywordauto` are the flat forms the CMS stores; both are re-derived from the
    arrays in _normalize(), so a model that fills them inconsistently can't drift.

    Built at call time so the allowed values always come from the CSV on disk,
    and so importing this module doesn't require the file to be present.
    """
    return create_model(
        "KeywordCategorySchema",
        categoryauto=(
            str,
            Field(..., description="the single best-matching leaf, identical to categories_auto[0].leaf"),
        ),
        categories_auto=(
            List[_category_auto_model()],  # type: ignore[valid-type]
            Field(
                min_length=CATEGORY_CHOICES,
                max_length=CATEGORY_CHOICES,
                description=(
                    f"{CATEGORY_CHOICES} distinct news categories from the taxonomy, "
                    "ranked best fit first"
                ),
            ),
        ),
        keywordauto=(
            str,
            Field(..., description="all keywords joined with '|', no surrounding spaces"),
        ),
        keywords_auto=keywords_field(),
    )


def _normalize(result):
    """Make the flat fields agree with the arrays they summarize.

    The model is asked for `rank`, `categoryauto` and `keywordauto` in the prompt,
    but they are pure functions of the arrays, so they are recomputed here rather
    than trusted: ranks renumber 1..n in array order, `categoryauto` becomes the
    first leaf, and `keywordauto` the pipe-joined keywords (in llm.py, shared with
    the article rewrite).
    """
    categories = result.get("categories_auto") or []
    for position, category in enumerate(categories, start=1):
        category["rank"] = position
    result["categoryauto"] = categories[0]["leaf"] if categories else ""
    return normalize_keywords(result)


def _format_category_labels():
    return "\n".join(f"- {leaf} — {desc}" for leaf, desc in _category_labels())


def render_system_instruction():
    """The full system prompt as sent to the model, placeholders filled in.

    The stored template keeps `{category_list}` and `{category_choices}` so the
    taxonomy stays a CSV edit rather than a prompt edit; this is what the model
    actually reads. scripts/dump_prompts.py writes it out for review.
    """
    return (
        load_prompt("keywords_category")
        .replace("{category_list}", _format_category_labels())
        .replace("{category_choices}", str(CATEGORY_CHOICES))
    )


def generate_keywords_category(content):
    """Derive the keyword and category metadata for a piece of content.

    `content` is a plain string — article body, HTML or plain text, in any
    language. Returns (result_dict, token_usage), where the dict carries
    `categories_auto` (CATEGORY_CHOICES scored labels, best fit first, each one
    from input/categoryauto_labelling.csv — enforced both in the prompt and by the
    response schema), `keywords_auto` (5-10 scored keywords, most central first),
    and the flat `categoryauto` / pipe-joined `keywordauto` forms of the two.
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("content must not be empty")

    api = client()

    system_instruction = render_system_instruction()

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

    result, usage = parse_response(response)
    return _normalize(result), usage