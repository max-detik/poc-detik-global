import csv
import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, Field, conlist, create_model
from typing import List
from typing import Literal

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite")
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.5"))
OPENROUTER_REASONING_EFFORT = os.getenv("OPENROUTER_REASONING_EFFORT", "medium")

class ContentArticleSchema(BaseModel):
    content: str = Field(..., description="content of the article.")
    style: Literal["text", "heading"] = Field(..., description="style of the article.")

class ArticleSchema(BaseModel):
    title: str = Field(..., max_length=80, description="title of the article")
    summary: str = Field(..., max_length=140, description="summary of the article")
    content: str = Field(..., description="content of the article, as HTML.")
    tags: List[str] = Field(min_length=5, max_length=10, description="news tags of the article")
    keywordauto: List[str] = Field(min_length=5, max_length=10, description="news keywords of the article")
    categoryauto: str = Field(..., description="news category of the article")
    image_cover_image_text: str = Field(..., description="rewritten caption of the image cover")
    image_cover_alt_image: str = Field(..., description="rewritten alt text of the image cover")

MAX_SOURCE_ARTICLES = 5

# System prompts live as plain text next to the code so they can be edited and
# diffed without touching Python.
PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def _load_prompt(name):
    """The system prompt stored in prompts/<name>.txt."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


# The allowed `categoryauto` labels and what each one covers, maintained by the
# desk as a CSV of "Leaf,Deskripsi" rows.
CATEGORY_LABELS_PATH = Path(__file__).parent / "input/categoryauto_labelling.csv"


def _clean_description(text):
    """One-line description; the CSV lists sub-items on their own lines."""
    lines = [" ".join(line.split()) for line in (text or "").splitlines()]
    return "; ".join(line for line in lines if line)


@lru_cache(maxsize=1)
def _category_labels():
    """[(leaf, description)] from the labelling CSV, in file order."""
    with open(CATEGORY_LABELS_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = [
            (row["Leaf"].strip(), _clean_description(row.get("Deskripsi")))
            for row in csv.DictReader(f)
            if (row.get("Leaf") or "").strip()
        ]
    if not rows:
        raise ValueError(f"no category labels found in {CATEGORY_LABELS_PATH}")
    return rows


@lru_cache(maxsize=1)
def _keyword_category_schema():
    """KeywordCategorySchema with `categoryauto` restricted to the CSV's leaves.

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
            Literal[leaves],  # type: ignore[valid-type]
            Field(..., description="single news category, exactly one label from the taxonomy"),
        ),
    )


def _client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def _format_news(news, label):
    return f"""
    [{label}]
    Title: {news["title"]}
    Content: {news["content"]}
    Resume: {news["resume"]}
    Tags: {news["tags"]}
    Image Caption: {news["image_cover_image_text"]}
    Alt Image: {news["image_cover_alt_image"]}
    Keyword Auto: {news["keywordauto"]}
    Category Auto: {news["categoryauto"]}"""


def _parse_response(response):
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


def generate_news_multi(news_items):
    """Rewrite 1..MAX_SOURCE_ARTICLES Indonesian articles into one English article.

    `news_items[0]` is the main article (the anchor): it fixes the story, the
    angle, the structure and the image metadata. Any remaining items are
    supporting sources whose facts are folded into the anchor's spine. A single
    article is the same path with no supporting sources — a straight rewrite of
    the anchor.

    Returns (article_dict, token_usage); article_dict follows ArticleSchema.
    """
    news_items = list(news_items)
    if not news_items:
        raise ValueError("news_items must contain at least one article")
    if len(news_items) > MAX_SOURCE_ARTICLES:
        raise ValueError(f"at most {MAX_SOURCE_ARTICLES} articles are supported, got {len(news_items)}")

    client = _client()

    system_instruction = _load_prompt("news_multi")

    anchor, *supporting = news_items
    blocks = [_format_news(anchor, "MAIN ARTICLE (ANCHOR)")]
    for i, item in enumerate(supporting, start=2):
        blocks.append(_format_news(item, f"SUPPORTING ARTICLE {i}"))

    prompt_input = (
        f"""
    Below are {len(news_items)} Indonesian source article(s).
    The first one is the main article (anchor) — it fixes the story, angle, structure and images.
    Any others are supporting sources; use them only to enrich the anchor's story. If no supporting
    article follows the anchor, the anchor is the only source: rewrite it alone, nothing merged in.
    """
        + "\n".join(blocks)
    )

    print(prompt_input)

    output_format = ArticleSchema
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_input}
    ]
    response = client.responses.parse(
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        input=messages,
        text_format=output_format,
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )

    # Extract structured JSON from function_call
    return _parse_response(response)


def _format_category_labels():
    return "\n".join(f"- {leaf} — {desc}" for leaf, desc in _category_labels())


def generate_keywords_category(content):
    """Derive `keywordauto` (5-10 terms) and `categoryauto` (one) from raw content text.

    `content` is a plain string — article body, HTML or plain text, in any
    language. `categoryauto` is always one of the labels in
    input/categoryauto_labelling.csv, enforced both in the prompt and by the
    response schema. Returns (result_dict, token_usage).
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("content must not be empty")

    client = _client()

    system_instruction = _load_prompt("keywords_category").replace(
        "{category_list}", _format_category_labels()
    )

    prompt_input = f"""
    Below is the content. Output only its keywords and category.

    Content: {content}"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_input},
    ]
    response = client.responses.parse(
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        input=messages,
        text_format=_keyword_category_schema(),
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )

    return _parse_response(response)
