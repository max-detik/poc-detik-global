"""Indonesian -> English article rewriting.

generate_news_multi() turns one to MAX_SOURCE_ARTICLES scraped detik.com
articles about the same story into a single English article: the first is the
anchor, the rest only enrich it.

Separate from keywords_category.py, which derives keywords and categories from a
piece of content. The shared OpenRouter plumbing is in llm.py.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from generation.llm import (
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
    OPENROUTER_TEMPERATURE,
    client,
    load_prompt,
    parse_response,
)
from openai.types.shared_params import Reasoning


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

    api = client()

    system_instruction = load_prompt("news_multi")

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
    response = api.responses.parse(
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        input=messages,
        text_format=output_format,
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )

    # Extract structured JSON from function_call
    return parse_response(response)


