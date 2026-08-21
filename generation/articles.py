"""Turn scraped detik.com articles into generated English articles.

generate_from_articles() is the entry point the web app uses; main() runs the
same thing over input/apis-data-all.json as a batch.

Run:  python -m generation.articles
"""

import json
import sys
from pathlib import Path

from generation.news import MAX_SOURCE_ARTICLES, generate_news_multi

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "input/apis-data-all.json"
OUTPUT_PATH = ROOT / "output/generated-articles-all.json"
MAX_ATTEMPTS = 3


def build_news_input(article):
    tags = article.get("tags")
    if not tags:
        tags = (article.get("tag") or "").split("|")

    image_cover = article.get("image_cover") or {}

    return {
        "title": article.get("title", ""),
        "content": article.get("content", ""),
        "resume": article.get("resume", ""),
        "tags": tags,
        "keywordauto": article.get("keywordauto", ""),
        "categoryauto": article.get("categoryauto", ""),
        "image_cover_image_text": image_cover.get("text", ""),
        "image_cover_alt_image": image_cover.get("alt_image", ""),
    }


def build_news_inputs(articles):
    """Prompt inputs for a group of source articles, anchor (main article) first.

    The caller passes the articles in display order: articles[0] is the one the
    user flagged as the main article/anchor.
    """
    articles = list(articles)
    if not articles:
        raise ValueError("at least one article is required")
    if len(articles) > MAX_SOURCE_ARTICLES:
        raise ValueError(
            f"at most {MAX_SOURCE_ARTICLES} articles are supported, got {len(articles)}"
        )
    return [build_news_input(article) for article in articles]


def generate_multi_with_retries(news_items, cost_tracker, attempts=MAX_ATTEMPTS):
    """Generate one article from a group of 1..MAX_SOURCE_ARTICLES source articles.

    news_items[0] is the anchor; the rest only enrich it. A single article takes
    the same path — generate_news_multi() handles it as an anchor with no
    supporting sources.
    """
    news_items = list(news_items)

    last_error = None
    for attempt in range(1, attempts + 1):
        # try:
        generated, token_usage = generate_news_multi(news_items)
        # except Exception as e:
        #     last_error = e
        #     print(f"  attempt {attempt}/{attempts} failed: {e}", file=sys.stderr)
        #     continue

        cost_tracker["total_cost"] += token_usage.get("cost") or 0.0
        cost_tracker["total_tokens"] += token_usage.get("total_tokens") or 0
        generated["usage"] = token_usage
        return generated

    print(f"  giving up after {attempts} attempts: {last_error}", file=sys.stderr)
    return None


def generate_from_articles(articles, cost_tracker=None):
    """One generated article from 1..MAX_SOURCE_ARTICLES scraped source articles.

    articles[0] is the main article/anchor. Returns (generated, token_usage).
    """
    tracker = cost_tracker if cost_tracker is not None else {"total_cost": 0.0, "total_tokens": 0}
    news_items = build_news_inputs(articles)
    generated = generate_multi_with_retries(news_items, tracker)
    if generated is None:
        raise RuntimeError("generation failed")
    return generated, generated.get("usage", {})


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []
    cost_tracker = {"total_cost": 0.0, "total_tokens": 0}
    for i, article in enumerate(articles[:2], start=1):
        print(f"[{i}/{len(articles)}] Generating: {article.get('title', '')!r}")
        news = build_news_input(article)
        generated = generate_multi_with_retries([news], cost_tracker)
        if generated is not None:
            results.append(generated)

    output = {
        "articles": results,
        "usage": {
            "total_tokens": cost_tracker["total_tokens"],
            "total_cost": round(cost_tracker["total_cost"], 6),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} article(s) to {OUTPUT_PATH}")
    print(f"Total cost: ${cost_tracker['total_cost']:.6f} ({cost_tracker['total_tokens']} tokens)")


if __name__ == "__main__":
    main()
