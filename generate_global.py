import json
import sys
from pathlib import Path

from prompt import generate_news_single

INPUT_PATH = Path(__file__).parent / "apis-data.json"
OUTPUT_PATH = Path(__file__).parent / "generated-articles.json"
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
        "image_cover_image_text": image_cover.get("text", ""),
        "image_cover_alt_image": image_cover.get("alt_image", ""),
    }


def generate_with_retries(news, cost_tracker, attempts=MAX_ATTEMPTS):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            generated, token_usage = generate_news_single(news)
        except Exception as e:
            last_error = e
            print(f"  attempt {attempt}/{attempts} failed: {e}", file=sys.stderr)
            continue

        cost_tracker["total_cost"] += token_usage.get("cost") or 0.0
        cost_tracker["total_tokens"] += token_usage.get("total_tokens") or 0
        generated["usage"] = token_usage
        return generated

    print(f"  giving up after {attempts} attempts: {last_error}", file=sys.stderr)
    return None


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)

    results = []
    cost_tracker = {"total_cost": 0.0, "total_tokens": 0}
    for i, article in enumerate(articles, start=1):
        print(f"[{i}/{len(articles)}] Generating: {article.get('title', '')!r}")
        news = build_news_input(article)
        generated = generate_with_retries(news, cost_tracker)
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
