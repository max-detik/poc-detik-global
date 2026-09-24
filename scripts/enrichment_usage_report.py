"""One-off cost/usage report for the Enrich Article pipeline (generate + TL;DR /
Takeaways / FAQ), run over a handful of curated samples.

Follows the same call path web.app's POST /api/enrich takes: generate_from_articles()
on one sample, then generate_enrichment() on the resulting English title/content.
Writes a row per sample to output/enrichment_usage_report.csv (cost/token numbers) and
the full generated article + enrichment text to output/enrichment_usage_report.md (for
reading the actual output, not just its cost), then prints a summary.

Run:  python -m scripts.enrichment_usage_report [--limit N] [--out FILE]
"""

import argparse
import csv
import json
from pathlib import Path

from bs4 import BeautifulSoup

from generation.articles import generate_from_articles
from generation.enrichment import generate_enrichment

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / "scraping/apis-data-all.json"
OUTPUT_PATH = ROOT / "output/enrichment_usage_report.csv"
DETAILS_PATH = ROOT / "output/enrichment_usage_report.md"

FIELDS = [
    "sample_id", "title",
    "generate_input_tokens", "generate_output_tokens", "generate_total_tokens", "generate_cost",
    "enrich_input_tokens", "enrich_output_tokens", "enrich_total_tokens", "enrich_cost",
    "grand_total_tokens", "grand_total_cost",
]


def _row(sample, generated, generate_usage, enrich_usage):
    g, e = generate_usage, enrich_usage
    grand_tokens = (g.get("total_tokens") or 0) + (e.get("total_tokens") or 0)
    grand_cost = (g.get("cost") or 0.0) + (e.get("cost") or 0.0)
    return {
        "sample_id": sample.get("id"),
        "title": generated.get("title") or sample.get("title"),
        "generate_input_tokens": g.get("input_tokens"),
        "generate_output_tokens": g.get("output_tokens"),
        "generate_total_tokens": g.get("total_tokens"),
        "generate_cost": g.get("cost"),
        "enrich_input_tokens": e.get("input_tokens"),
        "enrich_output_tokens": e.get("output_tokens"),
        "enrich_total_tokens": e.get("total_tokens"),
        "enrich_cost": e.get("cost"),
        "grand_total_tokens": grand_tokens,
        "grand_total_cost": round(grand_cost, 6),
    }


def _html_to_text(html):
    """Paragraph-per-line plain text. Splitting on every tag (get_text("\\n\\n"))
    would also break at inline tags like <a>, fragmenting a sentence that merely
    links a name — so split only at block-level boundaries instead."""
    soup = BeautifulSoup(html or "", "html.parser")
    blocks = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
    if not blocks:
        return soup.get_text(" ", strip=True)
    return "\n\n".join(b.get_text(" ", strip=True) for b in blocks if b.get_text(strip=True))


def _detail_section(sample, generated, result):
    lines = [
        f"## {sample.get('id')} — {generated.get('title', '')}",
        "",
        f"**Original title (ID):** {sample.get('title', '')}",
        "",
        "### Generated article (EN)",
        f"**Title:** {generated.get('title', '')}",
        "",
        f"**Summary:** {generated.get('summary', '')}",
        "",
        "**Content:**",
        "",
        _html_to_text(generated.get("content", "")),
        "",
        f"**Tags:** {', '.join(generated.get('tags') or [])}",
        "",
        f"**Category:** {generated.get('categoryauto', '')}",
        "",
        "### Enrichment",
        "**TL;DR:**",
    ]
    lines += [f"- {b}" for b in result.get("tldr", [])]
    lines += ["", f"**Key Takeaway:** {result.get('key_takeaway', '')}", "", "**FAQ:**"]
    for qa in result.get("faq", []):
        lines += [
            f"- Q: {qa.get('question', '')}",
            f"  A: {qa.get('answer', '')}",
            f"  Evidence: {qa.get('evidence', '')}",
        ]

    flagged = {k: v for k, v in (result.get("diagnostics") or {}).items() if v}
    if flagged:
        lines += ["", "**QA diagnostics flagged:**"]
        lines += [f"- {k}: {v}" for k, v in flagged.items()]

    lines += ["", "---", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5, help="number of samples to run")
    parser.add_argument("--ids", help="comma-separated sample ids to run instead of the first --limit")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--details-out", type=Path, default=DETAILS_PATH)
    args = parser.parse_args()

    all_samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
    if args.ids:
        wanted = [s.strip() for s in args.ids.split(",") if s.strip()]
        by_id = {str(s.get("id")): s for s in all_samples}
        samples = [by_id[i] for i in wanted if i in by_id]
    else:
        samples = all_samples[: args.limit]

    rows = []
    detail_sections = []
    for i, sample in enumerate(samples, start=1):
        title = sample.get("title", "")
        print(f"[{i}/{len(samples)}] {sample.get('id')} - {title[:60]!r}")
        try:
            generated, generate_usage = generate_from_articles([sample])
            result, enrich_usage = generate_enrichment(generated.get("title", ""), generated.get("content", ""))
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            continue
        row = _row(sample, generated, generate_usage, enrich_usage)
        rows.append(row)
        detail_sections.append(_detail_section(sample, generated, result))
        print(f"  generate ${row['generate_cost']:.6f}  enrich ${row['enrich_cost']:.6f}  "
              f"total ${row['grand_total_cost']:.6f} ({row['grand_total_tokens']} tok)")

    if not rows:
        raise SystemExit("no rows produced - every sample failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.details_out, "w", encoding="utf-8") as f:
        f.write(f"# Enrichment usage report — {len(rows)} article(s)\n\n")
        f.write("\n".join(detail_sections))

    n = len(rows)
    total_cost = sum(r["grand_total_cost"] for r in rows)
    total_tokens = sum(r["grand_total_tokens"] for r in rows)
    avg_cost = total_cost / n
    avg_tokens = total_tokens / n

    print(f"\nWrote {n} row(s) to {args.out}")
    print(f"Wrote full generated text to {args.details_out}")
    print(f"  tokens    {total_tokens} total, {avg_tokens:.0f} per article")
    print(f"  cost      ${total_cost:.6f} total, ${avg_cost:.6f} per article")
    print(f"  projected ${avg_cost * 1000:.4f} per 1,000 articles")


if __name__ == "__main__":
    main()
