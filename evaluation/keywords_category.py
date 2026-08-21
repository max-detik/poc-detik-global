"""Evaluate the `categoryauto` output of generate_keywords_category().

Runs the function over a labelled dataset and scores the category it picks:

  input/test_catauto.csv     — the default: the `text` column in, scored against
                               the `category` column.
  input/apis-data-all.json   — `content` in, scored against `categoryauto`.

Single-label multiclass metrics: accuracy plus per-class and macro/weighted
precision, recall, F1. The keywords the function returns are recorded in the JSON
output but not scored — the datasets have no keyword labels worth scoring against.

Only the article text is sent to the model — the same contract the function has
in production. Category labels are compared on a loosened key (lowercased,
punctuation and spacing dropped), so "Kisah Inspiratif" matches "kisah
inspiratif" and "Musik K-pop" matches "musik kpop".

Run:  python eval_keywords_category.py [--input FILE] [--limit N] [--workers N]
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from category_scoring import (  # noqa: E402
    TIERS,
    clean as _clean,
    loose as _loose,
    norm as _norm,
    print_ranked_summary,
    print_summary,
    print_tier,
    rank_metrics,
    resolve,
    score_tiers,
)
from prompt import generate_keywords_category  # noqa: E402

INPUT_PATH = BASE_DIR / "input/test_catauto.csv"
OUTPUT_PATH = BASE_DIR / "output/eval-keywords-category.json"
CATEGORY_CSV_PATH = BASE_DIR / "output/eval-categories.csv"


# ---------- input ----------

def load_records(path):
    """Normalized evaluation records from a JSON or CSV dataset.

    Each record: id, title, content, gold_category.
    """
    if path.suffix.lower() == ".csv":
        return _records_from_csv(path)
    return _records_from_json(path)


def _records_from_json(path):
    """input/apis-data-all.json: `content` is the text, `categoryauto` the label."""
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return [
        {
            "id": article.get("id"),
            "title": _clean(article.get("title")),
            "content": article.get("content", ""),
            "gold_category": _norm(_clean(article.get("categoryauto"))),
        }
        for article in articles
        if _clean(article.get("content")) and _clean(article.get("categoryauto"))
    ]


def _records_from_csv(path):
    """input/test_catauto.csv: `text` is the content, `category` is the label.

    Rows without text, or without a category, are dropped — an unlabelled row
    cannot be scored, and a stringified null ("nan", "none", "null", ...) counts
    as no label. A few articles appear twice under one `original_id` with
    identical text (two `chunk_order` rows); only the first is kept, so the article
    is neither generated nor counted twice.
    """
    # Article bodies run past the default 128 KB field cap.
    csv.field_size_limit(sys.maxsize)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"text", "category"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path} is missing required column(s): {', '.join(sorted(missing))}"
            )
        records, skipped, duplicates, seen = [], 0, 0, set()
        for i, row in enumerate(reader):
            content = _clean(row.get("text"))
            category = _norm(_clean(row.get("category")))
            if not content or not category:
                skipped += 1
                continue
            record_id = _clean(row.get("original_id")) or _clean(row.get("")) or str(i)
            if record_id in seen:
                duplicates += 1
                continue
            seen.add(record_id)
            records.append({
                "id": record_id,
                "title": _clean(row.get("title")),
                "content": content,
                "gold_category": category,
            })
    if skipped:
        print(f"  skipped {skipped} row(s) with no text or no category label")
    if duplicates:
        print(f"  skipped {duplicates} duplicate row(s) sharing an original_id")
    return records


# ---------- run ----------

def evaluate_article(record):
    """Generate for one record and score it. Errors are captured, not raised."""
    result = {
        "id": record["id"],
        "title": record["title"],
        "gold_category": record["gold_category"],
    }
    try:
        generated, usage = generate_keywords_category(record["content"])
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    # `categoryauto` is a ranked list, best fit first; rank 1 is the single-label
    # prediction. A plain string is still accepted, for older saved output.
    raw = generated.get("categoryauto")
    predicted = [_norm(c) for c in (raw if isinstance(raw, list) else [raw]) if _clean(c)]
    result["predicted_categories"] = predicted
    result["predicted_category"] = predicted[0] if predicted else ""
    result["gold_tiers"] = resolve(record["gold_category"])
    result["predicted_tiers"] = resolve(result["predicted_category"])
    result["candidate_tiers"] = [resolve(c) for c in predicted]
    if len(set(predicted)) != len(predicted):
        result["duplicate_labels"] = True
    # Recorded for review only — the datasets carry no keyword labels to score.
    result["predicted_keywords"] = generated.get("keywordauto") or []
    result["usage"] = usage
    return result


def summarize(results):
    scored = [r for r in results if "error" not in r]
    # score_tiers() speaks reference/predicted; here the reference is the gold label.
    tiers = score_tiers([
        {
            "reference": r["gold_category"],
            "reference_tiers": r["gold_tiers"],
            "predicted": r["predicted_category"],
            "predicted_tiers": r["predicted_tiers"],
        }
        for r in scored
    ])

    ranked = rank_metrics([
        {
            "reference_tiers": r["gold_tiers"],
            # Output saved before categoryauto became a ranked list has one.
            "candidate_tiers": r.get("candidate_tiers") or [r["predicted_tiers"]],
        }
        for r in scored
    ])

    return {
        "articles": len(results),
        "scored": len(scored),
        "failed": len(results) - len(scored),
        "choices": max((len(r.get("candidate_tiers") or [1]) for r in scored), default=0),
        "duplicate_label_rows": sum(1 for r in scored if r.get("duplicate_labels")),
        "category": tiers,
        "ranked": ranked,
        "usage": _usage_totals(scored),
    }


def _usage_totals(scored):
    total_tokens = sum((r.get("usage") or {}).get("total_tokens") or 0 for r in scored)
    total_cost = sum((r.get("usage") or {}).get("cost") or 0.0 for r in scored)
    n = len(scored) or 1
    return {
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "avg_tokens": total_tokens / n,
        "avg_cost": round(total_cost / n, 6),
    }


def print_report(summary, results, per_class=True):
    print_summary(summary["category"], label="SUMMARY (rank 1)")
    if summary["choices"] > 1:
        print_ranked_summary(summary["ranked"], summary["choices"])
        if summary["duplicate_label_rows"]:
            print(f"\n  {summary['duplicate_label_rows']} article(s) returned a duplicate label")
    for tier in TIERS:
        print_tier(tier, summary["category"][tier], summary["scored"], per_class)

    print("\n=== PER ARTICLE ===   (leaf/level2/level1: Y hit, n miss, . not scorable)")
    for r in results:
        if "error" in r:
            print(f"  [{r['id']}] ERROR {r['error']}")
            continue
        hits = "".join(
            "." if not (r["gold_tiers"][t] and r["predicted_tiers"][t])
            else ("Y" if _loose(r["gold_tiers"][t]) == _loose(r["predicted_tiers"][t]) else "n")
            for t in TIERS
        )
        usage = r.get("usage") or {}
        predicted = " > ".join(r.get("predicted_categories") or [r["predicted_category"]])
        print(f"  [{r['id']}] {hits} category: {predicted} "
              f"(gold {r['gold_category']!r})  "
              f"{usage.get('total_tokens') or 0} tok, ${usage.get('cost') or 0.0:.6f}")

    u = summary["usage"]
    print("\n=== COST ===")
    print(f"  articles generated  {summary['scored']}"
          + (f" ({summary['failed']} failed)" if summary["failed"] else ""))
    print(f"  tokens              {u['total_tokens']} total, {u['avg_tokens']:.0f} per article")
    print(f"  cost                ${u['total_cost']:.6f} total, "
          f"${u['avg_cost']:.6f} per article")
    print(f"  projected           ${u['avg_cost'] * 1000:.4f} per 1,000 articles")


def write_category_csv(results, path):
    """title / gold / predicted category, rolled up to each taxonomy tier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    empty = {t: None for t in TIERS}
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        depth = max((len(r.get("predicted_categories") or []) for r in results), default=1)
        writer.writerow(
            ["id", "title", "categoryauto", "predicted_categoryauto"]
            + [f"predicted_rank{k}" for k in range(2, depth + 1)]
            + ["hit_rank_leaf"]
            + [c for t in TIERS for c in (f"gold_{t}", f"predicted_{t}", f"match_{t}")]
        )
        for r in results:
            gold_tiers = r.get("gold_tiers") or empty
            predicted_tiers = r.get("predicted_tiers") or empty
            predicted_list = r.get("predicted_categories") or []
            row = [
                r.get("id", ""),
                r.get("title", ""),
                r.get("gold_category", ""),
                r.get("error") or r.get("predicted_category", ""),
            ]
            row += [
                predicted_list[k] if k < len(predicted_list) else "" for k in range(1, depth)
            ]
            # Which rank first matched the gold leaf, blank when none did.
            hit_rank = ""
            for i, candidate in enumerate(r.get("candidate_tiers") or [], start=1):
                if candidate["leaf"] and gold_tiers["leaf"] \
                        and _loose(candidate["leaf"]) == _loose(gold_tiers["leaf"]):
                    hit_rank = i
                    break
            row.append(hit_rank)
            for tier in TIERS:
                gold, predicted = gold_tiers[tier], predicted_tiers[tier]
                row += [
                    gold or "",
                    predicted or "",
                    "" if not (gold and predicted) else ("1" if _loose(gold) == _loose(predicted) else "0"),
                ]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="CSV dataset (text/category) or JSON (content/categoryauto)",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=CATEGORY_CSV_PATH,
        help="CSV of title / categoryauto / predicted categoryauto",
    )
    parser.add_argument("--limit", type=int, help="only evaluate the first N articles")
    parser.add_argument("--workers", type=int, default=4, help="parallel generations")
    parser.add_argument(
        "--no-per-class",
        action="store_true",
        help="only print the accuracy/macro/weighted lines, not the per-class tables",
    )
    args = parser.parse_args()

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]

    print(f"Evaluating {len(records)} article(s) from {args.input} ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_article, records))

    summary = summarize(results)
    print_report(summary, results, per_class=not args.no_per_class)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Wrote details to {args.out}")

    write_category_csv(results, args.out_csv)
    print(f"Wrote categories to {args.out_csv}")


if __name__ == "__main__":
    main()
