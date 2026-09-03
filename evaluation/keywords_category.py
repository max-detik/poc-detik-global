"""Evaluate the `categoryauto` output of generate_keywords_category().

Runs the function over a labelled dataset and scores the category it picks:

  input/eval_catauto_train.csv — the default: the `content` column in, scored
                               against the `catauto` column.
  input/test_catauto.csv     — same thing under the older header (`text` /
                               `category`); both layouts are accepted.
  input/apis-data-all.json   — `content` in, scored against `categoryauto`.
  input/sample_categoryauto_v7.csv — the per-category sample, scored against
                               `categoryauto_new`.

Single-label multiclass metrics: accuracy plus per-class and macro/weighted
precision, recall, F1. The keywords the function returns are recorded in the JSON
output but not scored — the datasets have no keyword labels worth scoring against.

Only the article text is sent to the model — the same contract the function has
in production. Category labels are compared on a loosened key (lowercased,
punctuation and spacing dropped), so "Kisah Inspiratif" matches "kisah
inspiratif" and "Musik K-pop" matches "musik kpop".

Run:  python -m evaluation.keywords_category [--input FILE] [--limit N] [--workers N]
      python -m evaluation.keywords_category --resume output/eval-sample-v7.json ...
        — re-runs only what errored last time and rewrites the full report.
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Importable as a package module (python -m evaluation.keywords_category) and
# runnable as a plain file; only the former puts the repo root on sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scoring import (
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
from generation.keywords_category import generate_keywords_category

INPUT_PATH = ROOT / "input/eval_catauto_train.csv"
OUTPUT_PATH = ROOT / "output/eval-keywords-category.json"
CATEGORY_CSV_PATH = ROOT / "output/eval-categories.csv"


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


# The two CSV layouts in use name the same three things differently, so each is
# read by the first header that's actually present.
CSV_COLUMNS = {
    "content": ("content", "text"),
    "category": ("catauto", "category", "categoryauto_new"),
    "id": ("id", "original_id"),
}


def _pick_column(fieldnames, names):
    """The first of `names` present in the CSV header, or None."""
    return next((name for name in names if name in fieldnames), None)


def _records_from_csv(path):
    """Evaluation records from a labelled CSV.

    input/eval_catauto_train.csv uses `content`/`catauto`; the older
    input/test_catauto.csv uses `text`/`category` — either is accepted.

    Rows without text, or without a category, are dropped — an unlabelled row
    cannot be scored, and a stringified null ("nan", "none", "null", ...) counts
    as no label. A few articles appear twice under one id with identical text (two
    `chunk_order` rows); only the first is kept, so the article is neither
    generated nor counted twice.
    """
    # Article bodies run past the default 128 KB field cap.
    csv.field_size_limit(sys.maxsize)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        columns = {
            field: _pick_column(fieldnames, names)
            for field, names in CSV_COLUMNS.items()
        }
        missing = [
            " / ".join(CSV_COLUMNS[field])
            for field in ("content", "category")
            if columns[field] is None
        ]
        if missing:
            raise SystemExit(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )
        records, skipped, duplicates, seen = [], 0, 0, set()
        for i, row in enumerate(reader):
            content = _clean(row.get(columns["content"]))
            category = _norm(_clean(row.get(columns["category"])))
            if not content or not category:
                skipped += 1
                continue
            # The unnamed first column is test_catauto.csv's row index.
            record_id = _clean(row.get(columns["id"])) or _clean(row.get("")) or str(i)
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
        print(f"  skipped {duplicates} duplicate row(s) sharing an id")
    return records


# ---------- run ----------

def load_previous(path):
    """{id: result} for the results of an earlier run that did not error.

    A run can end early — an API outage, or the account running out of credit
    mid-pass — leaving a JSON where some articles carry `error` instead of a
    prediction. Those ids are left out so `--resume` regenerates them; the rest
    are reused as they are, and are not paid for twice.
    """
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    results = saved.get("results", saved) if isinstance(saved, dict) else saved
    return {r["id"]: r for r in results if r.get("id") and not r.get("error")}



def _keywords(generated):
    """The predicted keywords as a plain list, whichever shape they arrive in.

    Current output carries `keywords_auto` (scored objects) plus the pipe-joined
    `keywordauto`; output saved before that had `keywordauto` as a list.
    """
    scored = generated.get("keywords_auto")
    if scored:
        return [k.get("keyword", "") for k in scored]
    raw = generated.get("keywordauto") or []
    if isinstance(raw, str):
        return [k for k in (part.strip() for part in raw.split("|")) if k]
    return raw


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

    # `categories_auto` is a ranked list of scored labels, best fit first; rank 1
    # is the single-label prediction. Older saved output carried the ranking in
    # `categoryauto` itself, as a list or a plain string — both still accepted.
    raw = generated.get("categories_auto")
    if raw:
        labels = [c.get("leaf") for c in raw]
    else:
        raw = generated.get("categoryauto")
        labels = raw if isinstance(raw, list) else [raw]
    predicted = [_norm(c) for c in labels if _clean(c)]
    result["predicted_categories"] = predicted
    result["predicted_category"] = predicted[0] if predicted else ""
    result["gold_tiers"] = resolve(record["gold_category"])
    result["predicted_tiers"] = resolve(result["predicted_category"])
    result["candidate_tiers"] = [resolve(c) for c in predicted]
    if len(set(predicted)) != len(predicted):
        result["duplicate_labels"] = True
    # Recorded for review only — the datasets carry no keyword labels to score.
    result["predicted_keywords"] = _keywords(generated)
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
        help="CSV dataset (content/catauto or text/category) or JSON (content/categoryauto)",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=CATEGORY_CSV_PATH,
        help="CSV of title / categoryauto / predicted categoryauto",
    )
    parser.add_argument("--limit", type=int, help="only evaluate the first N articles")
    parser.add_argument(
        "--resume",
        type=Path,
        help="reuse the successful results in this earlier --out JSON and only "
             "generate the articles that errored or are missing from it",
    )
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

    done = load_previous(args.resume) if args.resume else {}
    todo = [r for r in records if r["id"] not in done]
    if done:
        print(f"Resuming from {args.resume}: {len(done)} article(s) already generated, "
              f"{len(todo)} to run", flush=True)

    print(f"Evaluating {len(todo)} article(s) from {args.input} ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fresh = {r["id"]: r for r in pool.map(evaluate_article, todo)}

    # Kept in dataset order, whichever run each result came from.
    results = [done.get(r["id"]) or fresh[r["id"]] for r in records]

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
