"""Evaluate generate_keywords_category() against labelled sample data.

Runs the function over a labelled dataset and scores its output:

  input/apis-data-all.json   — `content` in, scored against `categoryauto` and
                               `keywordauto`.
  input/test_catauto.csv     — the `text` column in, scored against the
                               `category` column. Category only; the file carries
                               no gold keywords, so keyword metrics are skipped.

Metrics:

  category  — single-label multiclass: accuracy plus per-class and macro/weighted
              precision, recall, F1.
  keywords  — multi-label set comparison per article: micro (pooled TP/FP/FN) and
              macro (mean of per-article scores), under exact and partial matching.

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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from prompt import _category_labels, generate_keywords_category  # noqa: E402

INPUT_PATH = BASE_DIR / "input/apis-data-all.json"
OUTPUT_PATH = BASE_DIR / "output/eval-keywords-category.json"
CATEGORY_CSV_PATH = BASE_DIR / "output/eval-categories.csv"


# ---------- normalization ----------

def _norm(text):
    """Lowercased, whitespace-collapsed comparison key."""
    return " ".join((text or "").split()).lower()


def _loose(text):
    """Category matching key: letters and digits only.

    The datasets spell the same label several ways ("Musik K-pop" / "musik kpop",
    "Fashion Syar'i" / "fashion syari", "E-sport" / "esport"), and those are the
    same class, not a miss.
    """
    return "".join(c for c in _norm(text) if c.isalnum())


def _split_keywords(raw):
    """The pipe-separated `keywordauto` as a de-duplicated, normalized list."""
    seen, out = set(), []
    for part in (raw or "").split("|"):
        key = _norm(part)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _dedupe(terms):
    seen, out = set(), []
    for term in terms:
        key = _norm(term)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---------- input ----------

def load_records(path):
    """Normalized evaluation records from a JSON or CSV dataset.

    Each record: id, title, content, gold_category, gold_keywords (None when the
    dataset carries no keyword labels).
    """
    if path.suffix.lower() == ".csv":
        return _records_from_csv(path)
    return _records_from_json(path)


def _records_from_json(path):
    """input/apis-data-all.json: `content`, `categoryauto`, `keywordauto`."""
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return [
        {
            "id": article.get("id"),
            "title": article.get("title", ""),
            "content": article.get("content", ""),
            "gold_category": _norm(article.get("categoryauto")),
            "gold_keywords": _split_keywords(article.get("keywordauto")),
        }
        for article in articles
    ]


def _records_from_csv(path):
    """input/test_catauto.csv: `text` is the content, `category` is the label.

    No keyword column, so gold_keywords is None and keyword metrics are skipped.
    Rows without text, or whose category is blank/"nan", are dropped — an unlabelled
    row cannot be scored. A few articles appear twice under one `original_id` with
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
            content = (row.get("text") or "").strip()
            category = _norm(row.get("category"))
            if not content or category in ("", "nan"):
                skipped += 1
                continue
            record_id = row.get("original_id") or row.get("") or str(i)
            if record_id in seen:
                duplicates += 1
                continue
            seen.add(record_id)
            records.append({
                "id": record_id,
                "title": (row.get("title") or "").strip(),
                "content": content,
                "gold_category": category,
                "gold_keywords": None,
            })
    if skipped:
        print(f"  skipped {skipped} row(s) with no text or no category label")
    if duplicates:
        print(f"  skipped {duplicates} duplicate row(s) sharing an original_id")
    return records


# ---------- scoring ----------

def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def _partial_match(a, b):
    """True when two keywords refer to the same thing without being identical.

    Either one contains the other as a whole-word substring ("mi ayam" vs "mi ayam
    micin"), or their word sets overlap by at least half of the smaller set.
    """
    if a == b:
        return True
    if f" {a} " in f" {b} " or f" {b} " in f" {a} ":
        return True
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= 0.5


def score_keyword_sets(predicted, gold, partial=False):
    """TP/FP/FN for one article, matching each prediction to at most one gold term."""
    unmatched = list(gold)
    tp = 0
    for term in predicted:
        for i, gold_term in enumerate(unmatched):
            if _partial_match(term, gold_term) if partial else term == gold_term:
                del unmatched[i]
                tp += 1
                break
    return tp, len(predicted) - tp, len(unmatched)


def category_metrics(pairs):
    """Per-class + macro/weighted P/R/F1 over (gold, predicted) label pairs."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for gold, predicted in pairs:
        if _loose(gold) == _loose(predicted):
            counts[gold]["tp"] += 1
        else:
            counts[gold]["fn"] += 1
            counts[predicted]["fp"] += 1

    per_class = {label: _prf(c["tp"], c["fp"], c["fn"]) for label, c in sorted(counts.items())}
    total = sum(m["support"] for m in per_class.values()) or 1
    correct = sum(c["tp"] for c in counts.values())

    def _avg(key, weighted):
        if weighted:
            return sum(m[key] * m["support"] for m in per_class.values()) / total
        return sum(m[key] for m in per_class.values()) / (len(per_class) or 1)

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "macro": {k: _avg(k, False) for k in ("precision", "recall", "f1")},
        "weighted": {k: _avg(k, True) for k in ("precision", "recall", "f1")},
        "per_class": per_class,
    }


def keyword_metrics(per_article):
    """Micro (pooled) and macro (averaged) scores from per-article TP/FP/FN."""
    micro = _prf(
        sum(a["tp"] for a in per_article),
        sum(a["fp"] for a in per_article),
        sum(a["fn"] for a in per_article),
    )
    n = len(per_article) or 1
    macro = {
        key: sum(a[key] for a in per_article) / n for key in ("precision", "recall", "f1")
    }
    return {"micro": micro, "macro": macro}


# ---------- run ----------

def evaluate_article(record):
    """Generate for one record and score it. Errors are captured, not raised."""
    result = {
        "id": record["id"],
        "title": record["title"],
        "gold_category": record["gold_category"],
        "gold_keywords": record["gold_keywords"],
    }
    try:
        generated, usage = generate_keywords_category(record["content"])
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["predicted_category"] = _norm(generated.get("categoryauto"))
    result["predicted_keywords"] = _dedupe(generated.get("keywordauto") or [])
    result["usage"] = usage

    if record["gold_keywords"] is not None:
        for mode in ("exact", "partial"):
            tp, fp, fn = score_keyword_sets(
                result["predicted_keywords"], record["gold_keywords"], partial=(mode == "partial")
            )
            result[f"keywords_{mode}"] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}
    return result


def summarize(results):
    known_labels = {_loose(leaf) for leaf, _ in _category_labels()}
    scored = [r for r in results if "error" not in r]
    # A gold label absent from the taxonomy can never be predicted, so scoring it
    # would only measure the label file's coverage. Reported separately instead.
    in_taxonomy = [r for r in scored if _loose(r["gold_category"]) in known_labels]
    off_taxonomy = sorted(
        {r["gold_category"] for r in scored if _loose(r["gold_category"]) not in known_labels}
    )
    with_keywords = [r for r in scored if r.get("keywords_exact")]

    summary = {
        "articles": len(results),
        "scored": len(scored),
        "failed": len(results) - len(scored),
        "category": {
            "scored": len(in_taxonomy),
            "gold_labels_not_in_taxonomy": off_taxonomy,
            **category_metrics(
                [(r["gold_category"], r["predicted_category"]) for r in in_taxonomy]
            ),
        },
        "keywords": {
            "scored": len(with_keywords),
            **{
                mode: keyword_metrics([r[f"keywords_{mode}"] for r in with_keywords])
                for mode in ("exact", "partial")
            },
        },
        "usage": _usage_totals(scored),
    }
    return summary


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


def _pct(value):
    return f"{value * 100:5.1f}%"


def print_report(summary, results):
    cat = summary["category"]
    print("\n=== CATEGORY ===")
    print(f"scored {cat['scored']} of {summary['scored']} articles "
          f"(accuracy {cat['correct']}/{cat['total']} = {_pct(cat['accuracy'])})")
    if cat["gold_labels_not_in_taxonomy"]:
        print(f"  gold labels absent from the taxonomy (excluded): "
              f"{', '.join(cat['gold_labels_not_in_taxonomy'])}")
    print(f"{'':34}{'prec':>7}{'rec':>8}{'f1':>8}{'n':>4}")
    for label, m in cat["per_class"].items():
        print(f"  {label[:32]:32}{_pct(m['precision']):>7}{_pct(m['recall']):>8}"
              f"{_pct(m['f1']):>8}{m['support']:>4}")
    for avg in ("macro", "weighted"):
        m = cat[avg]
        print(f"  {avg + ' avg':32}{_pct(m['precision']):>7}{_pct(m['recall']):>8}"
              f"{_pct(m['f1']):>8}")

    print("\n=== KEYWORDS ===")
    if not summary["keywords"]["scored"]:
        print("  no gold keywords in this dataset — skipped")
    else:
        print(f"{'':34}{'prec':>7}{'rec':>8}{'f1':>8}")
        for mode in ("exact", "partial"):
            for avg in ("micro", "macro"):
                m = summary["keywords"][mode][avg]
                print(f"  {mode + ' / ' + avg:32}{_pct(m['precision']):>7}{_pct(m['recall']):>8}"
                      f"{_pct(m['f1']):>8}")

    print("\n=== PER ARTICLE ===")
    for r in results:
        if "error" in r:
            print(f"  [{r['id']}] ERROR {r['error']}")
            continue
        hit = "OK " if _loose(r["gold_category"]) == _loose(r["predicted_category"]) else "MISS"
        usage = r.get("usage") or {}
        keywords = ""
        if r.get("keywords_exact"):
            keywords = (f"  keywords f1 exact {_pct(r['keywords_exact']['f1'])} / "
                        f"partial {_pct(r['keywords_partial']['f1'])}")
        print(f"  [{r['id']}] {hit} category: {r['predicted_category']!r} "
              f"(gold {r['gold_category']!r}){keywords}  "
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
    """title / gold category / predicted category, one row per article."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "categoryauto", "predicted_categoryauto", "match"])
        for r in results:
            predicted = r.get("predicted_category", "")
            writer.writerow([
                r.get("id", ""),
                r.get("title", ""),
                r.get("gold_category", ""),
                r.get("error") or predicted,
                "" if r.get("error")
                else ("1" if _loose(predicted) == _loose(r.get("gold_category")) else "0"),
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="JSON dataset (content/categoryauto/keywordauto) or CSV (text/category)",
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
    args = parser.parse_args()

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]

    print(f"Evaluating {len(records)} article(s) from {args.input} ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_article, records))

    summary = summarize(results)
    print_report(summary, results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Wrote details to {args.out}")

    write_category_csv(results, args.out_csv)
    print(f"Wrote categories to {args.out_csv}")


if __name__ == "__main__":
    main()
