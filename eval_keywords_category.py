"""Evaluate generate_keywords_category() against the labelled sample data.

Runs the function over input/apis-data-all.json and scores its output against
each article's own `categoryauto` / `keywordauto`:

  category  — single-label multiclass: accuracy plus per-class and macro/weighted
              precision, recall, F1.
  keywords  — multi-label set comparison per article: micro (pooled TP/FP/FN) and
              macro (mean of per-article scores), under exact and partial matching.

Only the article's `content` is sent to the model — the same contract the
function has in production. Gold labels are normalized (lowercased, whitespace
collapsed) before comparison, so the CSV's "Kisah Inspiratif" matches the data's
"kisah inspiratif".

Run:  python eval_keywords_category.py [--limit N] [--workers N] [--out FILE]
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


def _gold_keywords(article):
    """The pipe-separated `keywordauto` as a de-duplicated, normalized list."""
    raw = article.get("keywordauto") or ""
    seen, out = set(), []
    for part in raw.split("|"):
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
        if gold == predicted:
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

def evaluate_article(article):
    """Generate for one article and score it. Errors are captured, not raised."""
    result = {
        "id": article.get("id"),
        "title": article.get("title", ""),
        "gold_category": _norm(article.get("categoryauto")),
        "gold_keywords": _gold_keywords(article),
    }
    try:
        generated, usage = generate_keywords_category(article.get("content", ""))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["predicted_category"] = _norm(generated.get("categoryauto"))
    result["predicted_keywords"] = _dedupe(generated.get("keywordauto") or [])
    result["usage"] = usage

    for mode in ("exact", "partial"):
        tp, fp, fn = score_keyword_sets(
            result["predicted_keywords"], result["gold_keywords"], partial=(mode == "partial")
        )
        result[f"keywords_{mode}"] = {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}
    return result


def summarize(results):
    known_labels = {_norm(leaf) for leaf, _ in _category_labels()}
    scored = [r for r in results if "error" not in r]
    # A gold label absent from the taxonomy can never be predicted, so scoring it
    # would only measure the label file's coverage. Reported separately instead.
    in_taxonomy = [r for r in scored if r["gold_category"] in known_labels]
    off_taxonomy = sorted({r["gold_category"] for r in scored} - known_labels)

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
            mode: keyword_metrics([r[f"keywords_{mode}"] for r in scored])
            for mode in ("exact", "partial")
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
        hit = "OK " if r["gold_category"] == r["predicted_category"] else "MISS"
        usage = r.get("usage") or {}
        print(f"  [{r['id']}] {hit} category: {r['predicted_category']!r} "
              f"(gold {r['gold_category']!r})  keywords f1 "
              f"exact {_pct(r['keywords_exact']['f1'])} / "
              f"partial {_pct(r['keywords_partial']['f1'])}  "
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
                "" if r.get("error") else ("1" if predicted == r.get("gold_category") else "0"),
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
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

    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)
    if args.limit:
        articles = articles[: args.limit]

    print(f"Evaluating {len(articles)} article(s) from {args.input} ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_article, articles))

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
