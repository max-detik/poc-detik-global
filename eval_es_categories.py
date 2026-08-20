"""Score the production categoriser's rank-N pick against a reference label.

`output/es-articles.csv` (written by get_from_es.py) carries each article's
`categories_auto`: a list of three ranked guesses, each with its own leaf and
both parent levels. This scores one rank — rank 2 by default, the runner-up —
at leaf, level 2 and level 1, the same three tiers as eval_keywords_category.py.

Two references, reported side by side:

  gold  — the `category` column of input/test_catauto.csv. Note this column is
          the rank-1 leaf itself (identical on all 514 joined rows), so rank 2
          can never match it at leaf level; the level-2 and level-1 numbers say
          how near the runner-up lands.
  llm   — the `predicted_categoryauto` column of output/eval-categories.csv,
          i.e. what generate_keywords_category() chose. This says how often the
          model's answer is the production system's second choice.

The hierarchy for every label is taken from the ES rows themselves (each guess
carries `tree_level1`/`tree_level2`), on top of input/categoryauto_labelling.csv
— the production data uses leaves that file doesn't list.

Run:  python eval_es_categories.py [--rank N] [--no-per-class]
"""

import argparse
import ast
import csv
import sys
from pathlib import Path

from category_scoring import (
    TIERS,
    clean,
    loose,
    norm,
    print_summary,
    print_tier,
    register_hierarchy,
    resolve,
    score_tiers,
)

BASE_DIR = Path(__file__).parent
ES_PATH = BASE_DIR / "output/es-articles.csv"
GOLD_PATH = BASE_DIR / "input/test_catauto.csv"
LLM_PATH = BASE_DIR / "output/eval-categories.csv"
OUTPUT_PATH = BASE_DIR / "output/eval-es-rank.csv"


def read_es_categories(path):
    """{article id: {rank: {leaf, level2, level1, score}}} from the ES export.

    `categories_auto` is a Python literal (as printed by the ES client), not JSON,
    so it is parsed with ast.literal_eval. Rows with an empty cell — ids the
    lookup did not find — are skipped.
    """
    csv.field_size_limit(sys.maxsize)
    by_id, titles, hierarchy, empty = {}, {}, [], 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = clean(row.get("categories_auto"))
            article_id = clean(row.get("_id")) or clean(row.get("id"))
            titles[article_id] = clean(row.get("title"))
            if not raw:
                empty += 1
                continue
            ranked = {}
            for entry in ast.literal_eval(raw):
                leaf = clean(entry.get("leaf"))
                level2 = clean(entry.get("tree_level2"))
                level1 = clean(entry.get("tree_level1"))
                if not leaf:
                    continue
                hierarchy.append((level1, level2, leaf))
                ranked[entry.get("rank")] = {
                    "leaf": leaf,
                    "level2": level2,
                    "level1": level1,
                    "score": entry.get("score"),
                }
            by_id[article_id] = ranked
    if empty:
        print(f"  {empty} row(s) have no categories_auto and were skipped")
    # The ES rows are the authority on where these leaves sit in the tree.
    register_hierarchy(hierarchy)
    return by_id, titles


def read_reference_column(path, id_column, label_column):
    """{article id: label} from a CSV, first row per id winning."""
    csv.field_size_limit(sys.maxsize)
    labels = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {id_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{path} is missing required column(s): {', '.join(sorted(missing))}"
            )
        for row in reader:
            article_id = clean(row.get(id_column))
            label = norm(clean(row.get(label_column)))
            if article_id and label:
                labels.setdefault(article_id, label)
    return labels


def build_rows(es_by_id, references, rank):
    """One row per article that has both a rank-N guess and a reference label."""
    rows = []
    for article_id, ranked in es_by_id.items():
        guess = ranked.get(rank)
        if not guess:
            continue
        reference = references.get(article_id)
        if not reference:
            continue
        rows.append({
            "id": article_id,
            "reference": reference,
            "reference_tiers": resolve(reference),
            "predicted": guess["leaf"],
            # The ES row states the guess's own parents; no lookup needed.
            "predicted_tiers": {
                "leaf": guess["leaf"], "level2": guess["level2"], "level1": guess["level1"]
            },
            "score": guess["score"],
        })
    return rows


def write_csv(path, rows_by_reference, titles, rank):
    """One row per article: the rank-N guess against each reference, per tier."""
    names = list(rows_by_reference)
    ids, seen = [], set()
    for name in names:
        for r in rows_by_reference[name]:
            if r["id"] not in seen:
                seen.add(r["id"])
                ids.append(r["id"])
    indexed = {name: {r["id"]: r for r in rows} for name, rows in rows_by_reference.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        header = ["id", "title", f"rank{rank}_leaf", f"rank{rank}_level2",
                  f"rank{rank}_level1", f"rank{rank}_score"]
        for name in names:
            header += [name, f"match_leaf_{name}", f"match_level2_{name}",
                       f"match_level1_{name}"]
        writer.writerow(header)

        for article_id in ids:
            any_row = next(indexed[n][article_id] for n in names if article_id in indexed[n])
            guess = any_row["predicted_tiers"]
            line = [article_id, titles.get(article_id, ""), guess["leaf"], guess["level2"],
                    guess["level1"], any_row["score"]]
            for name in names:
                row = indexed[name].get(article_id)
                if not row:
                    line += ["", "", "", ""]
                    continue
                line.append(row["reference"])
                for tier in TIERS:
                    reference, predicted = row["reference_tiers"][tier], row["predicted_tiers"][tier]
                    line.append(
                        "" if not (reference and predicted)
                        else ("1" if loose(reference) == loose(predicted) else "0")
                    )
            writer.writerow(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--es", type=Path, default=ES_PATH)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--llm", type=Path, default=LLM_PATH,
                        help="eval_keywords_category.py output; skipped when absent")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--rank", type=int, default=2, help="which ranked guess to score")
    parser.add_argument("--no-per-class", action="store_true",
                        help="only print the accuracy/macro/weighted lines")
    args = parser.parse_args()

    print(f"Reading {args.es} ...")
    es_by_id, titles = read_es_categories(args.es)
    with_rank = sum(1 for ranked in es_by_id.values() if args.rank in ranked)
    print(f"  {len(es_by_id)} article(s), {with_rank} with a rank-{args.rank} guess")

    references = {"gold": read_reference_column(args.gold, "original_id", "category")}
    if args.llm.exists():
        references["llm"] = read_reference_column(args.llm, "id", "predicted_categoryauto")
    else:
        print(f"  {args.llm} not found — scoring against gold only")

    rows_by_reference = {}
    for name, labels in references.items():
        rows = build_rows(es_by_id, labels, args.rank)
        rows_by_reference[name] = rows
        tiers = score_tiers(rows)
        print(f"\n{'=' * 62}")
        print(f"rank {args.rank} vs {name.upper()} — {len(rows)} article(s)")
        print("=" * 62)
        print_summary(tiers, label=f"SUMMARY (rank {args.rank} vs {name})")
        for tier in TIERS:
            print_tier(tier, tiers[tier], len(rows), per_class=not args.no_per_class,
                       prefix=f"rank{args.rank} vs {name}")

    write_csv(args.out, rows_by_reference, titles, args.rank)
    print(f"\nWrote details to {args.out}")


if __name__ == "__main__":
    main()
