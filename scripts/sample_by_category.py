"""Sample N articles per `categoryauto_new` from the training CSV.

Streams input/training-categoryauto-v7.csv (~500 MB, so it is never loaded whole)
and keeps a reservoir of N rows per category, writing id / title / content /
categoryauto_new for the sampled rows.

Run:  python -m scripts.sample_by_category [--n 10] [--source FILE] [--out FILE]
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "input/training-categoryauto-v7.csv"
OUTPUT_PATH = ROOT / "input/sample_categoryauto_v7.csv"

CATEGORY_COLUMN = "categoryauto_new"
KEEP_COLUMNS = ["id", "title", "content", CATEGORY_COLUMN]
SEED = 42

# A single `content` cell runs well past the 128 KB default.
csv.field_size_limit(sys.maxsize)


def sample_per_category(path, n, column=CATEGORY_COLUMN, seed=SEED):
    """(rows, seen) — up to `n` rows per category, and the per-category row count.

    Reservoir sampling: every row of a category has the same chance of being
    kept, in one pass and holding only n rows per category in memory.
    """
    rng = random.Random(seed)
    reservoir = defaultdict(list)
    seen = Counter()

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in KEEP_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} has no column(s): {', '.join(missing)}")

        for row in reader:
            category = (row.get(column) or "").strip()
            # Rows with no category, or with no text to classify, are not usable.
            if not category or not (row.get("content") or "").strip():
                continue

            seen[category] += 1
            kept = reservoir[category]
            picked = {c: row.get(c) for c in KEEP_COLUMNS}
            if len(kept) < n:
                kept.append(picked)
            else:
                slot = rng.randrange(seen[category])
                if slot < n:
                    kept[slot] = picked

    rows = [row for category in sorted(reservoir) for row in reservoir[category]]
    return rows, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--n", type=int, default=10, help="rows per category")
    parser.add_argument("--seed", type=int, default=SEED, help="0 for a different draw")
    args = parser.parse_args()

    print(f"Reading {args.source} ...", flush=True)
    rows, seen = sample_per_category(args.source, args.n, seed=args.seed)
    if not rows:
        raise SystemExit(f"no rows with a {CATEGORY_COLUMN} value in {args.source}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    short = sorted(c for c in seen if seen[c] < args.n)
    print(f"{len(seen)} categories, {sum(seen.values())} usable rows read")
    if short:
        listed = ", ".join(f"{c} ({seen[c]})" for c in short[:10])
        more = f", +{len(short) - 10} more" if len(short) > 10 else ""
        print(f"  {len(short)} category/ies under {args.n} rows: {listed}{more}")
    print(f"Wrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
