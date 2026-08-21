"""Fetch article fields from Elasticsearch for the ids in a local CSV.

Reads `original_id` from input/test_catauto.csv, looks each id up in the
`news*` indices, and writes id / title / categories_auto to a CSV.

Run:  python -m scripts.get_from_es [--limit N] [--input FILE] [--out FILE]
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "input/test_catauto.csv"
OUTPUT_PATH = ROOT / "output/es-articles.csv"

# GCP / ES - 7, detikcom. The calls below use the elasticsearch-py 7.x style
# (`body=` + `_source=`); the 8.x keyword form is a TypeError on that client.
ES_URL = os.getenv("ES_URL", "http://10.250.1.170:80/")
INDEX = "news*"
SOURCE_FIELDS = ["id", "title", "original.categories_auto"]
# An ids query is capped by index.max_result_window (10k by default), so ask in
# batches rather than for one page as large as the id list.
BATCH_SIZE = 500


def read_ids(path, limit=None):
    """Unique, non-empty `original_id` values from the CSV, in file order.

    Ids are returned as strings: an Elasticsearch `_id` is a string, and reading
    the column as a number would turn a missing value into a float and match
    "8296403.0" against nothing.
    """
    df = pd.read_csv(path, usecols=["original_id"], dtype={"original_id": "string"})
    ids = df["original_id"].dropna().str.strip()
    ids = ids[ids != ""].drop_duplicates().tolist()
    return ids[:limit] if limit else ids


def fetch_articles(es, ids, index=INDEX, batch_size=BATCH_SIZE):
    """[{id, title, categories_auto}] for the ids that exist in `index`."""
    rows = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        response = es.search(
            index=index,
            size=len(batch),
            _source=SOURCE_FIELDS,
            body={"query": {"ids": {"values": batch}}},
        )
        for hit in response["hits"]["hits"]:
            source = hit.get("_source") or {}
            original = source.get("original") or {}
            rows.append({
                "_id": hit["_id"],
                "id": source.get("id"),
                "title": source.get("title"),
                "categories_auto": original.get("categories_auto"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, help="only look up the first N ids")
    parser.add_argument("--url", default=ES_URL, help="Elasticsearch URL")
    args = parser.parse_args()

    ids = read_ids(args.input, args.limit)
    if not ids:
        raise SystemExit(f"no usable original_id values in {args.input}")

    es = Elasticsearch(args.url)
    print(f"Looking up {len(ids)} id(s) in {INDEX} ...", flush=True)
    rows = fetch_articles(es, ids)

    df = pd.DataFrame(rows, columns=["_id", "id", "title", "categories_auto"])
    missing = [i for i in ids if i not in set(df["_id"])]
    if missing:
        shown = ", ".join(missing[:10]) + (f", +{len(missing) - 10} more" if len(missing) > 10 else "")
        print(f"  {len(missing)} id(s) not found: {shown}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.shape)
    print(df.head(2))
    print(f"Wrote {len(df)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
