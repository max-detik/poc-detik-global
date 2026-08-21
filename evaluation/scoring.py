"""Shared category scoring: taxonomy resolution and multiclass metrics.

Both evaluation scripts score a predicted category against a reference one at
three levels of the taxonomy — leaf, level 2, level 1 — so a sibling-leaf error
still counts as a hit further up the tree. The pieces live here so the two
scripts report identical numbers.

The tree is read from input/categoryauto_labelling.csv. Extra leaf -> parent
mappings can be registered at runtime (see register_hierarchy) for labels that
exist in the data but not in that file.
"""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_LABELS_PATH = ROOT / "input/categoryauto_labelling.csv"

TIERS = ("leaf", "level2", "level1")
TIER_TITLES = {"leaf": "LEAF", "level2": "LEVEL 2", "level1": "LEVEL 1"}

# Pandas writes missing values out as these strings; they are not labels.
MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a", "<na>", "-"}


# ---------- normalization ----------

def norm(text):
    """Lowercased, whitespace-collapsed comparison key."""
    return " ".join((text or "").split()).lower()


def loose(text):
    """Category matching key: letters and digits only.

    The datasets spell the same label several ways ("Musik K-pop" / "musik kpop",
    "Fashion Syar'i" / "fashion syari", "E-sport" / "esport"), and those are the
    same class, not a miss.
    """
    return "".join(c for c in norm(text) if c.isalnum())


def is_missing(value):
    """True for an empty cell or a stringified null."""
    return norm(value) in MISSING_VALUES


def clean(value):
    """The cell's text, or "" when it is empty or a stringified null."""
    return "" if is_missing(value) else (value or "").strip()


# ---------- taxonomy ----------

_nodes = None
_extra = []


def _column(row, *names):
    """Row value by header name, tolerant of case and spacing ("level 1"/"Level1")."""
    keys = {"".join((k or "").split()).lower(): k for k in row}
    for name in names:
        key = keys.get("".join(name.split()).lower())
        if key is not None:
            return (row[key] or "").strip()
    return ""


def read_taxonomy(path=CATEGORY_LABELS_PATH):
    """[(level1, level2, leaf)] from the labelling CSV, in file order."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = [
            (_column(row, "level 1"), _column(row, "level 2"), _column(row, "Leaf"))
            for row in csv.DictReader(f)
        ]
    return [r for r in rows if r[2]]


def register_hierarchy(triples):
    """Add (level1, level2, leaf) mappings on top of the labelling CSV.

    The production data labels articles with leaves the CSV doesn't list (e.g.
    "berita hiburan"), but carries their parents alongside. Registering those
    lets such a label be scored instead of dropped.
    """
    _extra.extend(t for t in triples if t[2])
    global _nodes
    _nodes = None


def _tier_nodes():
    """Loose key -> {leaf, level2, level1} for every node in the taxonomy.

    A label is looked up at whatever depth it names: a leaf resolves all three
    tiers, a level-2 node resolves two, a level-1 node resolves one. Data labelled
    with a parent rather than a leaf can still be scored at the tiers where it is
    defined.
    """
    global _nodes
    if _nodes is None:
        nodes = {}
        # Registered mappings first, the labelling CSV last: where both define a
        # leaf, the CSV's spelling is the canonical one for display.
        for level1, level2, leaf in _extra + list(read_taxonomy()):
            if level1:
                nodes.setdefault(loose(level1), {"leaf": None, "level2": None, "level1": level1})
            if level2:
                nodes.setdefault(
                    loose(level2), {"leaf": None, "level2": level2, "level1": level1}
                )
            # A leaf wins over a parent of the same name ("Kesehatan" is both a
            # level-1 heading and a leaf), so this assignment is not setdefault.
            nodes[loose(leaf)] = {"leaf": leaf, "level2": level2, "level1": level1}
        _nodes = nodes
    return _nodes


def resolve(label):
    """The label's position in the taxonomy; all tiers None when it is unknown."""
    return _tier_nodes().get(loose(label), {"leaf": None, "level2": None, "level1": None})


# ---------- metrics ----------

def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def category_metrics(pairs):
    """Per-class + macro/weighted P/R/F1 over (reference, predicted) label pairs."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for reference, predicted in pairs:
        if loose(reference) == loose(predicted):
            counts[reference]["tp"] += 1
        else:
            counts[reference]["fn"] += 1
            counts[predicted]["fp"] += 1

    per_class = {label: prf(c["tp"], c["fp"], c["fn"]) for label, c in sorted(counts.items())}
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


def score_tiers(rows):
    """Metrics per tier from rows of {"reference_tiers", "predicted_tiers"}.

    A row is scored at a tier only when both sides resolve there; the labels that
    don't are reported instead, so the numbers never silently exclude data.
    """
    tiers = {}
    for tier in TIERS:
        pairs, unresolved = [], set()
        for r in rows:
            reference = r["reference_tiers"][tier]
            predicted = r["predicted_tiers"][tier]
            if reference and predicted:
                pairs.append((reference, predicted))
            else:
                unresolved.add(r["reference"] if not reference else r["predicted"])
        tiers[tier] = {
            "scored": len(pairs),
            "labels_not_in_taxonomy": sorted(x for x in unresolved if x),
            **category_metrics(pairs),
        }
    return tiers


def rank_metrics(rows):
    """Hit@k and hit-rank stats for rows carrying a ranked candidate list.

    Each row needs "reference_tiers" and "candidate_tiers" (a list, best first).
    A row counts at a tier once its reference resolves there; a candidate that
    doesn't resolve simply cannot match. hit@k asks whether any of the first k
    candidates matches — the question "would the right label be on the menu?" —
    while accuracy on candidate 1 stays the single-label number.
    """
    depth = max((len(r["candidate_tiers"]) for r in rows), default=0)
    out = {}
    for tier in TIERS:
        scored, hits_at, ranks = 0, [0] * depth, []
        for r in rows:
            reference = r["reference_tiers"][tier]
            if not reference:
                continue
            scored += 1
            hit_rank = None
            for i, candidate in enumerate(r["candidate_tiers"]):
                if candidate[tier] and loose(candidate[tier]) == loose(reference):
                    hit_rank = i + 1
                    break
            ranks.append(hit_rank)
            if hit_rank:
                for i in range(hit_rank - 1, depth):
                    hits_at[i] += 1
        n = scored or 1
        out[tier] = {
            "scored": scored,
            "hit_at": {k + 1: hits_at[k] / n for k in range(depth)},
            "hit_at_counts": {k + 1: hits_at[k] for k in range(depth)},
            "rank_histogram": {
                **{k + 1: sum(1 for r in ranks if r == k + 1) for k in range(depth)},
                "none": sum(1 for r in ranks if r is None),
            },
            # Mean reciprocal rank: 1.0 when the reference is always first.
            "mrr": sum(1 / r for r in ranks if r) / n,
        }
    return out


# ---------- reporting ----------

def pct(value):
    return f"{value * 100:5.1f}%"


def print_summary(tiers, label="SUMMARY"):
    print(f"\n=== {label} ===")
    print(f"{'':34}{'acc':>7}{'macro f1':>10}{'wtd f1':>9}{'n':>6}")
    for tier in TIERS:
        m = tiers[tier]
        print(f"  {TIER_TITLES[tier]:32}{pct(m['accuracy']):>7}{pct(m['macro']['f1']):>10}"
              f"{pct(m['weighted']['f1']):>9}{m['scored']:>6}")


def print_ranked_summary(ranked, depth, label="RANKED CATEGORIES"):
    """hit@1..k per tier — how often the reference is anywhere in the list."""
    print(f"\n=== {label} ===")
    header = "".join(f"{'hit@' + str(k):>9}" for k in range(1, depth + 1))
    print(f"{'':34}{header}{'mrr':>8}{'n':>6}")
    for tier in TIERS:
        m = ranked[tier]
        cells = "".join(f"{pct(m['hit_at'].get(k, 0.0)):>9}" for k in range(1, depth + 1))
        print(f"  {TIER_TITLES[tier]:32}{cells}{m['mrr']:>8.3f}{m['scored']:>6}")
    print(f"\n{'':34}{'rank of the matching label':<34}")
    for tier in TIERS:
        hist = ranked[tier]["rank_histogram"]
        parts = [f"#{k}: {hist[k]}" for k in range(1, depth + 1)] + [f"none: {hist['none']}"]
        print(f"  {TIER_TITLES[tier]:32}{'  '.join(parts)}")


def print_tier(tier, metrics, total, per_class=True, prefix="CATEGORY"):
    print(f"\n=== {prefix} / {TIER_TITLES[tier]} ===")
    print(f"scored {metrics['scored']} of {total} articles "
          f"(accuracy {metrics['correct']}/{metrics['total']} = {pct(metrics['accuracy'])})")
    if metrics["labels_not_in_taxonomy"]:
        labels = metrics["labels_not_in_taxonomy"]
        shown = ", ".join(labels[:8]) + (f", +{len(labels) - 8} more" if len(labels) > 8 else "")
        print(f"  labels the taxonomy has no {TIER_TITLES[tier].lower()} for (excluded): {shown}")

    print(f"{'':34}{'prec':>7}{'rec':>8}{'f1':>8}{'n':>4}")
    if per_class:
        # Classes with no reference rows are predictions only; listed after the
        # table rather than padding it with zero-support lines.
        graded = {k: m for k, m in metrics["per_class"].items() if m["support"]}
        for label, m in sorted(graded.items(), key=lambda kv: (-kv[1]["support"], kv[0])):
            print(f"  {label[:32]:32}{pct(m['precision']):>7}{pct(m['recall']):>8}"
                  f"{pct(m['f1']):>8}{m['support']:>4}")
        spurious = [k for k, m in metrics["per_class"].items() if not m["support"]]
        if spurious:
            shown = ", ".join(sorted(spurious)[:8])
            print(f"  ({len(spurious)} label(s) predicted but never reference: {shown}"
                  + (", ...)" if len(spurious) > 8 else ")"))
    for avg in ("macro", "weighted"):
        m = metrics[avg]
        print(f"  {avg + ' avg':32}{pct(m['precision']):>7}{pct(m['recall']):>8}"
              f"{pct(m['f1']):>8}")
