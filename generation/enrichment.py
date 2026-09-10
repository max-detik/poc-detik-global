"""TL;DR, Key Takeaway, and FAQ generation from an article's title and content.

generate_enrichment() runs all three prompts over the same title/content and returns
them together with lightweight QA diagnostics: a spec check per section (the length/count
constraints the response schema doesn't strictly guarantee) and a grounding check (a
low word-overlap heuristic that flags a generated string as possibly unmoored from the
source — not a substitute for manual review).

Separate from news.py and keywords_category.py: different prompts, different schemas,
and — unlike those two — this reads a single already-written article (source language
either Indonesian or English) rather than merging multiple sources. The shared
OpenRouter plumbing is in llm.py.
"""

from typing import List

from pydantic import BaseModel, Field

from generation.llm import (
    OPENROUTER_MODEL,
    OPENROUTER_REASONING_EFFORT,
    OPENROUTER_TEMPERATURE,
    client,
    load_prompt,
    parse_response,
)
from openai.types.shared_params import Reasoning

TLDR_BULLETS = 3
TLDR_MAX_CHARS = 150
TAKEAWAY_MAX_CHARS = 500
FAQ_MIN_ITEMS, FAQ_MAX_ITEMS = 3, 6
FAQ_ANSWER_MAX_CHARS = 200

# Grounding heuristic: a word only counts as "found" if it's long enough to be
# distinctive (short words match by coincidence too often to be evidence).
GROUNDING_MIN_WORD_LEN = 5
GROUNDING_THRESHOLD = 3


class TldrSchema(BaseModel):
    tldr: List[str] = Field(
        min_length=TLDR_BULLETS,
        max_length=TLDR_BULLETS,
        description=f"exactly {TLDR_BULLETS} narrative bullet points, each <= {TLDR_MAX_CHARS} characters",
    )


class TakeawaySchema(BaseModel):
    key_takeaway: str = Field(
        ..., description=f"exactly one sentence, may be long, <= {TAKEAWAY_MAX_CHARS} characters"
    )


class FaqItem(BaseModel):
    question: str = Field(..., description="phrased the way a reader would search on Google")
    evidence: str = Field(..., description="short paraphrase of the article part supporting the answer")
    answer: str = Field(..., max_length=FAQ_ANSWER_MAX_CHARS, description="1-2 sentences, factual")


class FaqSchema(BaseModel):
    faq: List[FaqItem] = Field(
        min_length=FAQ_MIN_ITEMS,
        max_length=FAQ_MAX_ITEMS,
        description=f"{FAQ_MIN_ITEMS} to {FAQ_MAX_ITEMS} Q&A pairs",
    )


def _user_message(title, content):
    return f"Title: {title}\n\nFull content:\n{content}"


def _call(prompt_name, schema, title, content):
    api = client()
    messages = [
        {"role": "system", "content": load_prompt(prompt_name)},
        {"role": "user", "content": _user_message(title, content)},
    ]
    response = api.responses.parse(
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        input=messages,
        text_format=schema,
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )
    return parse_response(response)


def check_grounding(texts, content, threshold=GROUNDING_THRESHOLD):
    """Flags strings with low word-overlap against the source article.

    Rough heuristic only — not a substitute for manual review.
    """
    content_lower = (content or "").lower()

    def overlap(text):
        words = [w.strip(".,").lower() for w in text.split() if len(w) > GROUNDING_MIN_WORD_LEN]
        return sum(1 for w in words if w in content_lower)

    return [t for t in texts if overlap(t) < threshold]


def check_tldr_spec(bullets):
    issues = []
    if len(bullets) != TLDR_BULLETS:
        issues.append(f"Expected {TLDR_BULLETS} bullets, got {len(bullets)}")
    for i, b in enumerate(bullets, 1):
        if len(b) > TLDR_MAX_CHARS:
            issues.append(f"Bullet #{i} > {TLDR_MAX_CHARS} chars ({len(b)})")
    return issues


def check_takeaway_spec(sentence):
    issues = []
    if len(sentence) > TAKEAWAY_MAX_CHARS:
        issues.append(f"Sentence > {TAKEAWAY_MAX_CHARS} chars ({len(sentence)})")
    return issues


def check_faq_spec(faqs):
    issues = []
    if not (FAQ_MIN_ITEMS <= len(faqs) <= FAQ_MAX_ITEMS):
        issues.append(f"Expected {FAQ_MIN_ITEMS}-{FAQ_MAX_ITEMS} Q&As, got {len(faqs)}")
    for i, qa in enumerate(faqs, 1):
        if len(qa["answer"]) > FAQ_ANSWER_MAX_CHARS:
            issues.append(f"FAQ #{i} answer > {FAQ_ANSWER_MAX_CHARS} chars ({len(qa['answer'])})")
        if not qa.get("evidence", "").strip():
            issues.append(f"FAQ #{i} has empty evidence field")
    return issues


def generate_tldr(title, content):
    """([bullet, bullet, bullet], token_usage)."""
    result, usage = _call("tldr", TldrSchema, title, content)
    return result["tldr"], usage


def generate_takeaways(title, content):
    """(sentence, token_usage)."""
    result, usage = _call("takeaways", TakeawaySchema, title, content)
    return result["key_takeaway"], usage


def generate_faq(title, content):
    """([{question, evidence, answer}, ...], token_usage)."""
    result, usage = _call("faq", FaqSchema, title, content)
    return result["faq"], usage


def _add_usage(total, usage):
    total["input_tokens"] += usage.get("input_tokens") or 0
    total["output_tokens"] += usage.get("output_tokens") or 0
    total["total_tokens"] += usage.get("total_tokens") or 0
    total["cost"] = (total["cost"] or 0.0) + (usage.get("cost") or 0.0)


def generate_enrichment(title, content):
    """TL;DR, Key Takeaway, and FAQ for one article, plus QA diagnostics.

    `title`/`content` are plain strings (content may be HTML), in any language.
    Returns (result, usage). `result` holds `tldr`, `key_takeaway`, `faq`, and a
    `diagnostics` dict of spec/grounding flags — heuristic checks run locally, not
    extra model calls. `usage` sums tokens/cost across the three underlying calls.
    """
    title = (title or "").strip()
    content = (content or "").strip()
    if not content:
        raise ValueError("content must not be empty")

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}

    tldr, tldr_usage = generate_tldr(title, content)
    _add_usage(usage, tldr_usage)

    key_takeaway, takeaway_usage = generate_takeaways(title, content)
    _add_usage(usage, takeaway_usage)

    faq, faq_usage = generate_faq(title, content)
    _add_usage(usage, faq_usage)

    diagnostics = {
        "tldr_spec_issues": check_tldr_spec(tldr),
        "tldr_grounding_flags": check_grounding(tldr, content),
        "takeaway_spec_issues": check_takeaway_spec(key_takeaway),
        "takeaway_grounding_flags": check_grounding([key_takeaway], content),
        "faq_spec_issues": check_faq_spec(faq),
        "faq_grounding_flags": check_grounding([qa["answer"] for qa in faq], content),
    }

    result = {"tldr": tldr, "key_takeaway": key_takeaway, "faq": faq, "diagnostics": diagnostics}
    return result, usage
