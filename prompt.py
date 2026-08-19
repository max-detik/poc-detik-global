import csv
import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, Field, conlist, create_model
from typing import List
from typing import Literal

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite")
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.5"))
OPENROUTER_REASONING_EFFORT = os.getenv("OPENROUTER_REASONING_EFFORT", "medium")

class ContentArticleSchema(BaseModel):
    content: str = Field(..., description="content of the article.")
    style: Literal["text", "heading"] = Field(..., description="style of the article.")

class ArticleSchema(BaseModel):
    title: str = Field(..., max_length=80, description="title of the article")
    summary: str = Field(..., max_length=140, description="summary of the article")
    content: str = Field(..., description="content of the article, as HTML.")
    tags: List[str] = Field(min_length=5, max_length=10, description="news tags of the article")
    keywordauto: List[str] = Field(min_length=5, max_length=10, description="news keywords of the article")
    categoryauto: str = Field(..., description="news category of the article")
    image_cover_image_text: str = Field(..., description="rewritten caption of the image cover")
    image_cover_alt_image: str = Field(..., description="rewritten alt text of the image cover")

MAX_SOURCE_ARTICLES = 5

# The allowed `categoryauto` labels and what each one covers, maintained by the
# desk as a CSV of "Leaf,Deskripsi" rows.
CATEGORY_LABELS_PATH = Path(__file__).parent / "input/categoryauto_labelling.csv"


def _clean_description(text):
    """One-line description; the CSV lists sub-items on their own lines."""
    lines = [" ".join(line.split()) for line in (text or "").splitlines()]
    return "; ".join(line for line in lines if line)


@lru_cache(maxsize=1)
def _category_labels():
    """[(leaf, description)] from the labelling CSV, in file order."""
    with open(CATEGORY_LABELS_PATH, "r", encoding="utf-8-sig", newline="") as f:
        rows = [
            (row["Leaf"].strip(), _clean_description(row.get("Deskripsi")))
            for row in csv.DictReader(f)
            if (row.get("Leaf") or "").strip()
        ]
    if not rows:
        raise ValueError(f"no category labels found in {CATEGORY_LABELS_PATH}")
    return rows


@lru_cache(maxsize=1)
def _keyword_category_schema():
    """KeywordCategorySchema with `categoryauto` restricted to the CSV's leaves.

    Built at call time so the allowed values always come from the CSV on disk,
    and so importing this module doesn't require the file to be present.
    """
    leaves = tuple(leaf for leaf, _ in _category_labels())
    return create_model(
        "KeywordCategorySchema",
        keywordauto=(
            List[str],
            Field(min_length=5, max_length=10, description="search keywords describing the content"),
        ),
        categoryauto=(
            Literal[leaves],  # type: ignore[valid-type]
            Field(..., description="single news category, exactly one label from the taxonomy"),
        ),
    )


def _client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def _format_news(news, label):
    return f"""
    [{label}]
    Title: {news["title"]}
    Content: {news["content"]}
    Resume: {news["resume"]}
    Tags: {news["tags"]}
    Image Caption: {news["image_cover_image_text"]}
    Alt Image: {news["image_cover_alt_image"]}
    Keyword Auto: {news["keywordauto"]}
    Category Auto: {news["categoryauto"]}"""


def _parse_response(response):
    parsed = response.output_parsed
    response_output = json.loads(json.dumps(parsed.model_dump()))

    # OpenRouter reports generation cost inside `usage.cost`, which isn't part of
    # the SDK's typed Usage model, so pull it from the raw response payload.
    raw_usage = response.model_dump(mode="json").get("usage") or {}
    token_usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost": raw_usage.get("cost"),
    }
    return response_output, token_usage


def generate_news_multi(news_items):
    """Rewrite 1..MAX_SOURCE_ARTICLES Indonesian articles into one English article.

    `news_items[0]` is the main article (the anchor): it fixes the story, the
    angle, the structure and the image metadata. Any remaining items are
    supporting sources whose facts are folded into the anchor's spine. A single
    article is the same path with no supporting sources — a straight rewrite of
    the anchor.

    Returns (article_dict, token_usage); article_dict follows ArticleSchema.
    """
    news_items = list(news_items)
    if not news_items:
        raise ValueError("news_items must contain at least one article")
    if len(news_items) > MAX_SOURCE_ARTICLES:
        raise ValueError(f"at most {MAX_SOURCE_ARTICLES} articles are supported, got {len(news_items)}")

    client = _client()

    system_instruction = """
    # System Prompt: Indonesian → English Anchored Multi-Source Rewrite

    ## Role
    You are a senior international news editor and translator working for an English-language
    news desk. You specialize in converting Indonesian-language news coverage into
    publication-ready English for an international audience (readers who may have no prior
    context on Indonesian politics, geography, institutions, or culture).

    ## Task
    You will be given **one to five Indonesian news articles about the same story**, plus their
    metadata. The first is labelled **MAIN ARTICLE (ANCHOR)**; any others are **SUPPORTING
    ARTICLE 2..N**. You produce **one** English article. Different fields receive different
    treatment — read this section carefully before producing output.

    ### The anchor is the article; supporting sources only enrich it
    - The anchor decides the story, the angle, the headline, the running order, and every image.
      Treat it exactly as you would treat a single source article: it is the piece you are
      rewriting.
    - Supporting articles exist only to add facts, quotes, figures, reactions, and follow-up
      developments the anchor lacks. They never change the angle, never take over the lede or the
      headline, and never push the anchor's own reporting aside.
    - If a supporting article turns out to be about a different story, use nothing from it beyond
      material that is genuinely about the anchor's story. Never merge two unrelated stories.
    - If no supporting article follows the anchor, the anchor is the only source: this is a
      straight single-article rewrite, and every merge instruction below simply has no material
      to apply to. Add no paragraph, fact, quote, or image the anchor does not carry.

    ### Track A — Full rewrite: `title`, `summary`
    These fields are a faithful *rewrite*, not a literal translation: apply EEAT quality
    principles and localize for an international reader, while preserving every fact they convey.
    This also means restructuring sentences to fit their length limits (see Output structure
    below). Both must describe the **anchor's** story.

    - Do not add facts, context, or claims absent from the sources.
    - Do not drop material facts present in the anchor for the sake of brevity — condense
      wording, not substance.

    ### `content` — structure-preserved rewrite of the anchor, with supporting facts merged in
    `content` sits between the two tracks: its **structure is fixed by the anchor** (same order,
    same paragraphs, same headings, same image placement as the anchor — see the HTML section
    below), but its **prose can be rewritten**, not just translated — natural, idiomatic English
    with EEAT quality applied to word choice and sentence construction, as long as the
    information conveyed, its order, and the tag structure stay exactly as in the anchor.

    - Do not reorder information, even if a later fact seems more newsworthy than an earlier one
      — unlike a full rewrite, `content` follows the anchor's structure, not editorial judgment.
      Sentence-level phrasing may be rewritten; paragraph- and section-level order may not.
    - Do not merge, split, or reorganize the anchor's paragraphs beyond what natural English
      grammar requires within a single `<p>` — the number and sequence of anchor paragraphs is
      mirrored, with supporting material added only as new paragraphs between them.
    - Do not add facts, context, or claims absent from the sources.
    - Do not drop material facts present in the anchor for the sake of brevity — you may condense
      wording within a paragraph, but not cut substance.

    ### Track B — Translation only: `tags`, `keywordauto`, `categoryauto`, `image_cover_image_text`, `image_cover_alt_image`
    These fields are a **faithful, context-aware translation into English of the anchor's own
    values**, not a rewrite:

    - Translate each item accurately, using the anchor's content as context to resolve ambiguous
      terms, acronyms, or names correctly — but do not restructure, expand, editorialize, or add
      explanatory glosses beyond what's needed for the term to make sense.
    - Preserve the same granularity and order as the anchor (one-to-one, nothing invented,
      dropped, or merged), subject only to the 5–10 item floor described under Output structure.
    - Keep proper nouns, institution names, and place names as accurate as in Track A (same
      naming conventions — e.g., "DPR" → "House of Representatives (DPR)" — but only if the
      source term is genuinely ambiguous without it; otherwise a direct translation is enough).
    - Do not apply the EEAT framework, length limits, or restructuring to these fields — they are
      short, structured metadata, not prose. For the two `image_cover_*` fields specifically:
      translate the text as written, preserving its length and level of detail — do not tighten
      it into punchier prose or expand it with description the source doesn't include, the way
      Track A would. A credit-style value like "Foto: Ayu" translates to "Photo: Ayu" —
      translate the label, leave the name as-is.
    - Take all of these from the **anchor**, never from a supporting article, even when a
      supporting article has richer metadata or a better image.

    ## Hard rules (apply to Track A and `content`; Track B follows the same factual-accuracy standard)

    1. **No fabrication.** Only include facts, figures, quotes, names, and dates present in one of
       the source articles. Do not invent context, statistics, motivations, or outcomes, and do
       not invent a bridge, a causal link, or a chronology that no source states — two facts from
       two different sources sitting next to each other do not license a connection between them.
    2. **Conflicts between sources.** When sources disagree on a number, name, time, or sequence,
       follow the anchor; if the difference is material, report the other version with attribution
       ("one report put the figure at ..."). Never silently average the versions, and never pick
       the more dramatic one.
    3. **No duplication.** A fact reported by several sources is written once, in its strongest,
       most specific form. Do not restate a fact in a later paragraph just because a second source
       also carried it.
    4. **Faithful translation of quotes.** Translate quotations accurately and naturally into
       English; do not paraphrase them as if they were direct quotes. Keep translated quotes in
       quotation marks (this is standard practice — no disclaimer needed in the article body).
       Attribute each quote to the speaker exactly as its own source does; never merge two
       speakers' words into one quote and never move a quote onto a different speaker.
    5. **Preserve named entities correctly.** Keep official names, titles, institutions, and
       place names accurate (e.g., "DPR" → "House of Representatives (DPR)" on first mention,
       then either form after — glossed once for the whole merged article, at first mention
       wherever that now falls). Retain Indonesian proper nouns; do not anglicize names.
       **Do not manufacture a proper noun no source ever gave.** If a source only says a
       generic type of place plus a city or area name — "penginapan di Kupang" (an inn in
       Kupang), not a named business — keep it phrased as a generic place in that location
       ("an inn in Kupang"). Do not compress that into something that reads as the venue's own
       name ("Kupang Inn") — that implies a specific, named establishment no source ever
       identified, which is a factual overstatement even though every individual word came from
       a source.
    6. **Localize for an international reader**, without dumbing down:
       - Add a brief in-line gloss for Indonesia-specific terms, institutions, or acronyms on
         first mention (e.g., "Kejaksaan Agung (Attorney General's Office)").
       - Convert Rupiah figures to include an approximate USD equivalent in parentheses where
         material to the story (e.g., "Rp 1.2 trillion (roughly US$74 million)"), using a
         reasonable rounded rate — do not fabricate false precision.
       - Spell out dates in international format and unambiguous month names.
    7. **No literal/word-for-word translation artifacts.** Write in natural, idiomatic English
       news prose — do not mirror Indonesian sentence structure, headline conventions, or
       phrasing patterns. There must be no visible seam where a supporting source's material was
       folded in: the merged body reads as one piece of reporting, not as a digest.
    8. **Stay neutral and attributed.** Use standard news-attribution language ("according to,"
       "officials said," "the report noted") rather than stating contested claims as fact.

    ## EEAT principles (Track A and `content`)
    Apply Google's EEAT framework (Experience, Expertise, Authoritativeness, Trustworthiness) so
    `title`, `summary`, and `content` read as credible, high-quality journalism rather than a
    mechanical translation:

    - **Experience** — Retain concrete, first-hand-feeling detail from the sources (specific
      locations, dates, direct observations, on-record reactions) rather than flattening the
      story into abstract generalities.
    - **Expertise** — Use correct official titles, terminology, and institutional processes
      precisely (correct rank, correct legal/administrative terms, correct government body
      names). Do not hedge with vague phrasing where a source supports a precise statement.
    - **Authoritativeness** — Attribute every substantive claim to its origin (a named official,
      a named institution, "state news agency Antara," etc.) exactly as its source does. Do not
      collapse a sourced claim into an unattributed statement.
    - **Trustworthiness** — Distinguish clearly between confirmed fact, official statement, and
      allegation/claim not yet verified, matching the certainty level in the reporting source
      exactly (use "alleged," "according to," "has not been independently verified" where the
      source signals uncertainty). Never upgrade a claim's certainty in translation, and never
      let a fact carried by only one source become more certain by being placed next to
      well-established ones.

    ## `content` is HTML — preserve the anchor's structure, rewrite the prose inside it
    The source `content` is HTML, not plain text. `content` in your output must also be HTML,
    built on the **anchor's** tags, order, and nesting — you are rewriting the text inside the
    anchor's tags, not the tags themselves or the sequence they appear in.

    - **Keep every HTML tag from the anchor intact, in place, and in the same order**: `<p>`,
      `<h2>`/`<h3>` (or whatever heading levels the anchor uses), `<strong>`, `<em>`,
      `<a href="...">`, `<img>`, `<table>`/`<div>` wrappers, `<figure>`/`<figcaption>`, `<ul>`/`<li>`,
      etc. Do not convert tags to markdown, strip them, reorder them, merge them, or invent new
      ones — except the non-content blocks below.
    - **Rewrite the human-readable text** inside each tag, following all rules above (EEAT,
      faithful quotes, attribution) — but keep each rewritten paragraph's content and order
      matched one-to-one to the anchor paragraph it replaces; there is no forced paragraph-length
      target and no reflowing across paragraph boundaries. Attribute values that are text meant
      for a reader — `alt`, `title`, `caption`, `<figcaption>` content — get rewritten too;
      attribute values that are data, not prose — `href` URLs, `src` paths, `class`, `id` — must
      be left exactly as in the source, byte-for-byte.
    - **Merging supporting material**: material from a supporting article goes into **new `<p>`
      elements inserted between the anchor's paragraphs**, placed where the subject matter fits —
      a supporting detail about the police statement goes next to the anchor's police paragraph;
      a later development goes after the anchor's last related paragraph. Never rewrite an anchor
      paragraph into a different paragraph's subject to make room, and never insert a new
      paragraph in the middle of a quote or between a heading and the section it introduces.
      Keep the added volume proportionate: the merged body should read as the anchor's article
      enriched — roughly the anchor's length plus the genuinely new material — not as a summary
      of everything submitted.
    - **`<img>` tags and their captions**: image markup varies by source — sometimes
      `<figure>/<figcaption>`, sometimes a `<table>`/`<div>` wrapper around `<img>` carrying
      `alt`, `title`, and `caption` attributes plus a trailing `<span>` with the visible caption.
      Whatever the wrapper, keep its tags, position, and `src` unchanged; rewrite the reader-facing
      text in `alt`/`title`/`caption` attributes and any `<figcaption>`/`<span>` caption text —
      but leave a trailing photo-credit fragment (e.g. `(Name/detikcom)`) exactly as in the source,
      since it's a credit line, not prose. Do not move an image to a different point in the
      markup, drop it, or add a new one. **Use only images that appear in the anchor's HTML —
      never carry an `<img>` over from a supporting article**, since its caption and credit belong
      to a different piece. (Note: these inline `<img>` tags inside `content` are separate from the
      standalone `image_cover_*` metadata fields, which are translation-only — see Track B above.)
    - **Headings** (`<h2>`, `<h3>`, etc., or bold-as-heading — see below): keep the anchor's tags,
      number, order, and nesting — do not add, remove, merge, or split sections. Rewrite the
      heading text itself with the same natural, idiomatic, AP-style treatment used for `title`,
      glossing Indonesia-specific terms on first mention if the heading contains one.
    - **Bold used as a heading, not a real `<h2>`/`<h3>` tag**: some sources mark section titles
      by wrapping the *entire* content of a `<p>` in `<strong>`/`<b>` instead of a semantic
      heading tag — e.g. `<p><strong>Skema Serahkan Aset KCIC</strong></p>` with no other text in
      that paragraph. Treat this the same as a real heading: keep it as that same bold-wrapped
      `<p>`, in the same position, and rewrite the text with heading-style treatment. Do not
      convert it to `<h2>`/`<h3>` (the anchor's own tag choice stays), and do not touch a
      `<strong>`/`<b>` run that sits *inside* a paragraph alongside other text — that's inline
      emphasis, not a heading, and should just be rewritten in place like any other text.
    - If the anchor's `content` has no heading tags (real or bold-as-heading), do not invent any —
      keep it as `<p>` paragraphs, not even to separate merged-in material.

    ## Non-article blocks to drop from `content`
    Source HTML often includes navigational or promotional inserts that are not part of the
    article's own reporting — most commonly a related-article widget (e.g. a block with
    `class="noncontent"` wrapping a "Baca juga:" / "Read also:" link to a separate, unrelated
    article) and embedded video promos (a "Tonton juga video ..." lead-in followed by a
    `class="noncontent"` block containing a video-embed link). These are template furniture from
    the source CMS, not facts belonging to this story:

    - **Drop these blocks entirely** — do not translate/rewrite the "Baca juga"/"Read also" or
      "Tonton juga video"/"Watch also" label, the linked headline, or the video-embed markup, and
      do not carry the block's tags into your output. Any leftover empty `<p></p>` immediately
      around a removed block can be dropped too.
    - **Only drop blocks that clearly match this pattern** (a `class="noncontent"` wrapper, a
      "Baca juga"/"Tonton juga video" style label, or a link to a separate unrelated article/video).
      Never drop a `<p>`, quote, or fact that is part of the actual reporting, even if it sits
      near one of these blocks.

    ## Input/output encoding
    A source `content` may arrive as raw HTML tags (e.g. `<p>...</p>`) or as HTML-entity-escaped
    text (e.g. `&lt;p&gt;...&lt;/p&gt;`). Detect which form the **anchor** uses and return
    `content` in that same form — do not switch a raw-tag anchor to escaped output, or vice
    versa, and normalize any merged-in supporting material to the anchor's form.

    ## Output structure
    Produce a single JSON object with exactly these fields: `title`, `summary`, `content`,
    `tags`, `keywordauto`, `categoryauto`, `image_cover_image_text`, `image_cover_alt_image`.

    **Track A fields (full rewrite):**

    - **`title`** — concise, active-voice, AP-style news headline (not clickbait, not a literal
      translation of the Indonesian headline), built from the **anchor's** headline and angle.
      Plain string, no trailing punctuation. **Maximum 65 characters, including spaces.** If the
      natural rewrite runs longer, you may restructure it — reorder clauses, cut filler words,
      use shorter synonyms, drop a secondary clause — rather than truncating mid-word or
      mid-thought; the title must still read as a complete, natural headline within the limit.
      **Preserve the anchor headline's own framing, not just its facts.** A word like "pamer"
      (flaunts/shows off) or a "faktanya ternyata" (turns out) twist isn't decorative — it's the
      angle the story is actually told from. If space is tight, cut a secondary clause or location
      detail before cutting the word that carries the story's tone or twist; a title that keeps
      every fact but loses the framing ("Alumnus Moves to the US" instead of "Alumnus Flaunts
      Move to the US") is a different, flatter story than the source told. **Never import a name,
      city, or fact from body text, a dateline, tags, keywords, or a supporting article into the
      title** — no location in the anchor headline means no location in the title.
    - **`summary`** — 1–2 sentence standalone summary of the anchor story's core
      who/what/when/where/why. Must make sense on its own without reading `content` (this is what
      will show in article previews/listings). Do not just copy the first sentence of `content`
      verbatim — write it as a distinct, compressed summary. **Maximum 130 characters, including
      spaces.** If the who/what/when/where/why doesn't fit in 130 characters, prioritize
      who/what/when over where/why and compress ruthlessly — the result must still be a
      grammatically complete sentence, not a fragment cut off mid-word.

    **`content` (structure-preserved rewrite):**

    - **`content`** — the full merged article body as HTML, with the anchor's own tag structure,
      order, headings, and image placement preserved exactly and supporting facts inserted as new
      paragraphs (see the HTML section above), the prose inside each element rewritten — natural,
      idiomatic, EEAT-quality English — rather than translated word-for-word. Do NOT include the
      headline inside `content` — that belongs only in `title`. There is no forced dateline and no
      paragraph-length target: each rewritten paragraph corresponds one-to-one to its anchor
      paragraph, and each inserted paragraph carries genuinely new supporting material.

    **Track B fields (translation only, no rewrite treatment — all taken from the ANCHOR):**

    - **`tags`** — the anchor's tags translated into English, same order; lowercase keyword/topic
      tags derived only from what's actually present in the source — do not invent, drop, merge,
      or add generic tags. **5–10 items are required**: only if the anchor supplies fewer than 5,
      top up — first from supporting-article tags that genuinely describe the anchor's story, then,
      if still short, from terms the anchor's own text clearly supports — appended after the
      anchor's own.
    - **`keywordauto`** — the anchor's auto-keywords translated into English, one-to-one, same
      order, **5–10 items**, topped up the same way only if the anchor supplies fewer than 5.
    - **`categoryauto`** — the anchor's category, translated. Never a supporting article's.
    - **`image_cover_image_text`** — the anchor's cover image caption/credit line, translated into
      natural English at the same length as the source (e.g. "Foto: Ayu" → "Photo: Ayu").
    - **`image_cover_alt_image`** — the anchor's cover image short alt-text, translated, same
      brevity as the source.

    Target length for `content`: proportional to the anchor's own length plus the genuinely new
    supporting material — do not significantly expand or compress the amount of substantive
    information conveyed.

    Example shape (values illustrative only):
    ```json
    {
      "title": "...",
      "summary": "...",
      "content": "<p>...</p><h2>Heading One</h2><p>...</p><table class=\\"pic_artikel_sisip_table\\">...<img src=\\"...\\" alt=\\"...\\" title=\\"...\\" caption=\\"...\\"/>...</table><p>...</p>",
      "tags": ["...", "...", "..."],
      "keywordauto": ["...", "...", "..."],
      "categoryauto": "...",
      "image_cover_image_text": "Photo: Ayu",
      "image_cover_alt_image": "..."
    }
    ```

    ## Before finalizing, check:
    - Does the article tell the ANCHOR's story with the anchor's angle, with no supporting article
      having taken over the lede, the title, or the summary?
    - Does every factual claim trace back to something in one of the source articles?
    - Would a reader with zero context on Indonesia understand every institution/term used, glossed
      once?
    - Does `content` read as natural, well-written English while preserving the anchor's exact
      paragraph order, heading structure, and image placement, with supporting material only in
      inserted paragraphs and no supporting-article image carried over?
    - Does any fact appear twice, was any source conflict silently resolved, or did any quote change
      speaker?
    - Are all figures, names, and dates consistent with the source that reported them?
    - Is every substantive claim attributed exactly as its source attributes it
      (authoritativeness)?
    - Is the certainty level of every claim matched to its source, with nothing upgraded
      (trustworthiness)?
    - Are official titles, terms, and institutional details precise rather than vague
      (expertise)?
    - Does `content` retain concrete, specific detail rather than reading as a flattened
      generic summary (experience)?
    - Is `title` at most 65 characters, and `summary` at most 130 characters — each still a
      complete, natural sentence or headline rather than a truncated fragment?
    - Does `title` keep the anchor headline's own framing/tone (e.g., "flaunts," a "turns out"
      twist) rather than flattening it into a neutral statement of facts?
    - Does any place mentioned in `title` or `summary` stay a generic description if that's what
      the source gave, instead of reading like the proper name of a specific venue no source
      actually named?
    - Are `tags`, `keywordauto`, `categoryauto`, and both `image_cover_*` fields faithful
      translations of the anchor's values — not rewritten, restructured, tightened, or expanded —
      while still accurate given the article's context, with 5–10 items in each list?
    - Have all "Baca juga"/"Tonton juga video" style non-content blocks been dropped from
      `content`, with no actual reporting removed along with them?
    - Does the output `content` use the same raw-vs-escaped HTML encoding as the anchor?
    """

    anchor, *supporting = news_items
    blocks = [_format_news(anchor, "MAIN ARTICLE (ANCHOR)")]
    for i, item in enumerate(supporting, start=2):
        blocks.append(_format_news(item, f"SUPPORTING ARTICLE {i}"))

    prompt_input = (
        f"""
    Below are {len(news_items)} Indonesian source article(s).
    The first one is the main article (anchor) — it fixes the story, angle, structure and images.
    Any others are supporting sources; use them only to enrich the anchor's story. If no supporting
    article follows the anchor, the anchor is the only source: rewrite it alone, nothing merged in.
    """
        + "\n".join(blocks)
    )

    print(prompt_input)

    output_format = ArticleSchema
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_input}
    ]
    response = client.responses.parse(
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        input=messages,
        text_format=output_format,
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )

    # Extract structured JSON from function_call
    return _parse_response(response)


def _format_category_labels():
    return "\n".join(f"    - {leaf} — {desc}" for leaf, desc in _category_labels())


def generate_keywords_category(content):
    """Derive `keywordauto` (5-10 terms) and `categoryauto` (one) from raw content text.

    `content` is a plain string — article body, HTML or plain text, in any
    language. `categoryauto` is always one of the labels in
    input/categoryauto_labelling.csv, enforced both in the prompt and by the
    response schema. Returns (result_dict, token_usage).
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("content must not be empty")

    client = _client()

    system_instruction = """
    # System Prompt: Keywords + Category from a piece of content

    ## Role
    News desk metadata editor. Read one piece of content and output the search keywords and the
    single section category it would be filed under. Nothing else — no summary, no rewrite.

    ## Language
    Write `keywordauto` in the **same language as the content**. If the content is Indonesian,
    output Indonesian terms; if English, English terms. Never mix the two. `categoryauto` is the
    exception: it is always one of the fixed labels listed below, copied exactly as written there,
    whatever the content's language.

    ## `keywordauto` — 5 to 10 terms, calibrated specificity
    These are the terms a reader would actually type into search to find this exact story. Aim for
    the middle band between too general and too niche:
    - **Too general (reject)** — a term that would match thousands of unrelated stories and says
    nothing about this one: "news", "Indonesia", "government", "viral", "today", "information",
    a bare year.
    - **Too niche (reject)** — a term so specific to one sentence that nobody would search it: a
    full quote, a long clause, a document/case number, an exact street address, a precise figure
    ("Rp 1.237.450"), a minor person mentioned once in passing.
    - **The right band (keep)** — the story's central actors, institutions, places, events, objects
    and issues, as a searcher would name them: a named person at the centre of the story, the
    institution or company involved, the city/region, the event or policy, the concrete topic.
    Two to four words each is typical; single common nouns are usually too general.

    Rules:
    1. **Only from the content** — every keyword must be grounded in what the text actually says.
    Never invent a name, place, or topic the content does not carry.
    2. **No duplication** — no two keywords that are the same term, a plural/singular pair, or one
    fully contained in another ("Prabowo" and "Prabowo Subianto" — keep the more searchable one).
    3. **Order by centrality** — most central to the story first.
    4. **No hashtags, no punctuation-as-syntax, no ALL CAPS** (beyond genuine acronyms), no quotes
    around terms. Keep each term's natural capitalization: proper nouns capitalized, common nouns
    lowercase.
    5. **Acronyms** — use the form the content uses; if it gives both ("Kereta Cepat Indonesia
    China (KCIC)"), the widely-searched short form is the better keyword.

    ## `categoryauto` — exactly one label from the taxonomy
    Choose the single label whose description best matches what this content is actually about.
    - **Copy the label verbatim** — exact spelling, spacing and capitalization as listed. Never
    translate it, never reword it, never invent one that isn't on the list, never return two.
    - **Decide by description, not by label wording** — a label's description states what it
    includes, and several descriptions carry explicit exclusions ("excl. ...", "selain ...").
    Honour those: a story the description excludes does not belong to that label.
    - **The story's main subject decides**, not a term mentioned in passing. If the content touches
    several labels, pick the one the bulk of the reporting serves.
    - When nothing fits well, use the matching **"... Lainnya"** (other) label of the closest
    section rather than forcing a specific one.

    ### Allowed labels (`Label` — what it covers)
{category_list}

    ## Before finalizing, check:
    - Between 5 and 10 keywords, all in the content's language, none of them a term that would
    match any random news story, none of them a phrase nobody would search.
    - Every keyword traces to something the content actually states.
    - No keyword repeats or contains another.
    - `categoryauto` is exactly one label copied verbatim from the allowed list, not a translation,
    not a list, not a keyword restated, and not excluded by that label's own description.
    """.replace("{category_list}", _format_category_labels())

    prompt_input = f"""
    Below is the content. Output only its keywords and category.

    Content: {content}"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_input},
    ]
    response = client.responses.parse(
        model=OPENROUTER_MODEL,
        temperature=OPENROUTER_TEMPERATURE,
        input=messages,
        text_format=_keyword_category_schema(),
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )

    return _parse_response(response)
