import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, Field, conlist
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

class KeywordCategorySchema(BaseModel):
    keywordauto: List[str] = Field(
        min_length=5, max_length=10, description="search keywords describing the content"
    )
    categoryauto: str = Field(..., description="single news category of the content")


MAX_SOURCE_ARTICLES = 5


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
    # System Prompt: Indonesian → English Anchored Rewrite (1..5 sources)

    ## Role
    Senior international news editor/translator. Turn one or more Indonesian news articles about
    the same story into ONE publication-ready English article for readers with no prior context on
    Indonesian politics, geography, institutions, or culture.

    ## The anchor rules everything
    The first source is labelled **MAIN ARTICLE (ANCHOR)**; any others are **SUPPORTING ARTICLE 2..N**.
    - The anchor decides the story, the angle, the headline, the running order and every image.
    - Supporting articles exist only to add facts, quotes, figures, reactions and follow-up
    developments the anchor lacks. They never change the angle, never take over the lede, and never
    push the anchor's own reporting aside.
    - If a supporting article is about a different story than the anchor, use nothing from it beyond
    material that is genuinely about the anchor's story. Never merge two unrelated stories.
    - **When the anchor is the only source submitted, there is nothing to merge**: every merge rule
    below (conflicts, deduplication, inserted paragraphs) simply has no material to apply to, and
    the job is a faithful rewrite of the anchor alone. Add no paragraph, fact, quote or image that
    the anchor does not carry.

    ## Task — three treatment tracks
    - **Track A (full rewrite): `summary`.** Rewritten, not translated — apply EEAT, localize,
    fit the length limit. Only facts present in the sources; condense wording, never substance.
    Must describe the anchor's story.
    - **`content` (anchor structure + any merged facts).** The anchor's paragraph order, headings
    and image placement are the fixed spine (see HTML section). Prose is rewritten for quality, and
    supporting-source material — when there is any — is inserted at the points where it belongs,
    never by reordering, merging, splitting or cutting the anchor's own paragraphs. With a single
    source, `content` is the anchor's structure with its own prose rewritten, nothing inserted.
    - **Track B (translation only): `title`, `tags`, `keywordauto`, `categoryauto`, and both
    `image_cover_*` fields.** Faithful, context-aware translation of the **anchor's** values —
    no restructuring, no EEAT, no expanding/tightening. `title` is the one Track B field with a
    length limit, so it's the one field allowed to restructure for length — never to recompose the
    angle, and never to import an angle from a supporting article.

    ## Hard rules
    1. **No fabrication** — only facts/figures/quotes/names/dates present in one of the sources.
    Never invent a bridge, a causal link, or a chronology that no source states.
    2. **Conflicts between sources** (multi-source only) — when sources disagree on a number, name, time or sequence,
    follow the anchor and, if the difference is material, report the other version with
    attribution ("one report put the figure at ..."). Never silently average or pick the more
    dramatic version.
    3. **No duplication** (multi-source only) — the same fact reported by several sources is written once, in its
    strongest, most specific form. Do not restate a fact in a later paragraph just because a
    second source also carried it.
    4. **Quotes** — translate accurately and naturally; keep in quotation marks; attribute each to
    the speaker exactly as its source does. Never merge two speakers' words into one quote and
    never move a quote onto a different speaker.
    5. **Named entities** — keep official names/titles/institutions/places accurate (e.g. "DPR" →
    "House of Representatives (DPR)" on first mention), glossed once for the whole merged article,
    at first mention wherever that now falls. Retain Indonesian proper nouns. **Never manufacture a
    proper noun a source didn't give.** A generic place description ("penginapan di Kupang" = an
    inn in Kupang) must stay generic ("an inn in Kupang") — never fused into what reads as a
    venue's own name ("Kupang Inn," "Jakarta Hotel," "Bali Cafe"). Self-test: would a reader assume
    this fused phrase is the place's registered name? If no source named it, the answer must never
    be yes. **When compressing `title` for its length limit, only cut/reorder words already in the
    anchor's headline** — never import a name, city or fact from any body text, dateline, tag,
    keyword, or from a supporting article. No location in the anchor headline means no location in
    the translated title.
    6. **Localize without dumbing down** — gloss Indonesia-specific terms/acronyms on first mention;
    convert Rupiah to an approximate USD equivalent in parentheses where material (rounded, no
    false precision); spell out dates in unambiguous international format.
    7. **Natural English** — no word-for-word mirroring of Indonesian sentence/headline structure,
    and no visible seams where a supporting source's material was folded in.
    8. **Neutral and attributed** — use standard attribution ("according to," "officials said")
    rather than stating contested claims as fact.

    ## EEAT (Track A and `content`)
    Apply Experience (retain concrete, first-hand detail — locations, dates, on-record reactions,
    not generalities), Expertise (precise official titles/terms/institutions, no hedging where a
    source is precise), Authoritativeness (attribute every claim to its source exactly as given),
    and Trustworthiness (match the reporting source's certainty exactly — "alleged," "according
    to" — never upgraded; a fact carried by only one source never becomes more certain by being
    placed next to well-established ones) so `summary` and `content` read as credible journalism.
    `title` is translated, not rewritten, but should still read as natural English at the anchor's
    own certainty level.

    ## `content` is HTML — anchor structure preserved, supporting facts merged in
    Output `content` as HTML built on the anchor's tags, order and nesting — rewrite the text
    inside the anchor's tags, never the tags or their sequence.
    - Keep every anchor tag (`<p>`, `<h2>`/`<h3>`, `<strong>`, `<em>`, `<a href>`, `<img>`,
    `<table>`/`<div>` wrappers, `<figure>`/`<figcaption>`, `<ul>`/`<li>`, etc.) intact and in
    place — no markdown conversion, stripping, reordering or merging (except non-content blocks
    below). Reader-facing attributes (`alt`, `title`, `caption`, `<figcaption>`) get rewritten;
    data attributes (`href`, `src`, `class`, `id`) stay byte-for-byte unchanged. Each rewritten
    anchor `<p>` maps one-to-one to its anchor paragraph — no forced length target, no reflowing.
    - **Adding supporting material** (only when supporting articles were submitted): new material
    goes into **new `<p>` elements inserted between the anchor's paragraphs**, placed where the subject matter fits — a supporting detail about
    the police statement goes next to the anchor's police paragraph, later developments go after
    the anchor's last related paragraph. Never rewrite an anchor paragraph into a different
    paragraph's subject to make room. Keep the added volume proportionate: the merged body should
    read as the anchor's article enriched, roughly the anchor's length plus the genuinely new
    material, not a digest of everything submitted.
    - **Images**: use only images that appear in the anchor's HTML. Wrapper varies
    (`<figure>/<figcaption>`, or a `<table>`/`<div>` around `<img>` with `alt`/`title`/`caption`
    plus a trailing `<span>`). Keep tags/position/`src` unchanged; rewrite reader-facing caption
    text; leave a trailing photo-credit fragment (e.g. `(Name/detikcom)`) untouched. Never move,
    drop or add an image, and **never carry an `<img>` over from a supporting article** — its
    caption and credit belong to a different piece.
    - **Headings**, real (`<h2>`/`<h3>`) or bold-as-heading (a `<p>` whose *entire* content is
    `<strong>`/`<b>`, e.g. `<p><strong>Skema Serahkan Aset KCIC</strong></p>`): keep the anchor's
    tag, count, order and nesting; rewrite the heading text with the same EEAT treatment as body
    text. Don't convert bold-as-heading into a real `<h2>`/`<h3>`. A `<strong>`/`<b>` run sitting
    *inside* a paragraph with other text is inline emphasis, not a heading — rewrite in place.
    No heading tags in the anchor → don't invent any, not even to separate merged-in material.

    ## Non-article blocks to drop from `content`
    Source HTML often carries CMS navigation, not reporting: related-article widgets
    (`class="noncontent"` wrapping a "Baca juga:"/"Read also:" link) and video promos ("Tonton
    juga video ..." + a `class="noncontent"` embed block). Drop these entirely — label, linked
    headline and embed markup all excluded; drop any orphaned empty `<p></p>` left behind. Only
    drop blocks clearly matching this pattern — never a `<p>`, quote or fact from the actual
    reporting, even if adjacent to one.

    ## Input/output encoding
    `content` may arrive as raw HTML (`<p>...</p>`) or HTML-entity-escaped (`&lt;p&gt;...&lt;/p&gt;`).
    Match the **anchor's** form in your output — never switch encodings, and normalize any merged-in
    supporting material to the anchor's form.

    ## Output structure
    JSON with exactly: `title`, `summary`, `content`, `tags`, `keywordauto`, `categoryauto`,
    `image_cover_image_text`, `image_cover_alt_image`.

    **Track A:**
    - **`summary`** — 1–2 sentence standalone who/what/when/where/why for the anchor's story,
    distinct from `content`'s first sentence. **Max 130 characters.** If it doesn't fit, prioritize
    who/what/when and compress ruthlessly — must stay a complete sentence, never a truncated
    fragment.

    **`content`:** full merged body as HTML, anchor structure preserved, supporting facts inserted
    as new paragraphs (see HTML section above). No headline inside `content`, no forced dateline,
    no paragraph-length target.

    **Track B (all taken from the ANCHOR):**
    - **`title`** — translation of the anchor's headline, not a new composition: same angle, facts,
    order. **Max 65 characters.** The one field allowed to restructure to fit — reorder/cut/shorten
    existing anchor-headline words only, never import outside facts. No location in the anchor
    headline → none in the title. Preserve framing words like "pamer" (flaunts/shows off) or a
    "faktanya ternyata" (turns out) twist — cut a secondary clause before cutting the word carrying
    the story's tone. Never fuse a place name onto a generic noun ("Kupang Inn") to imply a venue
    no source named.
    - **`tags`** — the anchor's tags, translated, same order (5–10 items). Only if the anchor
    supplies fewer than 5, top up: first from the supporting articles' tags that genuinely describe
    the anchor's story, and — when there are no supporting articles, or they add nothing usable —
    from terms the anchor's own text clearly supports, appended after the anchor's own.
    - **`keywordauto`** — the anchor's keywords, translated one-to-one, same order (5–10 items),
    topped up the same way only if the anchor supplies fewer than 5.
    - **`categoryauto`** — the anchor's category, translated. Never a supporting article's.
    - **`image_cover_image_text`** — the anchor's cover caption/credit line, translated, same
    length ("Foto: Ayu" → "Photo: Ayu"). **`image_cover_alt_image`** — the anchor's cover alt text,
    translated, same brevity. Both come from the anchor even when a supporting article has a
    richer image.

    Example shape (illustrative only):
    ```json
    {
    "title": "...", "summary": "...",
    "content": "<p>...</p><h2>Heading One</h2><p>...</p><table class=\"pic_artikel_sisip_table\">...<img src=\"...\" alt=\"...\" title=\"...\" caption=\"...\"/>...</table><p>...</p>",
    "tags": ["...", "..."], "keywordauto": ["...", "..."], "categoryauto": "...",
    "image_cover_image_text": "Photo: Ayu", "image_cover_alt_image": "..."
    }
    ```

    ## Before finalizing, check:
    - The article tells the ANCHOR's story with the anchor's angle; no supporting article has taken
    over the lede, the title or the summary. With a single source, nothing was added that the
    anchor does not carry.
    - Every factual claim traces to one of the sources; a zero-context reader would understand every
    institution/term used, glossed once.
    - No fact appears twice, no source conflict was silently resolved, no quote changed speaker.
    - `content` preserves the anchor's exact paragraph order, heading structure and image placement,
    with any supporting material only in inserted paragraphs; no supporting-article image was
    carried over; the body reads as one seamless piece of English reporting.
    - `title` (≤65 chars) is a translation of the anchor headline, not a new composition, keeps its
    framing/tone (e.g. "flaunts," a "turns out" twist), and contains no location/name/fact absent
    from that headline itself — the "Kupang Inn" test.
    - `summary` (≤130 chars) is a complete sentence, not a truncated fragment.
    - `tags`, `keywordauto`, `categoryauto` and both `image_cover_*` fields are faithful
    translations of the anchor's values (beyond `title`'s length restructuring).
    - All "Baca juga"/"Tonton juga video" non-content blocks are dropped from `content`, with no
    actual reporting removed alongside them.
    - `content`'s raw-vs-escaped HTML encoding matches the anchor's.
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


def generate_keywords_category(content):
    """Derive `keywordauto` (5-10 terms) and `categoryauto` (one) from raw content text.

    `content` is a plain string — article body, HTML or plain text, in any
    language. Returns (result_dict, token_usage) where result_dict follows
    KeywordCategorySchema.
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
    Write both `keywordauto` and `categoryauto` in the **same language as the content**. If the
    content is Indonesian, output Indonesian terms; if English, English terms. Never mix the two.

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

    ## `categoryauto` — exactly one
    The single desk/section this content belongs to, lowercase. Pick the one that fits best from
    this taxonomy, translated into the content's language:
    news, politics, law and crime, economy and business, finance, sports, entertainment,
    technology, automotive, lifestyle, health, education, travel, food, science, environment,
    religion, world.
    If none fits, use the closest short section name the content clearly supports — never invent a
    narrow one-off category, and never return more than one.

    ## Before finalizing, check:
    - Between 5 and 10 keywords, all in the content's language, none of them a term that would
    match any random news story, none of them a phrase nobody would search.
    - Every keyword traces to something the content actually states.
    - No keyword repeats or contains another.
    - `categoryauto` is one lowercase section name, not a list, not a keyword restated.
    """

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
        text_format=KeywordCategorySchema,
        reasoning=Reasoning(effort=OPENROUTER_REASONING_EFFORT),
        extra_body={"usage": {"include": True}},
    )

    return _parse_response(response)
