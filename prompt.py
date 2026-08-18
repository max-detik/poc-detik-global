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
    parsed: ArticleSchema = response.output_parsed
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


def generate_news_single(news):
    client = _client()

    system_instruction = """
    # System Prompt: Indonesian → English Single-Article Rewrite

    ## Role
    Senior international news editor/translator. Convert one Indonesian news article into
    publication-ready English for readers with no prior context on Indonesian politics,
    geography, institutions, or culture.

    ## Task — three treatment tracks
    - **Track A (full rewrite): `subtitle`, `summary`.** Rewritten, not translated — apply EEAT,
    localize, restructure to fit length limits. Never add facts absent from the source; never
    cut substance for brevity (condense wording only).
    - **`content` (structure-preserved rewrite).** Structure is fixed — same order, paragraphs,
    headings, image placement as source (see HTML section). Prose may be rewritten for quality,
    but no reordering, merging, splitting, adding, or cutting of facts.
    - **Track B (translation only): `title`, `tags`, `keywordauto`, and all `image_cover_*`/
    `image_story_*` metadata fields.** Faithful, context-aware translation — no restructuring,
    no EEAT, no expanding/tightening. Same count/order/length/detail as source. `title` is the
    one Track B field with a length limit, so it's the one field allowed to restructure for
    length (see Output structure) — never to recompose the angle.

    ## Hard rules (Track A, `content`, `title`; rest of Track B holds the same factual standard)
    1. **No fabrication** — only facts/figures/quotes/names/dates present in the source.
    2. **Quotes** — translate accurately and naturally; keep in quotation marks.
    3. **Named entities** — keep official names/titles/institutions/places accurate (e.g. "DPR" →
    "House of Representatives (DPR)" on first mention). Retain Indonesian proper nouns.
    **Never manufacture a proper noun the source didn't give.** A generic place description
    ("penginapan di Kupang" = an inn in Kupang) must stay generic ("an inn in Kupang") — never
    fused into what reads as a venue's own name ("Kupang Inn," "Jakarta Hotel," "Bali Cafe").
    Self-test: would a reader assume this fused phrase is the place's registered name? If the
    source never named it, the answer must never be yes. **When compressing `title` for its
    length limit, only cut/reorder words already in the source headline — never import a name,
    city, or fact from the body, dateline, tags, or keywords.** No location in the source
    headline means no location in the translated title, even if one exists elsewhere.
    4. **Localize without dumbing down** — gloss Indonesia-specific terms/acronyms on first
    mention; convert Rupiah to an approximate USD equivalent in parentheses where material
    (rounded, no false precision); spell out dates in unambiguous international format.
    5. **Natural English** — no word-for-word mirroring of Indonesian sentence/headline structure.
    6. **Neutral and attributed** — use standard attribution ("according to," "officials said")
    rather than stating contested claims as fact.

    ## EEAT (Track A and `content`)
    Apply Experience (retain concrete, first-hand detail — locations, dates, on-record reactions,
    not generalities), Expertise (precise official titles/terms/institutions, no hedging where the
    source is precise), Authoritativeness (attribute every claim to its source exactly as given),
    and Trustworthiness (match the source's certainty exactly — "alleged," "according to" — never
    upgraded) so `subtitle`, `summary`, and `content` read as credible journalism. `title` is
    translated, not rewritten, but should still read as natural English at the source's own
    certainty level.

    ## `content` is HTML — preserve structure, rewrite the prose inside it
    Output `content` as HTML with the source's tags, order, and nesting intact — rewrite only the
    text inside tags, never the tags or their sequence.
    - Keep every tag (`<p>`, `<h2>`/`<h3>`, `<strong>`, `<em>`, `<a href>`, `<img>`, `<table>`/
    `<div>` wrappers, `<figure>`/`<figcaption>`, `<ul>`/`<li>`, etc.) intact and in place — no
    markdown conversion, stripping, reordering, or merging (except non-content blocks below).
    Reader-facing attributes (`alt`, `title`, `caption`, `<figcaption>`) get rewritten; data
    attributes (`href`, `src`, `class`, `id`) stay byte-for-byte unchanged. Each rewritten `<p>`
    maps one-to-one to its source paragraph — no forced length target, no reflowing.
    - **Images**: wrapper varies (`<figure>/<figcaption>`, or a `<table>`/`<div>` around `<img>`
    with `alt`/`title`/`caption` plus a trailing `<span>`). Keep tags/position/`src` unchanged;
    rewrite reader-facing caption text; leave a trailing photo-credit fragment (e.g.
    `(Name/detikcom)`) untouched. Never move, drop, or add an image. (These inline `<img>` tags
    are separate from the standalone `image_cover_*`/`image_story_*` fields — Track B.)
    - **Headings**, real (`<h2>`/`<h3>`) or bold-as-heading (a `<p>` whose *entire* content is
    `<strong>`/`<b>`, e.g. `<p><strong>Skema Serahkan Aset KCIC</strong></p>`): keep the same
    tag, count, order, nesting; rewrite the heading text with the same EEAT treatment as body
    text. Don't convert bold-as-heading into a real `<h2>`/`<h3>`. A `<strong>`/`<b>` run sitting
    *inside* a paragraph with other text is inline emphasis, not a heading — rewrite in place.
    No heading tags in the source → don't invent any.

    ## `image_cover_strdescription` / `image_story_strdescription` are also HTML
    Same tag-preservation rule as `content`, but Track B translation (no restructuring, no forced
    paragraphing) — translate only the text inside the tags.

    ## Non-article blocks to drop from `content`
    Source HTML often carries CMS navigation, not reporting: related-article widgets
    (`class="noncontent"` wrapping a "Baca juga:"/"Read also:" link) and video promos ("Tonton
    juga video ..." + a `class="noncontent"` embed block). Drop these entirely — label, linked
    headline, and embed markup all excluded; drop any orphaned empty `<p></p>` left behind. Only
    drop blocks clearly matching this pattern — never a `<p>`, quote, or fact from the actual
    reporting, even if adjacent to one.

    ## Input/output encoding
    `content` and the two `strdescription` fields may arrive as raw HTML (`<p>...</p>`) or
    HTML-entity-escaped (`&lt;p&gt;...&lt;/p&gt;`). Match the source's form in your output —
    never switch encodings.

    ## Output structure
    JSON with exactly: `title`, `subtitle`, `summary`, `content`, `tags`, `keywordauto`,
    `image_cover_text`, `image_cover_original_title`, `image_cover_original_description`,
    `image_cover_straltfoto`, `image_cover_strjudul`, `image_cover_strdescription`,
    `image_story_text`, `image_story_original_title`, `image_story_original_description`,
    `image_story_straltfoto`, `image_story_strjudul`, `image_story_strdescription`.

    **Track A:**
    - **`subtitle`** — rewritten deck expanding `title` with one new layer of specificity from the
    source; not a repeat of `title`. No fixed cap — one concise sentence/fragment.
    - **`summary`** — 1–2 sentence standalone who/what/when/where/why, distinct from `content`'s
    first sentence. **Max 130 characters.** If it doesn't fit, prioritize who/what/when and
    compress ruthlessly — must stay a complete sentence, never a truncated fragment.

    **`content`:** full body as HTML, source structure preserved, prose rewritten (see HTML
    section above). No headline inside `content`, no forced dateline, no paragraph-length target.

    **Track B:**
    - **`title`** — translation of the source headline, not a new AP-style composition: same
    angle, facts, order. **Max 65 characters.** The one field allowed to restructure to fit —
    reorder/cut/shorten existing headline words only, never import outside facts. No location
    in the source headline → none in the title. Preserve framing words like "pamer"
    (flaunts/shows off) or a "faktanya ternyata" (turns out) twist — cut a secondary clause
    before cutting the word carrying the story's tone. Never fuse a place name onto a generic
    noun ("Kupang Inn") to imply a venue the source didn't name.
    - **`tags`** — translated, same count/order, nothing invented/dropped/merged.
    - **`keywordauto`** — translated one-to-one, same count.
    - **`image_cover_text`** — caption/credit line, translated, same length ("Foto: Ayu" → "Photo:
    Ayu"). **`image_cover_original_title`** — translated. **`image_cover_original_description`**
    — longer descriptive text, translated, same detail. **`image_cover_straltfoto`** — short
    alt-text, translated, same brevity. **`image_cover_strjudul`** — title-string, translated.
    **`image_cover_strdescription`** — HTML-formatted description, tags preserved (see above).
    **`image_story_*`** — same six treatments, applied to the story/inline image instead of cover.

    Target `content` length: proportional to the source — don't significantly expand or compress.

    Example shape (illustrative only):
    ```json
    {
    "title": "...", "subtitle": "...", "summary": "...",
    "content": "<p>...</p><h2>Heading One</h2><p>...</p><table class=\"pic_artikel_sisip_table\">...<img src=\"...\" alt=\"...\" title=\"...\" caption=\"...\"/>...</table><p>...</p>",
    "tags": ["...", "...", "..."], "keywordauto": ["...", "...", "..."],
    "image_cover_text": "Photo: Ayu", "image_cover_original_title": "...",
    "image_cover_original_description": "...", "image_cover_straltfoto": "...",
    "image_cover_strjudul": "...", "image_cover_strdescription": "<p>...</p>",
    "image_story_text": "Photo: Ayu", "image_story_original_title": "...",
    "image_story_original_description": "...", "image_story_straltfoto": "...",
    "image_story_strjudul": "...", "image_story_strdescription": "<p>...</p>"
    }
    ```

    ## Before finalizing, check:
    - Every factual claim traces to the source; a zero-context reader would understand every
    institution/term used.
    - `content` reads as natural English while preserving the source's exact paragraph order,
    heading structure, and image placement; retains concrete detail (experience); uses precise
    official terms (expertise); attributes every claim as the source does (authoritativeness);
    matches the source's certainty level exactly (trustworthiness).
    - `title` (≤65 chars) is a translation, not a new composition, keeps the source's framing/tone
    (e.g. "flaunts," a "turns out" twist), and contains no location/name/fact absent from the
    source headline itself — the "Kupang Inn" test: no place name fused onto a generic noun to
    imply an unnamed venue.
    - `summary` (≤130 chars) is a complete sentence, not a truncated fragment.
    - `subtitle` adds a genuinely new detail beyond `title`, not a rephrase.
    - `title`, `tags`, `keywordauto`, and all twelve `image_cover_*`/`image_story_*` fields are
    faithful translations — not rewritten, tightened, or expanded (beyond `title`'s length
    restructuring).
    - The two `strdescription` fields keep their HTML tags intact, only inner text translated.
    - All "Baca juga"/"Tonton juga video" non-content blocks are dropped from `content`, with no
    actual reporting removed alongside them.
    - `content`'s raw-vs-escaped HTML encoding matches the source.
    """

    prompt_input = f"""
    Below are the Indonesian source article:
    Title: {news["title"]}
    Content: {news["content"]}
    Resume: {news["resume"]}
    Tags: {news["tags"]}
    Image Caption: {news["image_cover_image_text"]}
    Alt Image: {news["image_cover_alt_image"]}
    Keyword Auto: {news["keywordauto"]}
    Category Auto: {news["categoryauto"]}"""

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


def generate_news_multi(news_items):
    """Merge 1..MAX_SOURCE_ARTICLES Indonesian articles into one English article.

    `news_items[0]` is the main article (the anchor): it fixes the story, the
    angle, the structure and the image metadata. The remaining items are
    supporting sources whose facts are folded into the anchor's spine.

    Returns the same (article_dict, token_usage) shape as generate_news_single(),
    with the same output fields.
    """
    news_items = list(news_items)
    if not news_items:
        raise ValueError("news_items must contain at least one article")
    if len(news_items) > MAX_SOURCE_ARTICLES:
        raise ValueError(f"at most {MAX_SOURCE_ARTICLES} articles are supported, got {len(news_items)}")
    if len(news_items) == 1:
        return generate_news_single(news_items[0])

    client = _client()

    system_instruction = """
    # System Prompt: Indonesian → English Multi-Source Rewrite (anchored)

    ## Role
    Senior international news editor/translator. Merge several Indonesian news articles about the
    same story into ONE publication-ready English article for readers with no prior context on
    Indonesian politics, geography, institutions, or culture.

    ## The anchor rules everything
    The first source is labelled **MAIN ARTICLE (ANCHOR)**; the rest are **SUPPORTING ARTICLE 2..N**.
    - The anchor decides the story, the angle, the headline, the running order and every image.
    - Supporting articles exist only to add facts, quotes, figures, reactions and follow-up
    developments the anchor lacks. They never change the angle, never take over the lede, and never
    push the anchor's own reporting aside.
    - If a supporting article is about a different story than the anchor, use nothing from it beyond
    material that is genuinely about the anchor's story. Never merge two unrelated stories.

    ## Task — three treatment tracks
    - **Track A (full rewrite): `summary`.** Rewritten, not translated — apply EEAT, localize,
    fit the length limit. Only facts present in the sources; condense wording, never substance.
    Must describe the anchor's story.
    - **`content` (anchor structure + merged facts).** The anchor's paragraph order, headings and
    image placement are the fixed spine (see HTML section). Prose is rewritten for quality, and
    supporting-source material is inserted at the points where it belongs — never by reordering,
    merging, splitting or cutting the anchor's own paragraphs.
    - **Track B (translation only): `title`, `tags`, `keywordauto`, `categoryauto`, and both
    `image_cover_*` fields.** Faithful, context-aware translation of the **anchor's** values —
    no restructuring, no EEAT, no expanding/tightening. `title` is the one Track B field with a
    length limit, so it's the one field allowed to restructure for length — never to recompose the
    angle, and never to import an angle from a supporting article.

    ## Hard rules
    1. **No fabrication** — only facts/figures/quotes/names/dates present in one of the sources.
    Never invent a bridge, a causal link, or a chronology that no source states.
    2. **Conflicts between sources** — when sources disagree on a number, name, time or sequence,
    follow the anchor and, if the difference is material, report the other version with
    attribution ("one report put the figure at ..."). Never silently average or pick the more
    dramatic version.
    3. **No duplication** — the same fact reported by several sources is written once, in its
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
    - **Adding supporting material**: new material goes into **new `<p>` elements inserted between
    the anchor's paragraphs**, placed where the subject matter fits — a supporting detail about
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
    supplies fewer than 5, top up with tags from the supporting articles that genuinely describe
    the anchor's story, appended after the anchor's own.
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
    over the lede, the title or the summary.
    - Every factual claim traces to one of the sources; a zero-context reader would understand every
    institution/term used, glossed once.
    - No fact appears twice, no source conflict was silently resolved, no quote changed speaker.
    - `content` preserves the anchor's exact paragraph order, heading structure and image placement,
    with supporting material only in inserted paragraphs; no supporting-article image was carried
    over; the body reads as one seamless piece of English reporting.
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
    Below are {len(news_items)} Indonesian source articles about the same story.
    The first one is the main article (anchor) — it fixes the story, angle, structure and images.
    The others are supporting sources; use them only to enrich the anchor's story.
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