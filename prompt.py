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

def generate_news_single(news):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

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