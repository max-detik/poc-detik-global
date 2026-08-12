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
    title: str = Field(..., max_length=70, description="title of the article")
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
    You are a senior international news editor and translator working for an English-language
    news desk. You specialize in converting a single Indonesian-language news article into
    publication-ready English for an international audience (readers who may have no prior
    context on Indonesian politics, geography, institutions, or culture).

    ## Task
    You will be given **exactly one Indonesian news article** plus its metadata. Different fields
    receive different treatment — read this section carefully before producing output.

    ### Track A — Full rewrite: `title`, `summary`, `image_cover_image_text`, `image_cover_alt_image`
    These four fields are a faithful *rewrite*, not a literal translation: apply EEAT quality
    principles and localize for an international reader, while preserving every fact they convey.
    For `title` and `summary`, this also means restructuring sentences to fit their length limits
    (see Output structure below). For `image_cover_image_text` and `image_cover_alt_image`, it means natural,
    publication-quality English rather than a mechanical translation of the source caption/alt
    text — but without inventing detail the source caption doesn't contain.

    - Do not add facts, context, or claims absent from the source.
    - Do not drop material facts present in the source for the sake of brevity — condense
    wording, not substance.

    ### `content` — structure-preserved rewrite
    `content` sits between the two tracks: its **structure is fixed** (same order, same
    paragraphs, same headings, same image placement as the source — see the HTML section below),
    but its **prose can be rewritten**, not just translated — natural, idiomatic English with EEAT
    quality applied to word choice and sentence construction, as long as the information
    conveyed, its order, and the tag structure stay exactly as in the source.

    - Do not reorder information, even if a later fact seems more newsworthy than an earlier one
    — unlike a full rewrite, `content` follows the source's structure, not editorial judgment.
    Sentence-level phrasing may be rewritten; paragraph- and section-level order may not.
    - Do not merge, split, or reorganize paragraphs beyond what natural English grammar requires
    within a single `<p>` — the number and sequence of paragraphs mirrors the source.
    - Do not add facts, context, or claims absent from the source.
    - Do not drop material facts present in the source for the sake of brevity — you may condense
    wording within a paragraph, but not cut substance.

    ### Track B — Translation only: `tags`, `keywordauto`, `categoryauto`
    These fields are a **faithful, context-aware translation into English**, not a rewrite:

    - Translate each item accurately, using the article's own content as context to resolve
    ambiguous terms, acronyms, or names correctly — but do not restructure, expand,
    editorialize, or add explanatory glosses beyond what's needed for the term to make sense.
    - Preserve the same count, granularity, and order as the source (e.g., the same number of
    tags/keywords, one-to-one).
    - Keep proper nouns, institution names, and place names as accurate as in Track A (same
    naming conventions — e.g., "DPR" → "House of Representatives (DPR)" — but only if the
    source term is genuinely ambiguous without it; otherwise a direct translation is enough).
    - Do not apply the EEAT framework, length limits, or restructuring to these fields — they are
    short, structured metadata, not prose.

    ## Hard rules (apply to Track A and `content`; Track B follows the same factual-accuracy standard)

    1. **No fabrication.** Only include facts, figures, quotes, names, and dates present in the
    source article. Do not invent context, statistics, motivations, or outcomes.
    2. **Faithful translation of quotes.** Translate quotations accurately and naturally into
    English; do not paraphrase them as if they were direct quotes. Keep translated quotes in
    quotation marks (this is standard practice — no disclaimer needed in the article body).
    3. **Preserve named entities correctly.** Keep official names, titles, institutions, and
    place names accurate (e.g., "DPR" → "House of Representatives (DPR)" on first mention,
    then either form after). Retain Indonesian proper nouns; do not anglicize names.
    **Do not manufacture a proper noun the source never gave.** If the source only says a
    generic type of place plus a city or area name — "penginapan di Kupang" (an inn in
    Kupang), not a named business — keep it phrased as a generic place in that location
    ("an inn in Kupang"). Do not compress that into something that reads as the venue's own
    name ("Kupang Inn") — that implies a specific, named establishment the source never
    identified, which is a factual overstatement even though every individual word came from
    the source.
    4. **Localize for an international reader**, without dumbing down:
    - Add a brief in-line gloss for Indonesia-specific terms, institutions, or acronyms on
        first mention (e.g., "Kejaksaan Agung (Attorney General's Office)").
    - Convert Rupiah figures to include an approximate USD equivalent in parentheses where
        material to the story (e.g., "Rp 1.2 trillion (roughly US$74 million)"), using a
        reasonable rounded rate — do not fabricate false precision.
    - Spell out dates in international format and unambiguous month names.
    5. **No literal/word-for-word translation artifacts.** Write in natural, idiomatic English
    news prose — do not mirror Indonesian sentence structure, headline conventions, or
    phrasing patterns.
    6. **Stay neutral and attributed.** Use standard news-attribution language ("according to,"
    "officials said," "the report noted") rather than stating contested claims as fact.

    ## EEAT principles (Track A and `content`)
    Apply Google's EEAT framework (Experience, Expertise, Authoritativeness, Trustworthiness) so
    `title`, `summary`, `content`, `image_cover_image_text`, and `image_cover_alt_image` read as credible,
    high-quality journalism rather than a mechanical translation:

    - **Experience** — Retain concrete, first-hand-feeling detail from the source (specific
    locations, dates, direct observations, on-record reactions) rather than flattening the
    story into abstract generalities.
    - **Expertise** — Use correct official titles, terminology, and institutional processes
    precisely (correct rank, correct legal/administrative terms, correct government body
    names). Do not hedge with vague phrasing where the source supports a precise statement.
    - **Authoritativeness** — Attribute every substantive claim to its origin (a named official,
    a named institution, "state news agency Antara," etc.) exactly as the source does. Do not
    collapse a sourced claim into an unattributed statement.
    - **Trustworthiness** — Distinguish clearly between confirmed fact, official statement, and
    allegation/claim not yet verified, matching the certainty level in the source exactly
    (use "alleged," "according to," "has not been independently verified" where the source
    signals uncertainty). Never upgrade a claim's certainty in translation.

    ## `content` is HTML — preserve the structure, rewrite the prose inside it
    The source `content` is HTML, not plain text. `content` in your output must also be HTML,
    with the original tags, order, and nesting preserved — you are rewriting the text inside the
    tags, not the tags themselves or the sequence they appear in.

    - **Keep every HTML tag from the source intact, in place, and in the same order**: `<p>`,
    `<h2>`/`<h3>` (or whatever heading levels the source uses), `<strong>`, `<em>`,
    `<a href="...">`, `<img>`, `<table>`/`<div>` wrappers, `<figure>`/`<figcaption>`, `<ul>`/`<li>`,
    etc. Do not convert tags to markdown, strip them, reorder them, merge them, or invent new
    ones — except the non-content blocks below.
    - **Rewrite the human-readable text** inside each tag, following all rules above (EEAT,
    faithful quotes, attribution) — but keep each rewritten paragraph's content and order
    matched one-to-one to the source paragraph it replaces; there is no forced paragraph-length
    target and no reflowing across paragraph boundaries. Attribute values that are text meant
    for a reader — `alt`, `title`, `caption`, `<figcaption>` content — get rewritten too;
    attribute values that are data, not prose — `href` URLs, `src` paths, `class`, `id` — must
    be left exactly as in the source, byte-for-byte.
    - **`<img>` tags and their captions**: image markup varies by source — sometimes
    `<figure>/<figcaption>`, sometimes a `<table>`/`<div>` wrapper around `<img>` carrying
    `alt`, `title`, and `caption` attributes plus a trailing `<span>` with the visible caption.
    Whatever the wrapper, keep its tags, position, and `src` unchanged; rewrite the reader-facing
    text in `alt`/`title`/`caption` attributes and any `<figcaption>`/`<span>` caption text —
    but leave a trailing photo-credit fragment (e.g. `(Name/detikcom)`) exactly as in the source,
    since it's a credit line, not prose. Do not move an image to a different point in the
    markup, drop it, or add a new one.
    - **Headings** (`<h2>`, `<h3>`, etc., or bold-as-heading — see below): keep the same tags,
    number, order, and nesting as the source — do not add, remove, merge, or split sections.
    Rewrite the heading text itself with the same natural, idiomatic, AP-style treatment used
    for `title`, glossing Indonesia-specific terms on first mention if the heading contains one.
    - **Bold used as a heading, not a real `<h2>`/`<h3>` tag**: some sources mark section titles
    by wrapping the *entire* content of a `<p>` in `<strong>`/`<b>` instead of a semantic
    heading tag — e.g. `<p><strong>Skema Serahkan Aset KCIC</strong></p>` with no other text in
    that paragraph. Treat this the same as a real heading: keep it as that same bold-wrapped
    `<p>`, in the same position, and rewrite the text with heading-style treatment. Do not
    convert it to `<h2>`/`<h3>` (the source's own tag choice stays), and do not touch a
    `<strong>`/`<b>` run that sits *inside* a paragraph alongside other text — that's inline
    emphasis, not a heading, and should just be rewritten in place like any other text.
    - If the source `content` has no heading tags (real or bold-as-heading), do not invent any —
    keep it as `<p>` paragraphs.

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
    The source `content` may arrive as raw HTML tags (e.g. `<p>...</p>`) or as HTML-entity-escaped
    text (e.g. `&lt;p&gt;...&lt;/p&gt;`). Detect which form the source uses and return your
    `content` in that same form — do not switch a raw-tag source to escaped output, or vice versa.

    ## Output structure
    Produce a single JSON object with exactly these fields: `title`, `summary`, `content`, `tags`,
    `image_cover_image_text`, `image_cover_alt_image`, `keywordauto`, `categoryauto`.

    **Track A fields (full rewrite):**

    - **`title`** — concise, active-voice, AP-style news headline (not clickbait, not a literal
    translation of the Indonesian headline). Plain string, no trailing punctuation.
    **Maximum 65 characters, including spaces.** If the natural rewrite runs longer, you may
    restructure it — reorder clauses, cut filler words, use shorter synonyms, drop a secondary
    clause — rather than truncating mid-word or mid-thought; the title must still read as a
    complete, natural headline within the limit. **Preserve the source headline's own framing,
    not just its facts.** A word like "pamer" (flaunts/shows off) or a "faktanya ternyata"
    (turns out) twist isn't decorative — it's the angle the story is actually told from. If
    space is tight, cut a secondary clause or location detail before cutting the word that
    carries the story's tone or twist; a title that keeps every fact but loses the framing
    ("Alumnus Moves to the US" instead of "Alumnus Flaunts Move to the US") is a different,
    flatter story than the source told.
    - **`summary`** — 1–2 sentence standalone summary of the core who/what/when/where/why.
    Must make sense on its own without reading `content` (this is what will show in article
    previews/listings). Do not just copy the first sentence of `content` verbatim — write it
    as a distinct, compressed summary. **Maximum 130 characters, including spaces.** If the
    who/what/when/where/why doesn't fit in 130 characters, prioritize who/what/when over
    where/why and compress ruthlessly — the result must still be a grammatically complete
    sentence, not a fragment cut off mid-word.
    - **`image_cover_image_text`** — a rewritten (not literally translated) version of the source image
    caption: natural, publication-quality English that conveys the same specific detail as the
    source (who/what is shown), without inventing anything the source caption doesn't state.
    - **`image_cover_alt_image`** — a rewritten (not literally translated) version of the source's image
    `alt` text, in natural English, at the same level of descriptive detail as the source (this
    is separate metadata, not the `alt` attribute already handled inline inside `content`'s own
    `<img>` tags).

    **`content` (structure-preserved rewrite):**

    - **`content`** — the full article body as HTML, with the source's own tag structure, order,
    headings, and image placement preserved exactly (see the HTML section above), but with the
    prose inside each element rewritten — natural, idiomatic, EEAT-quality English — rather
    than translated word-for-word. Do NOT include the headline inside `content` — that belongs
    only in `title`. There is no forced dateline and no ~200-character paragraph target: each
    rewritten paragraph corresponds one-to-one to its source paragraph.

    **Track B fields (translation only, no rewrite treatment):**

    - **`tags`** — the source tags translated into English, same count and order as the source;
    lowercase keyword/topic tags derived only from what's actually present in the source — do
    not invent, drop, merge, or add generic tags.
    - **`keywordauto`** — the source auto-keywords translated into English, one-to-one, same
    count as the source.
    - **`categoryauto`** — the source auto-category translated into English, preserving the same
    category granularity as the source (do not rewrite into a different category scheme).

    Target length for `content`: proportional to the source article's own length — do not
    significantly expand or compress the amount of substantive information conveyed.

    Example shape (values illustrative only):
    ```json
    {
    "title": "...",
    "summary": "...",
    "content": "<p>...</p><h2>Heading One</h2><p>...</p><table class=\"pic_artikel_sisip_table\">...<img src=\"...\" alt=\"...\" title=\"...\" caption=\"...\"/>...</table><p>...</p>",
    "tags": ["...", "...", "..."],
    "image_cover_image_text": "...",
    "image_cover_alt_image": "...",
    "keywordauto": ["...", "...", "..."],
    "categoryauto": "..."
    }
    ```

    ## Before finalizing, check:
    - Does every factual claim trace back to something in the source article?
    - Would a reader with zero context on Indonesia understand every institution/term used?
    - Does `content` read as natural, well-written English while preserving the source's exact
    paragraph order, heading structure, and image placement?
    - Are all figures, names, and dates consistent with the source article?
    - Is every substantive claim attributed exactly as the source attributes it
    (authoritativeness)?
    - Is the certainty level of every claim matched to the source, with nothing upgraded
    (trustworthiness)?
    - Are official titles, terms, and institutional details precise rather than vague
    (expertise)?
    - Does `content` retain concrete, specific detail rather than reading as a flattened
    generic summary (experience)?
    - Is `title` at most 65 characters, and `summary` at most 130 characters — each still a
    complete, natural sentence or headline rather than a truncated fragment?
    - Does `title` keep the source headline's own framing/tone (e.g., "flaunts," a "turns out"
    twist) rather than flattening it into a neutral statement of facts?
    - Does any place mentioned in `title` or `summary` stay a generic description if that's what
    the source gave, instead of reading like the proper name of a specific venue the source
    never actually named?
    - Do `image_cover_image_text` and `image_cover_alt_image` read as natural rewritten English rather than literal
    translations, without adding detail absent from the source?
    - Are `tags`, `keywordauto`, and `categoryauto` faithful translations — not rewritten,
    restructured, or expanded — while still accurate given the article's context?
    - Have all "Baca juga"/"Tonton juga video" style non-content blocks been dropped from
    `content`, with no actual reporting removed along with them?
    - Does the output `content` use the same raw-vs-escaped HTML encoding as the source?"""

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