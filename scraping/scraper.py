"""Scrape a detik.com article page into the same dict shape as the curated
dataset (input/apis-data-all.json), so the rest of the pipeline is unchanged.

Only detik.com hosts are accepted — anything else raises ScrapeError.

Usage:  python scraper.py <url>
"""

import json
import re
import sys
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup, Comment

ALLOWED_HOST_SUFFIX = ".detik.com"
TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Elements inside the article body that aren't reporting: related-article
# boxes, embeds' scaffolding, inline promos, the tag strip at the end.
JUNK_SELECTORS = [
    "script", "style", "noscript", "ins", "iframe", "svg", "button", "form",
    ".lihatjg", ".linksisip", ".detail__body-tag", ".detail__multibrand",
    ".staticdetail_container", ".parallaxindetail", ".clearfix",
    ".sisipan_artikel", ".detail__anchor-tag", ".pagination", ".paging",
    # "Daftar Isi" table-of-contents widget, plus share/subscribe UI.
    ".table-content", "details.table-content", ".detail__share", ".sharebar",
    ".noncontent", ".aevp", ".detail__newstag", ".pic_artikel_sisip",
]
JUNK_CLASS_HINTS = ("ads", "advert", "banner", "baca-juga", "bacajuga")

BACA_JUGA_RE = re.compile(r"^\s*(baca juga|tonton juga|simak juga|lihat juga)\s*:", re.I)
# Byline initials detik appends to the end of every article, e.g. "(hrp/fdl)".
BYLINE_RE = re.compile(r"^\(\s*[\w.\-]+\s*/\s*[\w.\-]+\s*\)$")


class ScrapeError(Exception):
    """Raised for a rejected URL or a page we can't parse."""


def normalize_url(url):
    """Validate the host and ask detik for the whole article on one page."""
    url = (url or "").strip()
    if not url:
        raise ScrapeError("url is required")
    if "://" not in url:
        url = "https://" + url

    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise ScrapeError(f"unsupported scheme {parts.scheme!r}")

    host = (parts.hostname or "").lower()
    if not (host == "detik.com" or host.endswith(ALLOWED_HOST_SUFFIX)):
        raise ScrapeError(f"only detik.com URLs are allowed, got {host!r}")

    # Long articles are paginated; single=1 returns the full text in one fetch.
    query = dict(parse_qsl(parts.query))
    query["single"] = "1"
    return urlunparse(parts._replace(netloc=host, query=urlencode(query), fragment=""))


def fetch_html(url):
    response = requests.get(
        url,
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "id-ID,id;q=0.9"},
    )
    response.raise_for_status()
    return response.text


def _meta(soup, **attrs):
    tag = soup.find("meta", attrs=attrs)
    return (tag.get("content") or "").strip() if tag else ""


def _pipe_join(values):
    seen, out = set(), []
    for value in values:
        value = (value or "").strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return "|".join(out)


def _article_id(soup, url):
    for attrs in ({"name": "articleid"}, {"name": "dtk:articleid"}):
        value = _meta(soup, **attrs)
        if value:
            return value
    match = re.search(r"/d-(\d+)", url)
    return match.group(1) if match else ""


def _select_first(soup, selectors):
    """select_one() with a comma list follows document order; we want our own
    preference order instead (the inner text node before its wrapper)."""
    for selector in selectors:
        found = soup.select_one(selector)
        if found is not None:
            return found
    return None


def _truncate_at_byline(body):
    """detik closes every article with "(initials/initials)"; tag strips, share
    widgets and related-video rails all live after it, so cut there."""
    for node in body.find_all(recursive=False):
        if node.name in ("strong", "b", "p") and BYLINE_RE.match(node.get_text(" ", strip=True)):
            for sibling in list(node.find_next_siblings()):
                sibling.decompose()
            node.decompose()
            return


def _clean_body(body):
    for comment in body.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    _truncate_at_byline(body)

    for selector in JUNK_SELECTORS:
        for node in body.select(selector):
            node.decompose()

    # Each pass snapshots the tree first, so skip nodes an earlier decompose()
    # already took out from under us.
    for node in list(body.find_all(class_=True)):
        if node.decomposed:
            continue
        classes = " ".join(node.get("class") or []).lower()
        if any(hint in classes for hint in JUNK_CLASS_HINTS):
            node.decompose()

    # Stray "Baca juga:" lines that aren't wrapped in a known container.
    for node in list(body.find_all(["p", "div", "strong"])):
        if not node.decomposed and BACA_JUGA_RE.match(node.get_text(" ", strip=True)):
            node.decompose()

    for node in list(body.find_all(["p", "div", "strong"])):
        if node.decomposed:
            continue
        if BYLINE_RE.match(node.get_text(" ", strip=True)):
            node.decompose()

    for node in list(body.find_all(["p", "div"])):
        if node.decomposed:
            continue
        if not node.get_text(strip=True) and not node.find(["img", "figure"]):
            node.decompose()

    return body.decode_contents().strip()


def _category(soup, url):
    """Breadcrumb leaf (e.g. "Ekonomi Bisnis"); falls back to the URL path."""
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
            items = data.get("itemListElement") or []
            if items:
                return str(items[-1].get("name") or "").lower()

    segments = [s for s in urlparse(url).path.split("/") if s and not s.startswith("d-")]
    return segments[0].replace("-", " ") if segments else ""


def parse_article(html, url):
    soup = BeautifulSoup(html, "html.parser")

    title_el = _select_first(soup, ["h1.detail__title", ".detail__title", "h1"])
    title = title_el.get_text(" ", strip=True) if title_el else _meta(soup, property="og:title")

    body = _select_first(soup, [".detail__body-text", ".itp_bodycontent", ".detail__body"])
    if not title or body is None:
        raise ScrapeError("could not find the article body — is this a detik article URL?")

    caption_el = _select_first(soup, [".detail__media-caption", ".detail__media figcaption"])
    image_el = _select_first(soup, [".detail__media-image img", ".detail__media img"])

    keywords = _meta(soup, name="dtk:keywords") or _meta(soup, name="keywords")
    tags = [a.get_text(" ", strip=True) for a in soup.select(".detail__body-tag a, .nav__tag a")]

    return {
        "id": _article_id(soup, url),
        "title": title,
        "content": _clean_body(body),
        "resume": _meta(soup, name="description") or _meta(soup, property="og:description"),
        "tag": _pipe_join(tags) or _pipe_join(keywords.split(",")),
        "keywordauto": _pipe_join(keywords.split(",")),
        "categoryauto": _category(soup, url),
        "image_cover": {
            "text": caption_el.get_text(" ", strip=True) if caption_el else "",
            "alt_image": (image_el.get("alt") or "").strip() if image_el else "",
        },
        "source_url": url,
    }


def scrape_article(url):
    normalized = normalize_url(url)
    try:
        html = fetch_html(normalized)
    except requests.RequestException as e:
        raise ScrapeError(f"could not fetch the page: {e}") from e
    return parse_article(html, normalized)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    print(json.dumps(scrape_article(sys.argv[1]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
