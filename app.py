"""Small stdlib web server for previewing an article and generating its
English rewrite side by side.

Input comes from either the curated local dataset (input/apis-data-all.json,
offered as a picklist) or a pasted detik.com article URL, scraped on demand.
Nothing here calls apis.detik.com.

Reuses the existing scripts:
  - scraper.py          -> scrape_article()     (detik.com URL -> article dict)
  - generate_articles.py-> build_news_input()   (article -> prompt input)
  - prompt.py           -> generate_news_single()(OpenRouter call)

Run:  python detikGlobal/app.py   (then open http://127.0.0.1:8000)
"""

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"
DATA_PATH = Path(os.getenv("ARTICLES_PATH") or BASE_DIR / "input/apis-data-all.json")

# prompt.py / generate_articles.py import each other by bare module name.
sys.path.insert(0, str(BASE_DIR))

from generate_articles import build_news_input  # noqa: E402
from prompt import generate_news_single  # noqa: E402
from scraper import ScrapeError, scrape_article  # noqa: E402


def load_articles():
    """id -> article, keyed by string so query params compare directly."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    return {str(a.get("id")): a for a in articles if a.get("id") is not None}


ARTICLES = load_articles()


class Handler(BaseHTTPRequestHandler):
    server_version = "detikGlobalPreview/1.0"

    # ---------- helpers ----------
    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status, payload):
        self._send(status, json.dumps(payload, ensure_ascii=False))

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------- routes ----------
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            index = WEB_DIR / "index.html"
            return self._send(200, index.read_bytes(), "text/html; charset=utf-8")

        if parsed.path == "/healthz":
            return self._send_json(200, {"status": "ok"})

        if parsed.path == "/api/articles":
            return self._send_json(200, {"articles": [
                {"id": aid, "title": a.get("title", "")} for aid, a in ARTICLES.items()
            ]})

        if parsed.path == "/api/article":
            article_id = (parse_qs(parsed.query).get("id") or [""])[0].strip()
            return self._handle_article(article_id)

        if parsed.path == "/api/scrape":
            url = (parse_qs(parsed.query).get("url") or [""])[0].strip()
            return self._handle_scrape(url)

        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            return self._send_json(404, {"error": "not found"})

        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            return self._send_json(400, {"error": "invalid JSON body"})

        article = payload.get("article")
        article_id = str(payload.get("id") or "").strip()
        url = str(payload.get("url") or "").strip()

        if not article and url:
            try:
                article = scrape_article(url)
            except ScrapeError as e:
                return self._send_json(400, {"error": str(e)})

        if not article:
            article = ARTICLES.get(article_id)
            if article is None:
                return self._send_json(404, {"error": f"unknown article id {article_id!r}"})

        try:
            generated, usage = generate_news_single(build_news_input(article))
        except Exception as e:
            traceback.print_exc()
            return self._send_json(502, {"error": f"{type(e).__name__}: {e}"})

        return self._send_json(200, {"generated": generated, "usage": usage})

    def _handle_article(self, article_id):
        article = ARTICLES.get(article_id)
        if article is None:
            return self._send_json(404, {"error": f"unknown article id {article_id!r}"})
        return self._send_json(200, {"article": article})

    def _handle_scrape(self, url):
        try:
            article = scrape_article(url)
        except ScrapeError as e:
            return self._send_json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._send_json(502, {"error": f"{type(e).__name__}: {e}"})
        return self._send_json(200, {"article": article})


def main():
    # Railway injects PORT; bind 0.0.0.0 so the platform proxy can reach us.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Loaded {len(ARTICLES)} article(s) from {DATA_PATH}", flush=True)
    print(f"Serving on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
        server.server_close()


if __name__ == "__main__":
    main()
