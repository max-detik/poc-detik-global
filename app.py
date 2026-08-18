"""Small stdlib web server for previewing an article and generating its
English rewrite side by side.

Input is one to five pasted detik.com article URLs, scraped on demand. The first
one is the main article (anchor); the rest only enrich it, and the output keeps
the same fields either way. Nothing here calls apis.detik.com.

Reuses the existing scripts:
  - scraper.py          -> scrape_article()       (detik.com URL -> article dict)
  - generate_articles.py-> generate_from_articles()(articles -> generated article)
  - prompt.py           -> generate_news_single() / generate_news_multi()

The whole site sits behind HTTP Basic auth; credentials come from BASIC_AUTH_USER
and BASIC_AUTH_PASS in .env (Railway: set them as service variables).

Run:  python detikGlobal/app.py   (then open http://127.0.0.1:8000)
"""

import base64
import hmac
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"

load_dotenv()
BASIC_AUTH_USER = os.getenv("BASIC_AUTH_USER", "")
BASIC_AUTH_PASS = os.getenv("BASIC_AUTH_PASS", "")
AUTH_REALM = "detikGlobal"

# prompt.py / generate_articles.py import each other by bare module name.
sys.path.insert(0, str(BASE_DIR))

from generate_articles import generate_from_articles  # noqa: E402
from prompt import MAX_SOURCE_ARTICLES  # noqa: E402
from scraper import ScrapeError, scrape_article  # noqa: E402


def check_credentials(header_value):
    """True when the Authorization header carries the configured user/pass."""
    scheme, _, encoded = (header_value or "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        user, _, password = base64.b64decode(encoded, validate=True).decode("utf-8").partition(":")
    except Exception:
        return False
    # Both compared, non-short-circuit, so timing does not leak the username.
    ok_user = hmac.compare_digest(user, BASIC_AUTH_USER)
    ok_pass = hmac.compare_digest(password, BASIC_AUTH_PASS)
    return ok_user & ok_pass


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

    def _authorized(self):
        """Gate everything but /healthz, which the platform probes unauthenticated."""
        if urlparse(self.path).path == "/healthz":
            return True
        if check_credentials(self.headers.get("Authorization")):
            return True
        body = json.dumps({"error": "authentication required"}).encode("utf-8")
        self.send_response(401)
        self.send_header("WWW-Authenticate", f'Basic realm="{AUTH_REALM}", charset="UTF-8"')
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return False

    # ---------- routes ----------
    def do_GET(self):
        if not self._authorized():
            return

        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            index = WEB_DIR / "index.html"
            return self._send(200, index.read_bytes(), "text/html; charset=utf-8")

        if parsed.path == "/healthz":
            return self._send_json(200, {"status": "ok"})

        if parsed.path == "/api/scrape":
            url = (parse_qs(parsed.query).get("url") or [""])[0].strip()
            return self._handle_scrape(url)

        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return

        if urlparse(self.path).path != "/api/generate":
            return self._send_json(404, {"error": "not found"})

        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            return self._send_json(400, {"error": "invalid JSON body"})

        # The first entry is the main article (anchor); the rest only enrich it.
        # `article`/`url` (singular) stay accepted for older callers.
        articles = payload.get("articles") or ([payload["article"]] if payload.get("article") else [])
        urls = payload.get("urls") or []
        if not urls and payload.get("url"):
            urls = [payload["url"]]

        if not articles:
            for url in urls:
                try:
                    articles.append(scrape_article(str(url).strip()))
                except ScrapeError as e:
                    return self._send_json(400, {"error": str(e)})

        if not articles:
            return self._send_json(400, {"error": "provide an article or a detik.com url"})
        if len(articles) > MAX_SOURCE_ARTICLES:
            return self._send_json(
                400, {"error": f"at most {MAX_SOURCE_ARTICLES} articles are supported"}
            )

        try:
            generated, usage = generate_from_articles(articles)
        except Exception as e:
            traceback.print_exc()
            return self._send_json(502, {"error": f"{type(e).__name__}: {e}"})

        return self._send_json(200, {"generated": generated, "usage": usage})

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
    # Refuse to start unauthenticated rather than exposing the site by accident.
    if not (BASIC_AUTH_USER and BASIC_AUTH_PASS):
        sys.exit("Set BASIC_AUTH_USER and BASIC_AUTH_PASS (see .env.example) before starting.")

    # Railway injects PORT; bind 0.0.0.0 so the platform proxy can reach us.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
        server.server_close()


if __name__ == "__main__":
    main()
