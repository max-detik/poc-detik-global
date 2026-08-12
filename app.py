"""Small stdlib web server for previewing an article by ID and generating its
English rewrite side by side.

Reuses the existing scripts:
  - apis-data.py        -> fetch_article()      (fetch + field extraction)
  - generate_articles.py-> build_news_input()   (article -> prompt input)
  - prompt.py           -> generate_news_single()(OpenRouter call)

Run:  python detikGlobal/app.py   (then open http://127.0.0.1:8000)
"""

import importlib.util
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"

# prompt.py / generate_articles.py import each other by bare module name.
sys.path.insert(0, str(BASE_DIR))


def _load_hyphenated_module(name, filename):
    """apis-data.py isn't a valid identifier, so load it by path."""
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

apis_data = _load_hyphenated_module("apis_data", "apis-data.py")
fetch_article = apis_data.fetch_article

from generate_articles import build_news_input  # noqa: E402
from prompt import generate_news_single  # noqa: E402


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

        if parsed.path == "/api/article":
            article_id = (parse_qs(parsed.query).get("id") or [""])[0].strip()
            return self._handle_article(article_id)

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

        try:
            if not article:
                if not article_id.isdigit():
                    return self._send_json(400, {"error": "numeric article id required"})
                article = fetch_article(int(article_id))

            generated, usage = generate_news_single(build_news_input(article))
        except Exception as e:
            traceback.print_exc()
            return self._send_json(502, {"error": f"{type(e).__name__}: {e}"})

        return self._send_json(200, {"generated": generated, "usage": usage})

    def _handle_article(self, article_id):
        if not article_id.isdigit():
            return self._send_json(400, {"error": "numeric article id required"})
        try:
            article = fetch_article(int(article_id))
        except Exception as e:
            traceback.print_exc()
            return self._send_json(502, {"error": f"{type(e).__name__}: {e}"})
        return self._send_json(200, {"article": article})


def main():
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
