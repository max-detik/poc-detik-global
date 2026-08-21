# detikGlobal — Railway deployment

Quick steps to deploy this project to Railway:

1. Create a new project on Railway and connect your Git repository.
2. Ensure the repo contains `Procfile` and `requirements.txt` (already present).
3. Set the following environment variables in the Railway project settings:
   - `OPENROUTER_API_KEY` (required)
   - `BASIC_AUTH_USER` / `BASIC_AUTH_PASS` (required) — HTTP Basic credentials guarding the site; the app exits at startup if either is missing
   - optional: `OPENROUTER_MODEL`, `OPENROUTER_TEMPERATURE`, `OPENROUTER_REASONING_EFFORT`
4. Deploy — Railway will install dependencies from `requirements.txt` and run `web: python -m web.app`.

Local testing:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OPENROUTER_API_KEY, BASIC_AUTH_USER and BASIC_AUTH_PASS in .env
python -m web.app
```

The server listens on the port provided by the `PORT` environment variable (Railway sets this automatically). Open `http://localhost:8000` when running locally (or the Railway-provided URL after deploy).

## Layout

Everything runs as a module from the repo root (`python -m <package>.<module>`),
so the packages can import each other without any `sys.path` juggling.

```
generation/    article rewriting and keyword/category generation
  llm.py               shared OpenRouter client, prompt loading, response parsing
  news.py              generate_news_multi() — 1..5 Indonesian sources -> one English article
  keywords_category.py generate_keywords_category() — content -> keywords + ranked categories
  articles.py          generate_from_articles() — scraped articles -> generated article
  prompts/             the system prompts, as plain text
scraping/      detik.com scraping
  scraper.py           scrape_article() — URL -> article dict
  apis_data.py         pulls sample articles from apis.detik.com
evaluation/    scoring the category output
  scoring.py           taxonomy resolution + multiclass metrics, shared by both evals
  keywords_category.py scores generate_keywords_category() against a labelled dataset
  es_categories.py     scores the production categoriser's rank-N pick
scripts/       one-off jobs
  get_from_es.py       fetches article fields from Elasticsearch
  dump_prompts.py      writes the system prompts out as the model receives them
  generate_global.py   older single-file generation run
web/           the preview site (app.py + index.html)
input/         datasets and the category labelling CSV
output/        generated articles and evaluation results
```

Common commands:

```bash
python -m web.app                        # preview site
python -m generation.articles            # batch-generate from input/apis-data-all.json
python -m evaluation.keywords_category   # score the category output
python -m evaluation.es_categories       # score the production categoriser
python -m scripts.get_from_es            # pull article fields from Elasticsearch
python -m scripts.dump_prompts           # render the system prompts to output/prompts/
```
