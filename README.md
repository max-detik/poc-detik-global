# detikGlobal — Railway deployment

Quick steps to deploy this project to Railway:

1. Create a new project on Railway and connect your Git repository.
2. Ensure the repo contains `Procfile` and `requirements.txt` (already present).
3. Set the following environment variables in the Railway project settings:
   - `OPENROUTER_API_KEY` (required)
   - optional: `OPENROUTER_MODEL`, `OPENROUTER_TEMPERATURE`, `OPENROUTER_REASONING_EFFORT`
4. Deploy — Railway will install dependencies from `requirements.txt` and run `web: python app.py`.

Local testing:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set OPENROUTER_API_KEY in .env
python app.py
```

The server listens on the port provided by the `PORT` environment variable (Railway sets this automatically). Open `http://localhost:8000` when running locally (or the Railway-provided URL after deploy).
