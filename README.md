# Global Treasury Agent

FastAPI service that reconciles cross-border payments to invoices, looks up FX
rates, screens for fraud, and exposes a WhatsApp ops channel. The reasoning
core is Hermes 4 (405B) served through the Chutes inference API, called via
the `openai` Python client with a custom `base_url`.

## Stack

- **API:** FastAPI + Uvicorn
- **Persistence:** SQLite via SQLModel (Pydantic v2 underneath)
- **LLM:** Hermes 4 on Chutes, accessed with the `openai` SDK
- **FX:** Frankfurter (no API key required)
- **Similarity / fuzzy match:** `sentence-transformers`, `rapidfuzz`, `scipy`
- **Channel:** WhatsApp Cloud API (Meta Graph)
- **OCR / vision:** Pillow + Hermes vision (wired in a later prompt)

## Project layout

```
treasury-agent/
  api/         # HTTP routers (/reconcile, /invoices, ...)
  agent/       # Hermes 4 client + prompts + tool dispatch
  db/          # SQLModel models, engine, seeds
  fraud/       # FlaggedAccount lookups, signal extraction
  ocr/         # payment-slip OCR + field extraction
  fx/          # Frankfurter client + fee inference
  whatsapp/    # inbound dispatch + outbound send
  exports/     # CSV / Excel batch exports
  demo/        # scripted demo flows
  tests/       # pytest suite
  main.py
  requirements.txt
  .env.example
  README.md
```

## Setup

```bash
# 1. Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in CHUTES_API_KEY, WHATSAPP_* values.

# 4. Initialise the SQLite database and seed minimal data
python -m db.seed_min
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Expose locally for WhatsApp webhooks

The Meta Cloud API needs a public HTTPS URL. Use ngrok:

```bash
ngrok http 8000
```

Take the `https://<subdomain>.ngrok-free.app` URL ngrok prints and configure
it in the Meta App dashboard:

- **Callback URL:** `https://<subdomain>.ngrok-free.app/webhook/whatsapp`
- **Verify token:** the value you set as `WHATSAPP_VERIFY_TOKEN` in `.env`

Meta will hit `GET /webhook/whatsapp` once with `hub.mode=subscribe` and the
verify token; `main.py` echoes the challenge back when the token matches.

## Tests

```bash
pytest -q
```

## Environment variables

See `.env.example`. Required keys: `CHUTES_API_KEY`, `WHATSAPP_TOKEN`,
`WHATSAPP_PHONE_ID`, `WHATSAPP_VERIFY_TOKEN`, `FRANKFURTER_BASE`,
`DATABASE_URL`.
