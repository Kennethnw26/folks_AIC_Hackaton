# Global Treasury Agent

WhatsApp AI agent for cross-border payment reconciliation and BEC fraud detection. A finance ops team sends a payment slip image via WhatsApp; the agent OCRs it, matches it to an open invoice, scores fraud risk, and replies with a confirmation prompt — all in under 60 seconds.

## Stack

- **API:** FastAPI + Uvicorn
- **Persistence:** SQLite via SQLModel (Pydantic v2)
- **Vision / OCR:** `google/gemma-4-31B-turbo` on Chutes (payment slip extraction)
- **LLM:** `deepseek-ai/DeepSeek-V3.2` on Chutes (matching arbitration + fraud narrative)
- **FX:** Frankfurter open API (no key required)
- **Fuzzy match:** `rapidfuzz`
- **Channel:** WhatsApp Cloud API (Meta Graph)

## Project layout

```
.
├── main.py               # FastAPI entry point + dev simulation endpoints
├── orchestrator.py       # WhatsApp message router (image / button / text)
├── config.py             # Pydantic settings (reads .env)
├── seed.py               # DB seed — tenants, invoices, vendor domains
├── agents/
│   ├── matching_agent.py # Invoice matching with LLM arbitration
│   └── fraud_agent.py    # BEC fraud scoring (urgency, domain, beneficiary)
├── tools/
│   ├── ocr.py            # Vision model call → PaymentProof struct
│   ├── matcher.py        # Deterministic DET scoring
│   ├── composer.py       # WhatsApp reply builder
│   ├── fx.py             # FX rate lookup + normalisation
│   ├── chutes_client.py  # Chutes API wrapper (text + vision)
│   └── whatsapp_client.py# Meta Graph API send/receive
├── webhooks/
│   └── whatsapp.py       # Inbound webhook router
├── schemas/
│   └── llm_responses.py  # Pydantic models for LLM JSON outputs
├── db/
│   ├── models.py         # SQLModel table definitions
│   └── session.py        # Engine + table creation
├── tests/
│   ├── conftest.py
│   └── test_health.py
├── test_simulate.py      # HTTP-based end-to-end test (requires server running)
└── test_whatsapp_local.py# Full orchestrator test without Meta network
```

## Setup

```bash
# 1. Create and activate a virtualenv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Fill in CHUTES_API_KEY and META_* values
```

## Environment variables

| Variable | Description |
|---|---|
| `CHUTES_API_KEY` | Chutes.ai bearer token |
| `META_ACCESS_TOKEN` | WhatsApp Cloud API token |
| `META_PHONE_ID` | WhatsApp sender phone number ID |
| `META_VERIFY_TOKEN` | Webhook verification token |
| `META_APP_SECRET` | App secret for signature verification |
| `ENV` | Set to `dev` to enable simulation endpoints |

## Run

```bash
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Local demo (no WhatsApp account needed)

```bash
# Full pipeline test via HTTP endpoint
python test_simulate.py

# Full orchestrator test (patches Meta network calls)
python test_whatsapp_local.py
```

## WhatsApp webhook (real device)

Use ngrok to expose the local server:

```bash
ngrok http 8000
```

Set in Meta App Dashboard:
- **Callback URL:** `https://<subdomain>.ngrok-free.app/webhook/whatsapp`
- **Verify token:** value of `META_VERIFY_TOKEN` in `.env`

## Tests

```bash
pytest -q
```
