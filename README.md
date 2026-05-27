# Global Treasury Agent

A WhatsApp AI agent for cross-border payment reconciliation and BEC (Business Email Compromise) fraud detection. A finance ops team sends a payment slip image via WhatsApp — the agent OCRs it, matches it against open invoices, scores fraud risk, converts the amount to MYR, and replies with a structured confirmation prompt in under 60 seconds.

---

## How It Works

```
User sends payment slip image via WhatsApp
        ↓
1. OCR  — Gemma 4 (31B) extracts: amount, currency, date, beneficiary, reference
        ↓
2. FX   — Frankfurter API converts payment amount → MYR (home currency)
        ↓
3. MATCH — Scores all open invoices using deterministic scoring (amount 40%,
           currency 20%, date 20%, name similarity 20%). If score < 85%,
           DeepSeek V3.2 arbitrates the top 5 candidates.
        ↓
4. FRAUD — 4 signals computed in parallel:
           • new_beneficiary  — not in trusted beneficiary history
           • trust_graph      — no confirmed payment history with this vendor
           • domain_spoof     — email domain similarity to known vendor domains
           • urgency_anomaly  — LLM detects BEC pressure language
        ↓
5. REPLY — WhatsApp interactive message with match details, MYR conversion,
           fraud risk level, narrative, and 3 buttons:
           [Confirm Match] [Flag Fraud] [Skip]
        ↓
6. ACTION — Confirm → invoice closed, beneficiary added to trusted history
            Flag    → payment marked for manual review
            Skip    → payment skipped
```

---

## Key Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI entry point, lifespan (DB init + seed), dev simulation endpoint |
| `orchestrator.py` | Routes WhatsApp messages: image → OCR pipeline, button → action handler |
| `seed.py` | Seeds DB on first start: 5 invoices, 5 trusted vendors, 6 confirmed payments |
| `agents/matching_agent.py` | Deterministic DET scoring + LLM arbitration |
| `agents/fraud_agent.py` | 4-signal fraud scoring with noisy-OR aggregation |
| `tools/ocr.py` | Calls Gemma 4 vision model, returns structured `PaymentProof` |
| `tools/matcher.py` | DET score formula (amount, currency, date, beneficiary) |
| `tools/composer.py` | Builds WhatsApp interactive reply with MYR conversion |
| `tools/fx.py` | Frankfurter + fallback FX rate lookup with disk cache |
| `tools/chutes_client.py` | Chutes AI API wrapper (vision + text, with retry logic) |
| `tools/whatsapp_client.py` | Meta Graph API: download media, send text, send interactive |
| `webhooks/whatsapp.py` | Inbound webhook: signature verification, message parsing |
| `db/models.py` | SQLModel tables: Tenant, Invoice, Payment, BeneficiaryHistory, VendorDomain |

---

## System Requirements

- Python 3.11+
- Windows / macOS / Linux
- [ngrok](https://ngrok.com) (for live WhatsApp webhook)
- A [Chutes.ai](https://chutes.ai) account with API key
- A Meta Developer account with WhatsApp Business API app

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd AIC_Hackaton_folks

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux
# Fill in all values in .env (see table below)
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `CHUTES_API_KEY` | Chutes.ai bearer token — get from chutes.ai dashboard |
| `META_ACCESS_TOKEN` | WhatsApp Cloud API access token (regenerate every 24h for testing) |
| `META_PHONE_ID` | WhatsApp sender phone number ID from Meta API Setup page |
| `META_VERIFY_TOKEN` | Any string you choose — must match what you set in Meta webhook config |
| `META_APP_SECRET` | App secret from Meta → App Settings → Basic |
| `DATABASE_URL` | SQLite path — leave as default: `sqlite:///./treasury.db` |
| `ENV` | Set to `dev` to enable `/dev/simulate_image` test endpoint |

---

## Running the Application

```bash
# Start the server (auto-creates and seeds the database on first run)
uvicorn main:app --reload --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
# → {"status":"ok","version":"1.0.0"}
```

The database (`treasury.db`) is created and seeded automatically on first startup with:
- 1 tenant: Demo Sdn Bhd (home currency: MYR)
- 5 open invoices (INV-001 to INV-005) across USD, SGD, EUR, MYR, CNY
- 5 trusted vendor profiles with pre-confirmed payment history

To reset the database to a clean state:
```bash
del treasury.db              # Windows
# rm treasury.db             # macOS / Linux
uvicorn main:app --reload --port 8000
```

---

## WhatsApp Integration (Live Demo)

### Step 1 — Expose local server via ngrok
```bash
ngrok http 8000
# Copy the https://<subdomain>.ngrok-free.dev URL
```

### Step 2 — Configure Meta webhook
In Meta Developer Console → Your App → Use Cases → Connect on WhatsApp → Configuration:
- **Callback URL:** `https://<subdomain>.ngrok-free.dev/webhooks/whatsapp`
- **Verify token:** value of `META_VERIFY_TOKEN` in `.env`
- **Subscribe** the `messages` webhook field

### Step 3 — Add recipient phone number
In Meta → API Setup → "To" dropdown → Manage phone number list → add your WhatsApp number.

### Step 4 — Send a payment slip
Send any payment receipt image to the test number (`+1 555 644 8023`) from your registered WhatsApp number. The agent will reply within 30–60 seconds.

---

## Demo Scenarios

| Receipt | Expected Result |
|---|---|
| $5,000 USD to Acme Corporation, ref INV-001 | Match INV-001, 90%+ confidence, LOW risk |
| RM 45,000 to Local Supplier Sdn Bhd, ref INV-004 | Match INV-004, 95%+ confidence, LOW risk |
| SGD 12,500 to Global Logistics Pte, ref INV-002 | Match INV-002, 95%+ confidence, LOW risk |
| $5,000 USD to **Acme Corporation LLC**, account FRAUD-9999 | Match INV-001 (amount matches), HIGH risk — unknown account + name mismatch |
| EUR 9,200 to Partners GmbH, ref **URGENT PAYMENT** | Match INV-003 (close amount), HIGH risk — urgency language detected |
| RM 45,000 to **Syarikat Mutiara Sdn Bhd** | No match or low confidence, HIGH risk — completely unknown beneficiary |

---

## Tests

```bash
pytest -q
```

---

## AI Models Used

| Task | Model | Provider |
|---|---|---|
| Payment slip OCR / vision | `google/gemma-4-31B-turbo-TEE` | Chutes.ai |
| Invoice matching arbitration | `deepseek-ai/DeepSeek-V3.2-TEE` | Chutes.ai |
| Fraud urgency detection | `deepseek-ai/DeepSeek-V3.2-TEE` | Chutes.ai |
| Fraud narrative generation | `deepseek-ai/DeepSeek-V3.2-TEE` | Chutes.ai |
| FX rates | Frankfurter open API | frankfurter.dev |
