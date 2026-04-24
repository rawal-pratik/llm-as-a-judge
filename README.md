# LLM-as-a-Judge

Evaluate code submissions using multiple LLM judges with bias detection and inter-rater agreement metrics.

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env from example
cp .env.example .env
# Edit .env and add your OpenRouter API key

# 3. Start the API server
uvicorn app.main:app --reload

# 4. Start the dashboard (new terminal)
streamlit run dashboard/app.py
```

- **API docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

## Submit an Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"problem": "Write a function to reverse a string", "code": "def reverse(s): return s[::-1]"}'
```

## Project Structure

```
├── app/                  # FastAPI REST API
│   ├── main.py           # App entry point
│   ├── routes.py         # API endpoints
│   └── schemas.py        # Pydantic request/response models
├── evaluation/           # Core judging engine
│   ├── judge.py          # LLM judge logic + multi-model orchestration
│   ├── prompts.py        # Prompt templates
│   ├── metrics.py        # Cohen's Kappa agreement
│   └── bias.py           # Bias detection
├── models/               # External API clients
│   └── openrouter_client.py
├── database/             # SQLAlchemy + SQLite persistence
│   ├── db.py             # Engine and session
│   ├── models.py         # ORM models
│   └── crud.py           # Database operations
├── dashboard/            # Streamlit visualization
│   └── app.py            # Dashboard entry point
├── config.py             # Central configuration
├── logging_config.py     # Logging setup
└── requirements.txt      # Python dependencies
```

## Deployment

### Render (API)
Set environment variables: `OPENROUTER_API_KEY`, `DATABASE_URL`, `SSL_VERIFY=true`

### Streamlit Cloud (Dashboard)
Set secret: `API_BASE=https://your-render-app.onrender.com`
