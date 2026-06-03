# Stock Insights Assistant

An AI-powered stock insights assistant that lets users ask natural language questions about stocks and get intelligent, data-backed answers using real-time financial data.

## Architecture

```
┌─────────────┐     POST /api/query      ┌──────────────┐
│   Browser    │ ──────────────────────── │   FastAPI     │
│  (HTML/JS)   │                          │   Routes      │
└─────────────┘                          └──────┬───────┘
                                                │
                                         ┌──────▼───────┐
                                         │ AI Assistant  │
                                         │ (tool loop)  │
                                         └──┬───────┬───┘
                                            │       │
                                  ┌─────────▼─┐ ┌───▼──────────┐
                                  │  OpenAI    │ │   Finnhub    │
                                  │  GPT-4o    │ │   Stock API  │
                                  └────────────┘ └──────────────┘
```

- **Frontend**: Single-file HTML/CSS/JS chat interface (Carbon Design inspired)
- **Backend**: FastAPI with async lifespan context manager
- **AI**: OpenAI function calling with 4 tools (quote, profile, news, symbol search)
- **Data**: Finnhub API for real-time US stock quotes, company profiles, and news
- **Anti-hallucination**: System prompt constraints prevent the model from inventing data, with a post-response check that flags any numbers not grounded in tool results

### Query Flow

1. User asks a question via the chat UI (session ID sent with each request)
2. FastAPI forwards it to the AI Assistant with conversation history
3. OpenAI interprets the question and decides which tools to call
4. Tool calls fetch real data from Finnhub (executed concurrently via `asyncio.gather`)
5. OpenAI generates a natural language answer grounded in the tool results
6. Post-validation checks that any numbers in the response match tool data
7. Response is returned and conversation history is stored for follow-ups

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Finnhub API key — free at [finnhub.io](https://finnhub.io) (sign up, key is on your dashboard)

### Setup

```bash
git clone https://github.com/shezoo-rahman/bvc-chatbot.git
cd bvc-chatbot
cp .env.example .env
nano .env
```

Replace the placeholder values with your real API keys, then save with `Ctrl+O` (Enter to confirm) and exit with `Ctrl+X`:

```
OPENAI_API_KEY=sk-your-openai-api-key
FINNHUB_API_KEY=your-finnhub-api-key
OPENAI_MODEL=gpt-4o
```

Then start the app:

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Running Tests Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
pytest -v
ruff check .
ruff format --check .
deactivate  # exit the virtual environment
```

## Design Decisions & Trade-offs

| Decision | Rationale |
|----------|-----------|
| FastAPI over Flask | Async support for concurrent Finnhub calls, clean dependency injection, automatic OpenAPI docs |
| Single HTML file over React | No build step, zero frontend dependencies, keeps the focus on backend/AI quality |
| `httpx.AsyncClient` injection | Enables concurrent API calls via `asyncio.gather` + clean test mocking without hitting external APIs |
| 4 OpenAI tools | `search_symbol` for disambiguation, plus quotes, profiles, and news — covers the core use cases without over-engineering |
| In-memory session history | Enables multi-turn conversations (e.g. disambiguation follow-ups) without adding a database dependency. Trade-off: history is lost on restart, which is acceptable for a demo |
| Errors returned as tool results | Instead of raising exceptions to the user, errors are passed back to the LLM so it can explain failures naturally (e.g. "I couldn't find that symbol") |
| Post-response validation | Regex extracts numbers from the LLM response and checks them against tool data — catches hallucinated figures without blocking responses that contain no numbers |
| Finnhub free tier | Generous rate limits (60 calls/min), real-time US stock data. Trade-off: no international stocks, news relevance varies for conglomerates |
| Prompt engineering over code logic | Behaviour like table formatting, disambiguation, and off-topic rejection is handled via system prompt rather than hard-coded rules — more flexible and easier to iterate |

## What I Would Improve With More Time

- **Persistent sessions** — Replace in-memory dict with Redis for conversation history that survives restarts and scales horizontally
- **Streaming responses** — WebSocket support so users see tokens as they arrive instead of waiting for the full response
- **Response caching** — Cache Finnhub responses (short TTL) to reduce API calls for repeated queries
- **Rate limiting** — Per-user rate limiting to prevent abuse in a shared deployment
- **Richer data** — Add historical price charts, earnings data, and sector comparisons (would require a paid data tier)
- **Observability** — Distributed tracing (e.g. OpenTelemetry) to correlate requests across LLM and Finnhub calls

## AI Tools Used

I built this project using **Claude Code** (Anthropic's CLI agent) as a pair-programming assistant to accelerate development:

- **Architecture** — I designed the service layer structure and chose FastAPI, OpenAI function calling, and Finnhub; Claude helped scaffold boilerplate and wire up Docker/CI
- **Prompt engineering** — I tested extensively against edge cases (disambiguation, international stocks, off-topic queries, hallucination) and refined the system prompt across multiple rounds based on real outputs
- **Debugging** — I identified issues like Finnhub 403s on non-US stocks, ticker filtering bugs (`.TO`, `.F` suffixes vs `.A`, `.B` share classes), and conversation history loss; Claude helped trace root causes and implement fixes
- **Code quality** — I rejected several over-engineered suggestions: a decorator-based tool registry (unnecessary for 4 tools), Redis for session storage (overkill for a demo), and excessive abstraction layers. I kept the codebase simple and appropriately scoped for the task
