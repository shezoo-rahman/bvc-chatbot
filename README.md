# Stock Insights Assistant

An AI-powered stock insights assistant that lets users ask natural language questions about stocks and get answers using real financial data.

## Architecture

- **Frontend**: Single-file HTML/CSS/JS chat interface
- **Backend**: FastAPI with async support
- **AI**: OpenAI function calling (gpt-4o) for query interpretation and response generation
- **Data**: Finnhub API for real-time stock quotes, company profiles, and news
- **Anti-hallucination**: Two-layer defense — few-shot prompted system instructions + post-response numeric validation

### Query Flow

1. User asks a question via the chat UI
2. FastAPI forwards it to the AI Assistant
3. OpenAI interprets the question and decides which tools to call
4. Tool calls fetch real data from Finnhub (executed concurrently via `asyncio.gather`)
5. OpenAI generates a natural language answer grounded in the tool results
6. Post-validation checks that any numbers in the response match tool data
7. Response is returned to the user

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Finnhub API key (free at [finnhub.io](https://finnhub.io))

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

You can get a free Finnhub API key at [finnhub.io](https://finnhub.io). Then start the app:

```bash
docker compose up
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Local Development

```bash
pip install ".[dev]"
pytest -v
ruff check .
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| FastAPI | Async support, clean API separation, automatic OpenAPI docs |
| Single HTML file | No build step, minimal complexity |
| `httpx.AsyncClient` injection | Enables concurrent API calls + clean test mocking |
| 3 OpenAI tools | Covers quotes, profiles, and news without over-engineering |
| Errors as tool results | Lets the LLM gracefully explain failures to users |
| Post-response validation | Catches hallucinated numbers without blocking the response |

## Future Improvements

- Response caching (Redis) to reduce API calls
- WebSocket support for streaming responses
- Historical price charts
- Portfolio tracking
- Rate limiting per user

## AI Tools Used

This project was built with assistance from Claude (Anthropic).
