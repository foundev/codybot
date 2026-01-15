# codybot

codybot is a CLI coding agent with a polished Bubble Tea UI. It streams responses from an OpenAI-compatible endpoint and defaults to local Ollama models.

## Quickstart

```bash
go run ./cmd/codybot
```

Defaults:
- Base URL: `http://localhost:11434/v1`
- Model: `llama3`

If `agents.md` is missing, codybot offers to create a starter file for you.

## Configuration

Flags:
- `--base-url` OpenAI-compatible endpoint (default `OPENAI_BASE_URL` or Ollama).
- `--model` model name (default `CODYBOT_MODEL` or `llama3`).
- `--api-key` API key (default `OPENAI_API_KEY`).
- `--agents` path to `agents.md` (default `CODYBOT_AGENTS` or `agents.md`).

Environment variables:
- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `CODYBOT_MODEL`
- `CODYBOT_AGENTS`
