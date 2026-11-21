---
applyTo: '**'
---

# project.instructions.md

This file provides guidance to AI Assistant when working with code in this repository.

## Commands
- Install: `bun install` (Bun only; never npm/node).
- Run proxy: `bun run proxy` or `bun run start` (host/port via `CLAUDE_PROXY_PORT`/`CLAUDE_PROXY_HOST`).
- Typecheck (mandatory after edits): `bun run typecheck` (`tsc --noEmit`). There is no build step.
- Tests: `bun test`. Single file: `bun test tests/proxy.test.ts`. Single test: `bun test -t "name"` or `bun test tests/proxy.test.ts -t "name"`. No test files exist yet; put new ones under `tests/`.

## Style
- TypeScript ESM, no semicolons, double quotes, 2-space indent. Extensionless relative imports (`./types`, `../logger`); type-only imports use `import type`.
- `strict` + `noUncheckedIndexedAccess` + `verbatimModuleSyntax`: type everything, avoid `any` except at SDK/JSON boundaries. Node builtins from `crypto`, `fs`, `path`, `os`; Bun APIs (`Bun.serve`) for the server.
- Naming: `camelCase` functions/variables, `PascalCase` types/interfaces, `SCREAMING_SNAKE` module constants, `CLAUDE_PROXY_*` env vars.
- Errors: throw `ProxyError(message, status, errorType)` for HTTP-facing failures; log via `claudeLog(event, {...})` (debug-gated) and `console.error` for real failures; `catch {}` only for best-effort cleanup. Never add comments unless they explain non-obvious intent (existing style uses long rationale comments).

## Overview
The proxy exposes an Anthropic Messages API (`POST /messages`, `/v1/messages`) on Hono/Bun and translates each request into a single Claude Agent SDK `query()` call that authenticates through the user's Claude CLI login (subscription billing, no API key). Caller tools are registered as inert MCP tools (`mcp__client__*`); the `tool_use` block is forwarded, the turn ends, and the client executes the tool. Responses are re-emitted as Anthropic SSE events or folded into one JSON message.

Most of `src/proxy/server.ts` handles token-efficiency invariants: the caller's system prompt is injected after Claude Code's dynamic boundary to keep first-party billing and cache the stable prefix; per-caller sessions (`x-opencode-session-id` or fingerprint) persist in `~/.claude-proxy/sessions/<sha256>.json` so later turns send only new messages via `resume`; integrity checks fall back to full replay on compaction/retry; per-turn serialization prevents races; a warm CLI pool keyed by conversation serves one request then respawns; a sweeper expires idle sessions and scratch data.

Entry point is `bin/claude-proxy.ts`; `src/proxy/types.ts` holds `ProxyConfig` defaults; `src/logger.ts` the debug logger. Everything runs locally and nothing is executed on the proxy host.
