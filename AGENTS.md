# AGENTS.md

Hard-won lessons about the Claude Agent SDK and this proxy, so future iterations don't repeat these failures.

## The prompt cache is the entire business model

- The cache is **per model**. A model switch (or a history built by a different model, e.g. Deepseek) invalidates the whole prefix. That is the one case worth compacting.
- A live transcript in the **same model** must never be compacted: resuming reads the cached prefix, so growth is already cheap. Never compact on `resume`.
- The trigger is *"is there a Claude transcript to resume?",* not *"is the history long?".* Persist `model` in `SessionState` to detect a mid-conversation model switch.

## First-party vs third-party billing

- The system prompt is what keeps a request first-party. Use `{ type: "preset", preset: "claude_code" }` and append any custom text **after** `SYSTEM_PROMPT_DYNAMIC_BOUNDARY`. Appending caller text directly onto the preset re-classifies the request as third-party and bills "extra usage".
- `tools: []` does **not** affect first-party classification. Only the system prompt does. The haiku summarizer runs the preset + boundary + its instruction and stays first-party regardless of its tool set.

## CLI errors are hidden behind "exited with code 1"

- The SDK's process transport reports upstream failures as a bare `Claude Code process exited with code 1`. The real cause is only on stderr or in the assistant message.
- Capture stderr per process (`stderr: (data) => buf.push(data)`) and **always** print it on failure. Without it a 429 looks like a 500.
- The CLI reports a spent plan / 5-hour window as a **plain assistant message**: `"You've hit your limit · resets 7:50pm (UTC)"` — not as `API Error: NNN`. Detect it in both `runTurn` and the summarizer and surface as HTTP 429.

## Retry is owned by the proxy, not the CLI

- The CLI retries a 429 up to **10 times** on a subscription (`CLAUDE_CODE_MAX_RETRIES` default 10, and it retries 429 when OAuth/subscription is detected). Set `CLAUDE_CODE_MAX_RETRIES` yourself; a spent plan can never succeed, so cap it at 0 and fail fast.
- opencode/litellm also retry on 429 on their side. The proxy's job is to answer fast and correctly, not to absorb retries.

## The summarizer (haiku)

- **Never summarise the whole cold history.** A model-switch request can carry ~1M chars (~250k tokens) of prior conversation. Summarising all of it was infeasible in time (and multi-chunk was 8 CLI start-ups + 8 cold cache writes), and feeding it to the turn model is the expensive line: ~$0.70-1.10/turn on Sonnet. The haiku summary itself is ~$0.01.
- Instead send haiku **only the recent tail** (~8 messages, ~12k chars / ~3k tokens) and nothing else. Exploring the project (`Read`/`Glob`/`Grep`) was tried and removed: it was slow and made the summary unpredictable. The main model does its own reading on the next turn. The tail must cover the current intent and the immediate tool results, nothing older.
- One warm haiku process (`maxTurns: 1`) is kept waiting so the CLI start-up and the cached `claude_code` prefix are paid once, not per summary. `persistSession: false` keeps it out of the session store.
- A single turn can emit several assistant messages (one with `thinking` content, then the final text). Read the summary from `content` blocks filtered by `type === "text"` and keep the last non-empty text.

## The main turn (`runTurn`)

- `includePartialMessages: true` is what produces `stream_event` messages; the assistant text is reconstructed from them. The summarizer uses `includePartialMessages: false` and reads the assistant message directly.
- The turn must drain the CLI to its `result` **after** emitting `message_stop`, because only `result` carries the session id and flushes the transcript needed to resume the next turn. Don't abort early.
- `warmPool` is keyed by conversation (not model shape): keying by shape handed out blank CLIs and burned the cache. A warm process must resume the conversation's transcript.
- The warm replacement is spawned **after** the turn, holding the id the turn actually produced; spawning it before resumed the previous id and forked the conversation.

## SDK gotchas

- If you ever drive a **multi-turn** `query({ prompt: <AsyncIterable> })` and read replies yourself, hold **one** `q[Symbol.asyncIterator]()` and reuse it across turns. A fresh `for await (const m of q)` creates a new iterator each turn, so the second turn's reply is never delivered. (The current summarizer is single-turn, so this does not bite it — but it did bite the earlier chunked version.)

## Other

- `errorStatus` maps "rate limit"/"hit your limit" text to 429; a ProxyError with status 429 is what the handler returns to litellm. Keep the CLI's real message in the ProxyError so it's visible.
- The proxy reads `OPENCODE_CLAUDE_PROVIDER_DEBUG` for debug-only logs (`claudeLog`); user-facing diagnostics go through `console.error`/`console.log` so they appear in the systemd journal unconditionally.
- `bun run typecheck` (tsc --noEmit) is the gate after every edit. No test suite exists yet; local reproduction scripts against the real SDK (`bun /tmp/x.mjs`) are the fastest way to confirm SDK behaviour before touching the proxy.
