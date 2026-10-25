# claude-subscription-proxy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/franlerma/claude-subscription-proxy.svg)](https://github.com/franlerma/claude-subscription-proxy/stargazers)

Use your **Claude subscription** (Max or Pro) with OpenCode or any other Anthropic-compatible tool — no separate API key or per-token billing required.

> **Disclaimer:** This project is non-profit and made for educational and research purposes only.

## The Problem

Anthropic doesn't allow Claude Max subscribers to use their subscription with third-party tools like OpenCode. If you want to use Claude in OpenCode, you have to pay for API access separately - even though you're already paying for "unlimited" Claude.

Your options are:
1. Use Claude's official apps only (limited to their UI)
2. Pay again for API access on top of your Max subscription
3. **Use this proxy**

## The Solution

This proxy bridges the gap using Anthropic's own tools:

```
OpenCode → Proxy (localhost:3456) → Claude Agent SDK → Your Claude Max Subscription
```

The [Claude Agent SDK](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk) is Anthropic's **official npm package** that lets developers build with Claude using their Max subscription. This proxy simply translates OpenCode's API requests into SDK calls.

**Your Max subscription. Anthropic's official SDK. Zero additional cost.**

## What this fork adds

This project builds on [opencode-claude-max-proxy](https://github.com/rynfar/opencode-claude-max-proxy) with a focus on efficient, safe multi-turn sessions:

- **Prompt caching** — Claude Code's system prompt is kept as a stable, cacheable prefix, so later turns in a conversation read from the prompt cache (~40k+ tokens) instead of rewriting the full context every turn.
- **First-party billing preserved** — the caller's system prompt is injected after Claude Code's dynamic-boundary marker, so Anthropic still bills the request against the subscription and never degrades it to a third-party "extra usage" call.
- **Multi-turn session resume** — each caller session maps to a persistent Claude Code transcript, so every turn sends only the newest message instead of replaying the whole history. The mapping survives proxy restarts.
- **Session integrity checks** — before resuming, the proxy verifies that the caller's history still contains the assistant turn the transcript holds. Compaction, regenerate/retry, truncation or a second writer triggers a full replay instead of appending to stale history.
- **Stable tool prefix** — tool definitions are deduplicated, sorted and schema-canonicalized, so a caller that reorders tools or JSON keys does not silently invalidate the whole prompt cache.
- **Frozen caller system prompt** — a system prompt that changes mid-conversation is delivered as a conversation message instead of rewriting Claude Code's cached system block and every tool definition ahead of it.
- **Environment hardening** — ambient `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_FORCE_GLOBAL_CACHE`, `MAX_THINKING_TOKENS` and cache-disabling flags are removed before the CLI starts, so they cannot change billing, thinking budget or caching.
- **MCP tool federation** — OpenCode's MCP tools are federated to the underlying Claude Code process, so tool calls keep working across turns.
- **Per-conversation warm pool** — pre-started Claude Code processes resume a conversation's transcript, cutting time-to-first-byte without losing context.

## Is This Allowed?

**Yes.** Here's why:

| Concern | Reality |
|---------|---------|
| "Bypassing restrictions" | No. We use Anthropic's public SDK exactly as documented |
| "Violating TOS" | No. The SDK is designed for programmatic Claude access |
| "Unauthorized access" | No. You authenticate with `claude login` using your own account |
| "Reverse engineering" | No. We call `query()` from their npm package, that's it |

The Claude Agent SDK exists specifically to let Max subscribers use Claude programmatically. We're just translating the request format so OpenCode can use it.

**Plain TypeScript. No hacks. No magic. Just format translation.**

## Features

| Feature | Description |
|---------|-------------|
| **Zero API costs** | Uses your Claude subscription, not per-token billing |
| **Prompt caching** | Keeps the system prompt a stable cacheable prefix so multi-turn sessions read from cache |
| **First-party billing** | Preserves subscription billing — never degrades to third-party "extra usage" |
| **Session resume** | Persists and resumes transcripts so later turns send only the newest message |
| **Session integrity** | Detects compaction, retries and branching and replays instead of resuming stale history |
| **Stable cache prefix** | Sorts/dedupes tools and canonicalizes JSON schemas so reordering cannot bust the cache |
| **System prompt freeze** | Applies mid-session system prompt changes as conversation messages, keeping the cached prefix warm |
| **MCP tool federation** | Federates OpenCode's MCP tools to the underlying Claude Code process |
| **Streaming support** | Real-time SSE streaming just like the real API |
| **Warm pool** | Pre-started processes resume a conversation, cutting time-to-first-byte |
| **Auto-start** | Optional launchd service for macOS |
| **Simple setup** | Two commands to get running |

## Prerequisites

1. **Claude Max subscription** - [Subscribe here](https://claude.ai/settings/subscription)

2. **Claude CLI** installed and authenticated:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude login
   ```

3. **Bun** runtime:
   ```bash
   curl -fsSL https://bun.sh/install | bash
   ```

## Installation

```bash
git clone https://github.com/franlerma/claude-subscription-proxy
cd claude-subscription-proxy
bun install
```

## Usage

### Start the Proxy

```bash
bun run proxy
```

### Run OpenCode

```bash
ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=http://127.0.0.1:3456 opencode
```

Select any `anthropic/claude-*` model (opus, sonnet, haiku).

### One-liner

```bash
bun run proxy & ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=http://127.0.0.1:3456 opencode
```

## Auto-start on macOS

Set up the proxy to run automatically on login:

```bash
cat > ~/Library/LaunchAgents/com.claude-max-proxy.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.claude-max-proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which bun)</string>
        <string>run</string>
        <string>proxy</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.claude-max-proxy.plist
```

Then add an alias to `~/.zshrc`:

```bash
echo "alias oc='ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=http://127.0.0.1:3456 opencode'" >> ~/.zshrc
source ~/.zshrc
```

Now just run `oc` to start OpenCode with Claude Max.

## Auto-start on Linux (systemd)

Example systemd unit (adjust `User`, `Environment` (`PATH`), `WorkingDirectory`, and `ExecStart` to match your host):

```ini
[Unit]
Description=Claude Code Proxy
After=network.target

[Service]
Type=simple
User=root
Environment=PATH=/root/.asdf/shims:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WorkingDirectory=/opt/claude-subscription-proxy
ExecStart=/root/.asdf/shims/bun run proxy
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Save it to `/etc/systemd/system/claude-subscription-proxy.service`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now claude-subscription-proxy
```

Check status and logs:

```bash
systemctl status claude-subscription-proxy
journalctl -u claude-subscription-proxy -f
```

## Model Mapping

| OpenCode Model | Claude SDK |
|----------------|------------|
| `anthropic/claude-opus-*` | opus |
| `anthropic/claude-sonnet-*` | sonnet |
| `anthropic/claude-haiku-*` | haiku |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CLAUDE_PROXY_PORT` | 3456 | Proxy server port |
| `CLAUDE_PROXY_HOST` | 127.0.0.1 | Proxy server host |
| `CLAUDE_PROXY_WARM_POOL` | 3 | Max number of pre-started Claude CLI processes kept ready, one per conversation (model + max tokens + tool set + caller session). Each warm process resumes that conversation's transcript, serves exactly one request, and is replaced. Set to the expected number of concurrent conversations, or `0` to disable. |
| `CLAUDE_PROXY_WARM_TTL_MS` | 60000 | Idle time before an unused warm process is killed |
| `CLAUDE_PROXY_SESSION` | 1 (enabled) | Resume the CLI session for each caller session so later turns send only the new message instead of the whole history. Uses the caller's session header (`x-opencode-session-id`, `x-session-id`, etc.) or a fingerprint of the leading messages. Set to `0` to send the full history every time. |
| `CLAUDE_PROXY_SESSION_TTL_MS` | 1800000 | Idle time before a session mapping is forgotten and its session file, transcript and scratch data are deleted |
| `CLAUDE_PROXY_SESSION_MAX` | 256 | Max tracked sessions; the least recently used one is evicted |
| `CLAUDE_PROXY_SESSION_FLUSH_MS` | 750 | Time granted to the CLI to write its transcript before the process is torn down |
| `CLAUDE_PROXY_CONFIG_DIR` | `~/.claude-proxy` | Where resumed transcripts and the per-session map (`sessions/<sha256>.json`) live. Kept separate from `~/.claude` so TTL cleanup never touches a real Claude Code install; credentials are symlinked from there on startup. |
| `CLAUDE_PROXY_TOOL_CHOICE_FILTER` | 0 (off) | When `1`, a `tool_choice: {type:"tool"}` request only sends that tool's schema. Saves schema tokens on the forced turn, but changes the cached tool prefix, so it is opt-in. |
| `CLAUDE_PROXY_SYSTEM_PROMPT_FREEZE` | 1 (on) | Keep the first caller system prompt in the cached system block and deliver later changes as a conversation message (Claude Code's own resume behavior). Set to `0` to send the live system prompt every turn. |
| `CLAUDE_PROXY_AMBIENT_ENV` | 0 (off) | When `1`, keeps ambient Anthropic/CLI environment variables instead of stripping them. Debug escape hatch: ambient `ANTHROPIC_API_KEY` silently switches to API-key billing, `CLAUDE_CODE_FORCE_GLOBAL_CACHE` switches to the global cache, and `MAX_THINKING_TOKENS` turns thinking on for every turn. |


The first request of each conversation is a miss (full startup); later turns of the same conversation can hit. Idle warm processes use memory, so keep `CLAUDE_PROXY_WARM_POOL` small. Set it to the number of conversations you expect to run concurrently; extra shapes (for example a title-generation call with a different max-token cap) evict the least recently used conversation.

### Session reuse

When `CLAUDE_PROXY_SESSION` is enabled and the caller sends a session header, the proxy remembers the CLI session id for that conversation. The first turn sends the full history and persists a transcript; later turns send only the newest message and pass `resume`, so the CLI replays its own transcript instead of the proxy re-sending everything. That is what keeps token usage flat as a conversation grows.

Reuse falls back to the full history whenever it cannot be trusted: the conversation is new, compaction/regeneration shortened or rewrote the history, the assistant message the caller replays differs from the one the transcript holds, or the remembered transcript has disappeared from disk.

Each mapping is written atomically to its own file under `CLAUDE_PROXY_CONFIG_DIR/sessions/` (one `sha256(callerKey).json` per conversation), so a proxy restart resumes the same transcripts instead of paying a cold cache write for every conversation. One session's write or corruption cannot take down the others, and expiry deletes exactly one file. The old single `sessions.json` is migrated automatically on first start (kept as `sessions.json.migrated` until the TTL sweep removes it). On restart the system prompt is re-frozen from the first resumed request.

The background sweep also deletes idle files under `projects/`, `debug/`, `todos/`, `shell-snapshots/` and `statsig/`, and keeps only the five newest `.claude.json.backup.*` files, inside the proxy's isolated config dir. It never touches `.claude.json`, credentials or `plugins/`.

When the caller's system prompt changes mid-conversation, the proxy keeps the original prompt in Claude Code's cached system block and appends the new one as a `<caller_system_prompt_update>` message. That mirrors how Claude Code handles resumed sessions and avoids invalidating the system prompt plus every tool definition ahead of it.

## How It Works

1. **OpenCode** sends a request to `http://127.0.0.1:3456/messages` (thinking it's the Anthropic API)
2. **Proxy** receives the request and extracts the messages
3. **Proxy** calls `query()` from the Claude Agent SDK with your prompt
4. **Claude Agent SDK** authenticates using your Claude CLI login (tied to your Max subscription)
5. **Claude** processes the request using your subscription
6. **Proxy** streams the response back in Anthropic SSE format
7. **OpenCode** receives the response as if it came from the real API

The proxy is plain TypeScript. No magic, no hacks.

## FAQ

### Why do I need `ANTHROPIC_API_KEY=dummy`?

OpenCode requires an API key to be set, but we never actually use it. The Claude Agent SDK handles authentication through your Claude CLI login. Any non-empty string works.

### Does this work with other tools besides OpenCode?

Yes! Any tool that uses the Anthropic API format can use this proxy. Just point `ANTHROPIC_BASE_URL` to `http://127.0.0.1:3456`.

### What about rate limits?

Your Claude Max subscription has its own usage limits. This proxy doesn't add any additional limits.

### Is my data sent anywhere else?

No. The proxy runs locally on your machine. Your requests go directly to Claude through the official SDK.

## Troubleshooting

### "Authentication failed"

Run `claude login` to authenticate with the Claude CLI.

### "Connection refused"

Make sure the proxy is running: `bun run proxy`

### Proxy keeps dying

Use the launchd service (see Auto-start section) which automatically restarts the proxy.

## License

MIT

## Credits

This project is a fork of [opencode-claude-max-proxy](https://github.com/rynfar/opencode-claude-max-proxy) by [rynfar](https://github.com/rynfar), which served as the foundation and inspiration for this project.

Built with the [Claude Agent SDK](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk) by Anthropic.
