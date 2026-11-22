import { join } from "path"
import { homedir } from "os"
import { mkdirSync } from "fs"
import { claudeExecutable } from "./models"
import { CLIENT_MCP_SERVER, type ClientTools } from "./tools"
import { SYSTEM_PROMPT_DYNAMIC_BOUNDARY, callerSystemBlock, renderBlocks } from "./prompt"
import { claudeLog } from "../logger"

export type Env = Record<string, string | undefined>

export function buildOptions(
  model: string,
  env: Env,
  client: ClientTools | null,
  abort: AbortController,
  persist = false,
  callerSystem = "",
  stderrBuf?: string[],
  maxThinkingTokens?: number
): Record<string, any> {
  const options: Record<string, any> = {
    model,
    pathToClaudeCodeExecutable: claudeExecutable,
    cwd: PROXY_CWD,
    // Keep Claude Code's own system prompt and append the caller's prompt after the dynamic boundary.
    // Appending the caller text directly (without the boundary) re-classifies the request as
    // third-party, which bills "extra usage" instead of the subscription.
    systemPrompt: callerSystem
      ? {
          type: "preset" as const,
          preset: "claude_code" as const,
          append: `${SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n${callerSystemBlock(callerSystem)}`
        }
      : { type: "preset" as const, preset: "claude_code" as const },
    tools: [], // no built-in Claude Code tools: nothing runs on this host
    // SDK isolation: do not load the host's CLAUDE.md, settings.json or MCP servers. Those would
    // add tokens to the system prompt and could change between turns, invalidating the cache.
    settingSources: [],
    maxTurns: 1,
    includePartialMessages: true,
    persistSession: persist,
    abortController: abort,
    env,
    stderr: (data: string) => {
      if (stderrBuf) stderrBuf.push(data)
      claudeLog("claude.stderr", { data: data.slice(0, 500) })
    }
  }
  if (client) {
    options.mcpServers = { [CLIENT_MCP_SERVER]: client.server }
    options.allowedTools = client.allowed
  }
  // The caller's `thinking` budget is forwarded when present. Leaving it undefined keeps the CLI's
  // own default, which the ambient-environment hardening already protects from MAX_THINKING_TOKENS.
  if (maxThinkingTokens !== undefined) options.maxThinkingTokens = maxThinkingTokens
  return options
}

// ---------------------------------------------------------------------------------------
// Warm CLI pool (opt-in)
//
// Starting the CLI is more than half of the time-to-first-byte. Up to
// CLAUDE_PROXY_WARM_POOL (N > 0, default 3) CLI processes are kept started and idle, waiting for
// their prompt, one per conversation (session key) within a given "shape" (model + max output
// tokens + exact tool set). A request takes its own conversation's warm process and the pool
// immediately starts a replacement carrying the same `resume` id, so the next turn lands on a live
// transcript instead of a blank one. When the pool is full, the least recently used conversation is
// dropped. A warm process serves exactly one request and is then killed.
//   CLAUDE_PROXY_WARM_POOL    0 disables the pool (default 3); set it to the expected number of
//                             concurrent conversations, not a fixed buffer
//   CLAUDE_PROXY_WARM_TTL_MS  idle time before an unused warm process is killed (default 60000)
// ---------------------------------------------------------------------------------------

const envInt = (name: string, fallback: number) => {
  const n = Number(process.env[name])
  return Number.isFinite(n) && n >= 0 && process.env[name] !== "" && process.env[name] !== undefined ? Math.floor(n) : fallback
}
export const WARM_POOL_SIZE = envInt("CLAUDE_PROXY_WARM_POOL", 3)
// Shorter than it was: a warm process now holds a live transcript, and reusing a stale one across
// writers is worse than a cold start.
export const WARM_POOL_TTL_MS = Math.max(1000, envInt("CLAUDE_PROXY_WARM_TTL_MS", 60_000))

// ---------------------------------------------------------------------------------------
// Session continuity (resume)
//
// Sending the whole conversation on every turn is what makes tokens grow. The CLI can resume a
// previous session, so once a conversation has a Claude session id, later turns only send the
// newest user message and let the CLI replay its own persisted transcript. Sessions are keyed by
// the caller's session id (opencode sends `x-opencode-session-id`; litellm forwards it since it is
// an `x-` header). The mapping lives in memory and expires after CLAUDE_PROXY_SESSION_TTL_MS.
// Transcripts are kept in the proxy's own config dir so TTL cleanup never touches a real Claude
// Code installation.
//   CLAUDE_PROXY_SESSION           1 enables resume (default 1); 0 disables it
//   CLAUDE_PROXY_SESSION_TTL_MS    idle time before a mapping is forgotten (default 3600000)
//   CLAUDE_PROXY_SESSION_MAX       max tracked sessions, least recently used evicted (default 256)
//   CLAUDE_PROXY_CONFIG_DIR        where transcripts live (default <home>/.claude-proxy)
// ---------------------------------------------------------------------------------------

export const SESSION_MODE = envInt("CLAUDE_PROXY_SESSION", 1) > 0
export const SESSION_TTL_MS = Math.max(1000, envInt("CLAUDE_PROXY_SESSION_TTL_MS", 3_600_000))
export const SESSION_MAX = Math.max(1, envInt("CLAUDE_PROXY_SESSION_MAX", 256))
// Time granted to the CLI to flush its transcript before its process is torn down.
export const SESSION_FLUSH_MS = Math.max(0, envInt("CLAUDE_PROXY_SESSION_FLUSH_MS", 750))
export const SESSION_CONFIG_DIR =
  process.env.CLAUDE_PROXY_CONFIG_DIR?.trim() ||
  join(homedir(), ".claude-proxy")
// Neutral cwd for the CLI. It is what the preset prints in <env>, so it must not leak the proxy's
// install dir to remote callers. It must stay constant: transcripts live under projects/<cwd>, so
// changing it breaks resume.
export const PROXY_CWD = join(SESSION_CONFIG_DIR, "proxy-sandbox")
mkdirSync(PROXY_CWD, { recursive: true })
// How many leading messages define a conversation identity when the caller sends no session header.
// Keep this at 1: a conversation's first request often has only one message, so hashing two would
// change the key on turn two (the second message appears) and silently start a new transcript.
export const SESSION_FINGERPRINT_MESSAGES = Math.max(1, envInt("CLAUDE_PROXY_SESSION_FINGERPRINT", 1))

// ---------------------------------------------------------------------------------------
// History compaction on model switch
//
// The prompt cache is per model. When a conversation arrives for a model with no Claude transcript
// to resume (a model switch, e.g. the caller was talking to Deepseek and now asks Claude), the whole
// history is re-sent and billed as a cold cache write with no chance of ever being read back. That
// is the one case where the history is worth compacting: summarise the old tail with a dedicated
// haiku process and keep only the newest messages verbatim. In the same model, resuming the live
// transcript already reads the stable prefix from cache, so history is never compacted there.
//   CLAUDE_PROXY_COMPACT_HISTORY_TOKENS  char/4 estimate over which a cold history is summarised
//                                        (default 40000; set 0 to disable)
//   CLAUDE_PROXY_COMPACT_KEEP_MESSAGES   newest messages kept verbatim after the summary (default 4)
// ---------------------------------------------------------------------------------------

const DEFAULT_COMPACT_HISTORY_TOKENS = 40_000

export const COMPACT_HISTORY_TOKENS = (() => {
  const raw = process.env.CLAUDE_PROXY_COMPACT_HISTORY_TOKENS
  if (raw === undefined || raw === "") return DEFAULT_COMPACT_HISTORY_TOKENS
  const n = Number(raw)
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : DEFAULT_COMPACT_HISTORY_TOKENS
})()
export const COMPACT_KEEP_MESSAGES = Math.max(1, envInt("CLAUDE_PROXY_COMPACT_KEEP_MESSAGES", 4))

// ---------------------------------------------------------------------------------------
// Helper requests
//
// Title generation and similar helpers arrive as a single message with no tools. They never need
// the conversation's expensive model, so they run on a cheap one instead of consuming the main
// model's subscription quota for one line of text.
//   CLAUDE_PROXY_HELPER_MODEL  CLI alias or exact id for helper requests (default haiku)
// ---------------------------------------------------------------------------------------

export const HELPER_MODEL = process.env.CLAUDE_PROXY_HELPER_MODEL?.trim() || "haiku"

/** Rough char/4 token estimate over the whole conversation, for deciding whether to compact. */
export function estimateHistoryTokens(messages: any[]): number {
  let chars = 0
  for (const m of messages) {
    if (!m) continue
    chars += 4 + (typeof m.role === "string" ? m.role.length : 0) + renderBlocks(m.content, new Map()).length
  }
  return Math.ceil(chars / 4)
}

// ---------------------------------------------------------------------------------------
// Environment hardening
//
// The proxy process often inherits the caller's shell environment. A stray ANTHROPIC_API_KEY
// silently switches the CLI from OAuth to API-key auth (third-party billing), a stray
// CLAUDE_CODE_FORCE_GLOBAL_CACHE turns on the global cache scope (also third-party), and a stray
// MAX_THINKING_TOKENS makes every turn think with a large budget. Remove the dangerous ones so
// token usage and billing stay determined by the request, not by the host's shell.
//   CLAUDE_PROXY_AMBIENT_ENV=1  keep the ambient values (advanced/debugging escape hatch)
// ---------------------------------------------------------------------------------------

const AMBIENT_ENV = envInt("CLAUDE_PROXY_AMBIENT_ENV", 0) > 0
const STRIPPED_ENV = [
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_AUTH_TOKEN",
  "ANTHROPIC_BASE_URL",
  "CLAUDE_CODE_FORCE_GLOBAL_CACHE",
  "CLAUDE_CODE_EXTRA_BODY",
  "DISABLE_PROMPT_CACHING",
  "DISABLE_PROMPT_CACHING_HAIKU",
  "DISABLE_PROMPT_CACHING_SONNET",
  "DISABLE_PROMPT_CACHING_OPUS",
  "MAX_THINKING_TOKENS"
]

// Claude Code auto-compacts its transcript when the context fills. The proxy runs one turn per CLI
// process (`maxTurns: 1`) and aborts it afterwards, so compaction can only happen when the CLI loads
// a resumed transcript that is already past its threshold. Compacting rewrites the transcript and
// invalidates the cached prefix the next turn would otherwise read, so it must be a last resort, not
// a routine event: the old 40% default fired while growth was still cheap. Keep a safety margin for
// the context limit but let a healthy transcript grow.
//   CLAUDE_PROXY_AUTOCOMPACT_PCT  % of context at which the CLI compacts on resume (default 90)
const AUTOCOMPACT_PCT = Math.max(1, Math.min(100, envInt("CLAUDE_PROXY_AUTOCOMPACT_PCT", 90)))

export function buildCliEnv(): Env {
  const env: Env = { ...process.env }
  if (!AMBIENT_ENV) {
    for (const key of STRIPPED_ENV) delete env[key]
  }
  // Telemetry, auto-update and feedback loops are not part of a subscription turn.
  if (env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC === undefined) {
    env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
  }
  // Deterministic auto-compact threshold, so the proxy controls transcript size instead of the host.
  if (env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE === undefined) {
    env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = String(AUTOCOMPACT_PCT)
  }
  // The CLI retries a 429 up to 10 times on a subscription, which only burns time (and, on a spent
  // plan, can never succeed). The proxy owns the retry decision, so cap it here and let a 429 surface
  // to the caller immediately. Raise CLAUDE_PROXY_MAX_RETRIES to allow a few transient retries.
  if (env.CLAUDE_CODE_MAX_RETRIES === undefined) {
    env.CLAUDE_CODE_MAX_RETRIES = String(envInt("CLAUDE_PROXY_MAX_RETRIES", 0))
  }
  return env
}
