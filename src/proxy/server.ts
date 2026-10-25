import { Hono } from "hono"
import { cors } from "hono/cors"
import { query, createSdkMcpServer, tool } from "@anthropic-ai/claude-agent-sdk"
import type { Query, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk"
import { createHash, randomUUID } from "crypto"
import { z } from "zod"
import type { Context } from "hono"
import type { ProxyConfig } from "./types"
import { DEFAULT_PROXY_CONFIG } from "./types"
import { claudeLog } from "../logger"
import { execSync } from "child_process"
import { existsSync } from "fs"
import { fileURLToPath } from "url"
import { join, dirname } from "path"
import { homedir } from "os"
import { lstatSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, rmdirSync, statSync, symlinkSync, writeFileSync } from "fs"

/**
 * Anthropic Messages API facade on top of the Claude Code CLI (Claude Agent SDK).
 *
 * Design:
 *  - The caller (litellm -> opencode) owns tool execution. Its `tools` are exposed to the
 *    model as inert MCP tools. When the model calls one, the `tool_use` block is forwarded
 *    untouched, the turn ends with `stop_reason: "tool_use"` and the CLI is aborted before
 *    anything runs. The next HTTP request carries the `tool_result`.
 *  - One HTTP request == exactly one assistant message (a single message_start/message_stop).
 *  - Nothing is ever executed on the proxy host.
 */

type StreamEvent = { type: string; [key: string]: any }

/** Error that carries the HTTP status / Anthropic error type to return to the caller. */
class ProxyError extends Error {
  constructor(
    message: string,
    readonly status: number = 500,
    readonly errorType: string = "api_error"
  ) {
    super(message)
  }
}

/** Turns the CLI's "API Error: 400 {json}" text into a ProxyError with the upstream status and message. */
function upstreamError(status: number, rest: string): ProxyError {
  try {
    const parsed = JSON.parse(rest)
    const message = parsed?.error?.message ?? rest
    return new ProxyError(String(message), status, parsed?.error?.type ?? "api_error")
  } catch {
    return new ProxyError(rest || `Upstream error ${status}`, status)
  }
}

const CLIENT_MCP_SERVER = "client"
const CLIENT_TOOL_PREFIX = `mcp__${CLIENT_MCP_SERVER}__`
const DEFAULT_MAX_OUTPUT_TOKENS = 32_000

function resolveClaudeExecutable(): string {
  // 1. Try the SDK's bundled cli.js (same dir as this module's SDK)
  try {
    const sdkPath = fileURLToPath(import.meta.resolve("@anthropic-ai/claude-agent-sdk"))
    const sdkCliJs = join(dirname(sdkPath), "cli.js")
    if (existsSync(sdkCliJs)) return sdkCliJs
  } catch {}

  // 2. Try the system-installed claude binary
  try {
    const claudePath = execSync("which claude", { encoding: "utf-8" }).trim()
    if (claudePath && existsSync(claudePath)) return claudePath
  } catch {}

  throw new Error("Could not find Claude Code executable. Install via: npm install -g @anthropic-ai/claude-code")
}

const claudeExecutable = resolveClaudeExecutable()

// ---------------------------------------------------------------------------------------
// Model mapping
// ---------------------------------------------------------------------------------------

function aliasForModel(model: string): "sonnet" | "opus" | "haiku" {
  if (model.includes("opus")) return "opus"
  if (model.includes("haiku")) return "haiku"
  return "sonnet"
}

/** Exact `claude-*` ids are passed through; anything else falls back to the CLI alias. */
function resolveModel(model: unknown, forceAlias = false): string {
  const name = typeof model === "string" ? model : ""
  if (!name) return "sonnet"
  if (!forceAlias && process.env.CLAUDE_PROXY_MODEL_PASSTHROUGH !== "0" && /^claude-[a-z0-9._-]+$/i.test(name)) {
    return name
  }
  return aliasForModel(name)
}

// ---------------------------------------------------------------------------------------
// Client tools -> inert MCP tools
// ---------------------------------------------------------------------------------------

function jsonSchemaToZod(schema: any, depth = 0): z.ZodTypeAny {
  if (!schema || typeof schema !== "object" || depth > 8) return z.any()
  try {
    let t: z.ZodTypeAny

    if (Array.isArray(schema.enum) && schema.enum.length > 0) {
      if (schema.enum.every((v: unknown) => typeof v === "string")) {
        t = z.enum(schema.enum as [string, ...string[]])
      } else {
        const literals = schema.enum.map((v: any) => z.literal(v))
        t = literals.length === 1 ? literals[0] : z.union(literals as any)
      }
    } else if ("const" in schema) {
      t = z.literal(schema.const)
    } else if (Array.isArray(schema.anyOf) || Array.isArray(schema.oneOf)) {
      const options = (schema.anyOf ?? schema.oneOf).map((o: any) => jsonSchemaToZod(o, depth + 1))
      t = options.length === 0 ? z.any() : options.length === 1 ? options[0] : z.union(options as any)
    } else {
      const types: string[] = (Array.isArray(schema.type) ? schema.type : [schema.type]).filter(Boolean)
      const nullable = types.includes("null")
      const main = types.filter((x) => x !== "null")
      const fromType = (type: string): z.ZodTypeAny => {
        switch (type) {
          case "string":
            return z.string()
          case "number":
            return z.number()
          case "integer":
            return z.number().int()
          case "boolean":
            return z.boolean()
          case "array":
            return z.array(schema.items ? jsonSchemaToZod(schema.items, depth + 1) : z.any())
          case "object":
            return schema.properties
              ? z.object(schemaToShape(schema, depth + 1)).passthrough()
              : z.record(z.string(), z.any())
          default:
            return z.any()
        }
      }
      t = main.length === 0 ? z.any() : main.length === 1 ? fromType(main[0]!) : z.union(main.map(fromType) as any)
      if (nullable) t = t.nullable()
    }

    if (typeof schema.description === "string" && schema.description) t = t.describe(schema.description)
    return t
  } catch {
    return z.any()
  }
}

function schemaToShape(schema: any, depth = 0): Record<string, z.ZodTypeAny> {
  const props = schema?.properties && typeof schema.properties === "object" ? schema.properties : {}
  const required = new Set<string>(Array.isArray(schema?.required) ? schema.required : [])
  const shape: Record<string, z.ZodTypeAny> = {}
  for (const [key, value] of Object.entries(props)) {
    const t = jsonSchemaToZod(value, depth)
    shape[key] = required.has(key) ? t : t.optional()
  }
  return shape
}

interface ClientTools {
  server: ReturnType<typeof createSdkMcpServer>
  allowed: string[]
  nameToAlias: Map<string, string>
  aliasToName: Map<string, string>
}

/**
 * Dedupe by name, then sort by code point. The tool list is the first part of the API request:
 * reordering it invalidates the entire prompt cache even though the definitions are unchanged.
 */
function buildClientTools(tools: Array<{ name: string; description?: string; input_schema?: any }>): ClientTools {
  const nameToAlias = new Map<string, string>()
  const aliasToName = new Map<string, string>()
  const maxLen = 64 - CLIENT_TOOL_PREFIX.length

  const sdkTools = tools
    .filter((t) => t && typeof t.name === "string" && t.name)
    .map((t, i) => {
      let alias = t.name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, maxLen)
      if (aliasToName.has(alias)) alias = `${alias.slice(0, maxLen - 4)}_${i}`
      nameToAlias.set(t.name, alias)
      aliasToName.set(alias, t.name)
      return tool(
        alias,
        t.description || t.name,
        schemaToShape(t.input_schema) as any,
        // Never meant to run: the turn is aborted as soon as the model emits the tool_use.
        async () => ({
          content: [{ type: "text" as const, text: "This tool is executed by the client, not by the proxy." }],
          isError: true
        })
      )
    })

  const server = createSdkMcpServer({ name: CLIENT_MCP_SERVER, version: "1.0.0", tools: sdkTools })
  return { server, allowed: [...aliasToName.keys()].map((a) => `${CLIENT_TOOL_PREFIX}${a}`), nameToAlias, aliasToName }
}

function originalToolName(name: string, client: ClientTools | null): string {
  if (!name.startsWith(CLIENT_TOOL_PREFIX)) return name
  const alias = name.slice(CLIENT_TOOL_PREFIX.length)
  return client?.aliasToName.get(alias) ?? alias
}

// ---------------------------------------------------------------------------------------
// Request -> prompt
// ---------------------------------------------------------------------------------------

function systemText(system: unknown): string {
  if (typeof system === "string") return system
  if (Array.isArray(system)) {
    return system
      .filter((b: any) => b?.type === "text" && b.text)
      .map((b: any) => b.text)
      .join("\n\n")
  }
  return ""
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content
  if (Array.isArray(content)) {
    return content
      .map((b: any) => {
        if (b?.type === "text") return b.text ?? ""
        if (b?.type === "image") return "[image omitted]"
        return ""
      })
      .join("\n")
  }
  return content == null ? "" : JSON.stringify(content)
}

function renderBlocks(content: unknown, nameToAlias: Map<string, string>): string {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return content == null ? "" : String(content)

  const parts: string[] = []
  for (const block of content as any[]) {
    switch (block?.type) {
      case "text":
        if (block.text) parts.push(block.text)
        break
      case "tool_use": {
        const alias = nameToAlias.get(block.name) ?? block.name
        parts.push(
          `<tool_call id="${block.id}" name="${CLIENT_TOOL_PREFIX}${alias}">${JSON.stringify(block.input ?? {})}</tool_call>`
        )
        break
      }
      case "tool_result": {
        const err = block.is_error ? ` is_error="true"` : ""
        parts.push(`<tool_result id="${block.tool_use_id}"${err}>\n${toolResultText(block.content)}\n</tool_result>`)
        break
      }
      case "image":
        parts.push("[image omitted]")
        break
      default:
        // thinking / redacted_thinking / unknown blocks are not replayed
        break
    }
  }
  return parts.join("\n")
}

/**
 * The caller's system prompt is appended in plain text, after Claude Code's dynamic boundary.
 * Base64 was inflating it by a third and, worse, made it a huge unbroken token blob that could
 * never share a cache prefix with anything.
 */
/**
 * Marker Claude Code uses to split its system prompt into a static, cacheable half and a dynamic
 * half. Everything before it gets `cache_control`; everything after stays uncached. Injecting the
 * caller's prompt after the marker keeps Claude Code's own prefix cacheable while the caller's
 * (which changes per turn) does not invalidate it. It must stay: appending the caller text directly
 * onto the first-party preset makes Anthropic classify the request as a third-party app and bill it
 * against "extra usage" instead of the plan.
 */
const SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

function callerSystemBlock(system: string): string {
  return ["<caller_system_prompt>", system, "</caller_system_prompt>"].join("\n")
}

function buildPrompt(messages: any[], nameToAlias: Map<string, string>): string {
  const body =
    messages.length === 1 && messages[0]?.role === "user"
      ? renderBlocks(messages[0].content, nameToAlias)
      : [
          "Below is the conversation so far. Write the next assistant turn.",
          "If you need a tool, call it through the normal tool-calling mechanism. Never type <tool_call> or <tool_result> tags yourself: they only appear in this history.",
          "",
          "<conversation>",
          messages
            .map((m) => {
              const role = m?.role === "assistant" ? "assistant" : "user"
              return `<${role}>\n${renderBlocks(m?.content, nameToAlias)}\n</${role}>`
            })
            .join("\n"),
          "</conversation>"
        ].join("\n")

  return body
}

/**
 * Prompt for a resumed session: only what came after the last assistant turn. That is the newest
 * user message, or the tool results that answer the assistant's `tool_use` blocks. Everything
 * before it already lives in the CLI's own transcript.
 */
function buildSessionPrompt(messages: any[], nameToAlias: Map<string, string>): string {
  let start = 0
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "assistant") {
      start = i + 1
      break
    }
  }
  const fresh = messages.slice(start).filter((m: any) => m?.role !== "assistant")
  if (fresh.length === 0) {
    return renderBlocks(messages[messages.length - 1]?.content, nameToAlias)
  }
  // A lone user turn must render exactly like `buildPrompt` does for the opening turn: bare content,
  // no role wrapper. Wrapping it here made the first resumed request send a different prefix than
  // the one the opening turn had just cached, so the prompt cache was thrown away once per
  // conversation - `cache_read` 0 on turn two, recovering only on turn three.
  if (fresh.length === 1 && fresh[0]?.role === "user") {
    return renderBlocks(fresh[0].content, nameToAlias)
  }
  return fresh
    .map((m) => {
      const role = m?.role === "assistant" ? "assistant" : "user"
      return `<${role}>\n${renderBlocks(m?.content, nameToAlias)}\n</${role}>`
    })
    .join("\n")
}

// ---------------------------------------------------------------------------------------
// Running one assistant turn
// ---------------------------------------------------------------------------------------

function* eventsFromAssistantMessage(m: any, client: ClientTools | null): Generator<StreamEvent> {
  yield {
    type: "message_start",
    message: {
      id: m.id ?? `msg_${Date.now()}`,
      type: "message",
      role: "assistant",
      model: m.model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: m.usage?.input_tokens ?? 0, output_tokens: 0 }
    }
  }
  const content: any[] = Array.isArray(m.content) ? m.content : []
  for (let index = 0; index < content.length; index++) {
    const block = content[index]
    if (block.type === "text") {
      yield { type: "content_block_start", index, content_block: { type: "text", text: "" } }
      yield { type: "content_block_delta", index, delta: { type: "text_delta", text: block.text ?? "" } }
      yield { type: "content_block_stop", index }
    } else if (block.type === "tool_use") {
      yield {
        type: "content_block_start",
        index,
        content_block: { type: "tool_use", id: block.id, name: originalToolName(block.name, client), input: {} }
      }
      yield {
        type: "content_block_delta",
        index,
        delta: { type: "input_json_delta", partial_json: JSON.stringify(block.input ?? {}) }
      }
      yield { type: "content_block_stop", index }
    }
  }
  yield {
    type: "message_delta",
    delta: { stop_reason: m.stop_reason ?? "end_turn", stop_sequence: m.stop_sequence ?? null },
    usage: { output_tokens: m.usage?.output_tokens ?? 0 }
  }
  yield { type: "message_stop" }
}

type Env = Record<string, string | undefined>

function buildOptions(
  model: string,
  env: Env,
  client: ClientTools | null,
  abort: AbortController,
  persist = false,
  callerSystem = ""
): Record<string, any> {
  const options: Record<string, any> = {
    model,
    pathToClaudeCodeExecutable: claudeExecutable,
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
    stderr: (data: string) => claudeLog("claude.stderr", { data: data.slice(0, 500) })
  }
  if (client) {
    options.mcpServers = { [CLIENT_MCP_SERVER]: client.server }
    options.allowedTools = client.allowed
  }
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
const WARM_POOL_SIZE = envInt("CLAUDE_PROXY_WARM_POOL", 3)
// Shorter than it was: a warm process now holds a live transcript, and reusing a stale one across
// writers is worse than a cold start.
const WARM_POOL_TTL_MS = Math.max(1000, envInt("CLAUDE_PROXY_WARM_TTL_MS", 60_000))

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

const SESSION_MODE = envInt("CLAUDE_PROXY_SESSION", 1) > 0
const SESSION_TTL_MS = Math.max(1000, envInt("CLAUDE_PROXY_SESSION_TTL_MS", 3_600_000))
const SESSION_MAX = Math.max(1, envInt("CLAUDE_PROXY_SESSION_MAX", 256))
// Time granted to the CLI to flush its transcript before its process is torn down.
const SESSION_FLUSH_MS = Math.max(0, envInt("CLAUDE_PROXY_SESSION_FLUSH_MS", 750))
const SESSION_CONFIG_DIR =
  process.env.CLAUDE_PROXY_CONFIG_DIR?.trim() ||
  join(homedir(), ".claude-proxy")
// How many leading messages define a conversation identity when the caller sends no session header.
// Keep this at 1: a conversation's first request often has only one message, so hashing two would
// change the key on turn two (the second message appears) and silently start a new transcript.
const SESSION_FINGERPRINT_MESSAGES = Math.max(1, envInt("CLAUDE_PROXY_SESSION_FINGERPRINT", 1))

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

function buildCliEnv(): Env {
  const env: Env = { ...process.env }
  if (!AMBIENT_ENV) {
    for (const key of STRIPPED_ENV) delete env[key]
  }
  // Telemetry, auto-update and feedback loops are not part of a subscription turn.
  if (env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC === undefined) {
    env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
  }
  return env
}

/**
 * Everything the proxy knows about one caller conversation. `messageCount` describes where the
 * assistant turn the transcript holds should sit in the caller's history on the next request, which
 * is what lets a later request prove that resuming is safe.
 */
interface SessionState {
  id: string
  messageCount: number
  updatedAt: number
  /** In-memory TTL timer; never persisted. */
  timer?: ReturnType<typeof setTimeout>
}

/** Caller session key -> conversation state. Map order is recency order, oldest first. */
const sessions = new Map<string, SessionState>()
const SESSION_DIR = join(SESSION_CONFIG_DIR, "sessions")
const LEGACY_SESSION_FILE = join(SESSION_CONFIG_DIR, "sessions.json")
const SESSION_TMP_TTL_MS = 5 * 60_000
let sessionsLoaded = false
/**
 * A caller can send the next turn as soon as it receives the assistant message, while this turn is
 * still draining the CLI for its `result` (which carries the session id and flushes the transcript).
 * Without this gate the next request finds no state and replays the whole history. Wait for the
 * previous turn of the same conversation to remember its state first.
 */
const pendingTurns = new Map<string, Promise<void>>()
const PENDING_TURN_TIMEOUT_MS = Math.max(60_000, SESSION_FLUSH_MS + 60_000)

/**
 * Stable identity for a conversation when the caller sends no session header.
 * Hashes the leading messages only: they never change as the conversation grows, which is what makes
 * the same conversation resolve to the same key on every turn. The system prompt is included because
 * two callers with different prompts must never share a transcript.
 */
function conversationFingerprint(messages: any[], system: string): string | null {
  if (!Array.isArray(messages) || messages.length === 0) return null
  const head = messages.slice(0, SESSION_FINGERPRINT_MESSAGES).map((m) => [
    m?.role === "assistant" ? "assistant" : "user",
    renderBlocks(m?.content, new Map())
  ])
  return `fp:${createHash("sha1").update(JSON.stringify([head, system])).digest("hex")}`
}

/**
 * Session identity: the caller's own header when present, otherwise a fingerprint of the
 * conversation prefix. The header is preferred because it survives prompt edits that would change
 * the fingerprint mid-conversation.
 */
function sessionKey(messages: any[], system: string, c?: Context): string | null {
  if (!SESSION_MODE) return null
  const header = c
    ? (
        c.req.header("x-opencode-session-id") ||
        c.req.header("x-session-affinity") ||
        c.req.header("x-session-id") ||
        // litellm normalises a vendor session header into this one.
        c.req.header("x-litellm-session-id") ||
        ""
      ).trim()
    : ""
  return header || conversationFingerprint(messages, system)
}

/** One file per session: sessions/<sha256(callerKey)>.json. */
function sessionFilePath(key: string): string {
  const name = createHash("sha256").update(key).digest("hex")
  return join(SESSION_DIR, `${name}.json`)
}

function readSessionFile(path: string): { key: string; state: SessionState } | null {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"))
    if (!parsed || typeof parsed.key !== "string" || !parsed.key) return null
    if (typeof parsed.id !== "string" || !parsed.id) return null
    if (!Number.isFinite(parsed.updated)) return null
    return {
      key: parsed.key,
      state: {
        id: parsed.id,
        messageCount: Number.isFinite(parsed.messages) ? Math.max(0, Math.floor(parsed.messages)) : 0,
        updatedAt: parsed.updated
      }
    }
  } catch {
    return null
  }
}

/** Writes one session atomically. Returns false when the file could not be replaced. */
function writeSessionFile(key: string, state: SessionState): boolean {
  try {
    mkdirSync(SESSION_DIR, { recursive: true })
    const file = sessionFilePath(key)
    // A unique tmp name keeps two writers from clobbering each other's partial file.
    const tmp = `${file}.${randomUUID()}.tmp`
    writeFileSync(
      tmp,
      JSON.stringify({
        v: 1,
        key,
        id: state.id,
        messages: state.messageCount,
        updated: state.updatedAt
      }),
      { mode: 0o600 }
    )
    renameSync(tmp, file)
    return true
  } catch (error) {
    claudeLog("proxy.session.persist_failed", { error: error instanceof Error ? error.message : String(error) })
    return false
  }
}

/** One-time import of the old single-file map, then leave a .migrated marker. */
function migrateLegacySessions(cutoff: number) {
  if (!existsSync(LEGACY_SESSION_FILE)) return
  try {
    const parsed = JSON.parse(readFileSync(LEGACY_SESSION_FILE, "utf8"))
    const list: any[] = Array.isArray(parsed?.sessions) ? parsed.sessions : []
    let migrated = 0
    let complete = true
    for (const entry of list) {
      if (!entry || typeof entry.key !== "string" || !entry.key) continue
      if (typeof entry.id !== "string" || !entry.id) continue
      if (!Number.isFinite(entry.updated) || entry.updated < cutoff) continue
      const state: SessionState = {
        id: entry.id,
        messageCount: Number.isFinite(entry.messages) ? Math.max(0, Math.floor(entry.messages)) : 0,
        updatedAt: entry.updated
      }
      sessions.set(entry.key, state)
      if (writeSessionFile(entry.key, state)) migrated++
      else complete = false
    }
    if (complete) {
      renameSync(LEGACY_SESSION_FILE, `${LEGACY_SESSION_FILE}.migrated`)
      claudeLog("proxy.session.migrated", { count: migrated })
    } else {
      claudeLog("proxy.session.migration_incomplete", { migrated })
    }
  } catch (error) {
    claudeLog("proxy.session.migration_failed", { error: error instanceof Error ? error.message : String(error) })
  }
}

/** Loads every valid session file, dropping expired ones and stale tmp files. */
function loadSessionFiles(cutoff: number) {
  let names: string[]
  try {
    names = readdirSync(SESSION_DIR)
  } catch {
    return
  }
  const now = Date.now()
  for (const name of names) {
    const path = join(SESSION_DIR, name)
    try {
      if (name.endsWith(".tmp")) {
        if (statSync(path).mtimeMs < now - SESSION_TMP_TTL_MS) rmSync(path, { force: true })
        continue
      }
      if (!name.endsWith(".json")) continue
      const entry = readSessionFile(path)
      if (!entry) {
        // A corrupt file is only removed once it is old; a transient partial write is left alone.
        if (statSync(path).mtimeMs < cutoff) rmSync(path, { force: true })
        continue
      }
      if (entry.state.updatedAt < cutoff) {
        rmSync(path, { force: true })
        continue
      }
      const current = sessions.get(entry.key)
      if (current && current.updatedAt >= entry.state.updatedAt) continue
      sessions.set(entry.key, entry.state)
    } catch {}
  }
}

/** Removes expired/corrupt session files and abandoned tmp files. */
function pruneSessionFiles(cutoff: number) {
  let names: string[]
  try {
    names = readdirSync(SESSION_DIR)
  } catch {
    return
  }
  const now = Date.now()
  let removed = 0
  for (const name of names) {
    const path = join(SESSION_DIR, name)
    try {
      if (name.endsWith(".tmp")) {
        if (statSync(path).mtimeMs < now - SESSION_TMP_TTL_MS) {
          rmSync(path, { force: true })
          removed++
        }
        continue
      }
      if (!name.endsWith(".json")) continue
      const entry = readSessionFile(path)
      if (entry ? entry.state.updatedAt < cutoff : statSync(path).mtimeMs < cutoff) {
        rmSync(path, { force: true })
        removed++
      }
    } catch {}
  }
  try {
    const migrated = `${LEGACY_SESSION_FILE}.migrated`
    if (existsSync(migrated) && statSync(migrated).mtimeMs < cutoff) rmSync(migrated, { force: true })
  } catch {}
  if (removed > 0) claudeLog("proxy.session.files_swept", { removed })
}

/** Restores the key -> transcript map after a proxy restart. Transcripts outlive the process. */
function loadSessions() {
  if (sessionsLoaded || !SESSION_MODE) return
  sessionsLoaded = true
  try {
    mkdirSync(SESSION_DIR, { recursive: true })
  } catch {}
  const cutoff = Date.now() - SESSION_TTL_MS
  migrateLegacySessions(cutoff)
  loadSessionFiles(cutoff)
  while (sessions.size > SESSION_MAX) {
    const oldest = [...sessions.entries()].sort((a, b) => a[1].updatedAt - b[1].updatedAt)[0]
    if (!oldest) break
    deleteSession(oldest[0])
  }
  for (const key of sessions.keys()) scheduleSessionExpiry(key)
  if (sessions.size > 0) claudeLog("proxy.session.loaded", { count: sessions.size })
}

function ensureSessionsLoaded() {
  loadSessions()
}

function scheduleSessionExpiry(key: string) {
  const state = sessions.get(key)
  if (!state) return
  if (state.timer) clearTimeout(state.timer)
  const remaining = Math.max(1000, SESSION_TTL_MS - (Date.now() - state.updatedAt))
  const timer = setTimeout(() => {
    const current = sessions.get(key)
    if (!current) return
    if (Date.now() - current.updatedAt >= SESSION_TTL_MS) {
      deleteSession(key)
      claudeLog("proxy.session.expired", { key: key.slice(0, 12) })
    } else {
      scheduleSessionExpiry(key)
    }
  }, remaining)
  timer.unref?.()
  state.timer = timer
}

function deleteSession(key: string) {
  const state = sessions.get(key)
  if (state?.timer) clearTimeout(state.timer)
  sessions.delete(key)
  try {
    rmSync(sessionFilePath(key), { force: true })
  } catch {}
}

function rememberSession(key: string, state: SessionState) {
  if (!state.id) return
  const previous = sessions.get(key)
  if (previous?.timer) clearTimeout(previous.timer)
  sessions.set(key, state)
  while (sessions.size > SESSION_MAX) {
    const oldest = sessions.keys().next().value as string
    if (!oldest) break
    deleteSession(oldest)
  }
  scheduleSessionExpiry(key)
  writeSessionFile(key, state)
}

/**
 * A resume is only safe if the caller's history still contains the assistant turn the CLI transcript
 * holds, and if at least one new user message follows it. Compaction, retry or a truncated history
 * breaks that invariant; resuming anyway would replay stale history (or duplicate it) and burn
 * tokens. Falling back to a full replay is always correct, just colder.
 */
interface ResumeDecision {
  resume: string | null
  state: SessionState | null
  resetReason: string | null
}

function resolveSessionForRequest(key: string | null, messages: any[]): ResumeDecision {
  if (!key || !SESSION_MODE) return { resume: null, state: null, resetReason: null }
  ensureSessionsLoaded()
  const state = sessions.get(key) ?? null
  if (!state) return { resume: null, state: null, resetReason: null }

  let reason: string | null = null
  if (messages.length <= state.messageCount) {
    reason = "no_new_turn"
  } else if (messages.length <= state.messageCount + 1) {
    reason = "no_new_user_message"
  } else {
    const previous = messages[state.messageCount]
    if (!previous || previous.role !== "assistant") reason = "previous_assistant_missing"
  }

  if (reason) {
    deleteSession(key)
    claudeLog("proxy.session.reset", {
      key: key.slice(0, 12),
      reason,
      messages: messages.length,
      expected: state.messageCount + 2
    })
    return { resume: null, state: null, resetReason: reason }
  }
  return { resume: state.id, state, resetReason: null }
}

/** Deletes files older than `cutoff` under `dir`, then removes directories left empty. */
function sweepTree(dir: string, cutoff: number, counter: { removed: number }) {
  let entries: string[]
  try {
    entries = readdirSync(dir)
  } catch {
    return
  }
  for (const name of entries) {
    const path = join(dir, name)
    let stats
    try {
      stats = lstatSync(path)
    } catch {
      continue
    }
    if (stats.isDirectory() && !stats.isSymbolicLink()) {
      sweepTree(path, cutoff, counter)
      try {
        if (readdirSync(path).length === 0) rmdirSync(path)
      } catch {}
    } else if (stats.mtimeMs < cutoff) {
      try {
        rmSync(path, { force: true })
        counter.removed++
      } catch {}
    }
  }
}

/** Removes proxy/CLI scratch data that outlived the session TTL. */
function sweepSessions() {
  const cutoff = Date.now() - SESSION_TTL_MS
  let dropped = 0
  for (const [key, state] of sessions) {
    if (state.updatedAt < cutoff) {
      deleteSession(key)
      dropped++
    }
  }
  if (dropped > 0) claudeLog("proxy.session.expired_batch", { dropped })
  pruneSessionFiles(cutoff)

  const counter = { removed: 0 }
  // transcripts plus CLI scratch written under the proxy's isolated CLAUDE_CONFIG_DIR
  for (const dir of ["projects", "debug", "todos", "shell-snapshots", "statsig"]) {
    const path = join(SESSION_CONFIG_DIR, dir)
    if (existsSync(path)) sweepTree(path, cutoff, counter)
  }
  // The CLI rotates .claude.json backups on every config write. Keep the newest five so a long
  // session cannot fill the config dir with them.
  const backups: { path: string; mtime: number }[] = []
  try {
    for (const name of readdirSync(SESSION_CONFIG_DIR)) {
      if (!name.startsWith(".claude.json.backup.")) continue
      const path = join(SESSION_CONFIG_DIR, name)
      try {
        backups.push({ path, mtime: lstatSync(path).mtimeMs })
      } catch {}
    }
  } catch {}
  backups.sort((a, b) => b.mtime - a.mtime)
  for (const backup of backups.slice(5)) {
    try {
      rmSync(backup.path, { force: true })
      counter.removed++
    } catch {}
  }
  if (counter.removed > 0) claudeLog("proxy.session.swept", { removed: counter.removed })
}

function startSessionSweeper() {
  if (!SESSION_MODE) return
  mkdirSync(SESSION_CONFIG_DIR, { recursive: true })
  // The CLI loads its login from CLAUDE_CONFIG_DIR. Link the real credentials in so the isolated
  // dir authenticates without duplicating (or drifting from) the user's login.
  const own = join(SESSION_CONFIG_DIR, ".credentials.json")
  const source = join(homedir(), ".claude", ".credentials.json")
  if (!existsSync(own) && existsSync(source)) {
    try {
      symlinkSync(source, own)
    } catch {}
  }
  ensureSessionsLoaded()
  const interval = setInterval(sweepSessions, Math.min(SESSION_TTL_MS, 300_000))
  interval.unref?.()
  sweepSessions()
}

interface WarmEntry {
  key: string
  query: Query
  client: ClientTools | null
  abort: AbortController
  /** Hands the prompt to the already running CLI. */
  send: (prompt: string) => void
  /** The session id this process was started with, when it resumed a transcript. */
  resume: string | null
  timer: ReturnType<typeof setTimeout>
  /** Message count the caller must have on its next turn for this process to still be valid. */
  expectedMessageCount: number | null
}

/**
 * One warm process per conversation, not per shape. Map order is recency order, oldest first.
 *
 * A shape-keyed pool was actively harmful: every hit handed out a blank CLI, so the caller's whole
 * history became a fresh cache write instead of a cache read. Keying by conversation lets the warm
 * process resume that conversation's transcript, which is the only version of the pool that saves
 * tokens rather than burning them.
 */
const warmPool = new Map<string, WarmEntry>()

function warmShape(model: string, env: Env, clientTools: any[]): string {
  return createHash("sha1")
    .update(
      JSON.stringify([
        model,
        env.CLAUDE_CODE_MAX_OUTPUT_TOKENS ?? null,
        clientTools.map((t) => [t.name, t.description ?? null, t.input_schema ?? null])
      ])
    )
    .digest("hex")
}

/** A warm process is only reusable for the exact conversation it was started (or last used) for. */
function warmKey(shape: string, sessionKey: string): string {
  return createHash("sha1").update(JSON.stringify([shape, sessionKey])).digest("hex")
}

function discardWarm(entry: WarmEntry) {
  clearTimeout(entry.timer)
  entry.send("")
  entry.abort.abort()
  try {
    entry.query.close()
  } catch {}
}

function discardWarmKey(key: string) {
  const entry = warmPool.get(key)
  if (!entry) return
  warmPool.delete(key)
  discardWarm(entry)
}

function spawnWarm(
  key: string,
  model: string,
  env: Env,
  clientTools: any[],
  callerSystem: string,
  resume: string | null,
  expectedMessageCount: number | null = null
): WarmEntry {
  const abort = new AbortController()
  const client = clientTools.length > 0 ? buildClientTools(clientTools) : null

  let release!: (prompt: string) => void
  const gate = new Promise<string>((resolve) => (release = resolve))
  async function* input(): AsyncGenerator<SDKUserMessage> {
    const prompt = await gate
    if (!prompt) return // discarded before use
    yield { type: "user", message: { role: "user", content: prompt }, parent_tool_use_id: null, session_id: "" }
  }

  // A warm process resumes the conversation's transcript when it has one. Without this the pool
  // serves blank processes, which defeat prompt caching entirely.
  const options = buildOptions(model, env, client, abort, resume !== null, callerSystem) as any
  if (resume) options.resume = resume
  const q = query({ prompt: input(), options })
  const entry: WarmEntry = {
    key,
    query: q,
    client,
    abort,
    resume,
    expectedMessageCount,
    send: release,
    timer: setTimeout(() => {
      if (warmPool.get(key) !== entry) return
      warmPool.delete(key)
      claudeLog("proxy.warm.expired", { key: key.slice(0, 8) })
      discardWarm(entry)
    }, WARM_POOL_TTL_MS)
  }
  entry.timer.unref?.()
  return entry
}

/**
 * Takes this conversation's warm process, or discards it if the caller has moved past the transcript
 * it was scheduled against. Two writers on one conversation (another opencode, another litellm
 * replica) would otherwise resume a stale transcript and silently answer with the wrong context.
 */
function takeWarm(key: string, callerMessageCount: number): WarmEntry | null {
  const entry = warmPool.get(key)
  if (!entry) return null
  warmPool.delete(key)
  clearTimeout(entry.timer)
  if (entry.expectedMessageCount !== null && entry.expectedMessageCount !== callerMessageCount) {
    claudeLog("proxy.warm.stale", {
      key: key.slice(0, 8),
      expected: entry.expectedMessageCount,
      actual: callerMessageCount
    })
    discardWarm(entry)
    return null
  }
  return entry
}

function topUpWarm(
  key: string,
  model: string,
  env: Env,
  clientTools: any[],
  callerSystem: string,
  resume: string | null,
  expectedMessageCount: number | null = null
) {
  if (warmPool.has(key)) return
  while (warmPool.size >= WARM_POOL_SIZE) {
    const oldest = warmPool.keys().next().value as string
    discardWarm(warmPool.get(oldest)!)
    warmPool.delete(oldest)
  }
  try {
    warmPool.set(key, spawnWarm(key, model, env, clientTools, callerSystem, resume, expectedMessageCount))
  } catch (error) {
    claudeLog("proxy.warm.spawn_failed", { error: error instanceof Error ? error.message : String(error) })
  }
}

async function* runTurn(
  body: any,
  abort: AbortController,
  forceAlias = false,
  callerSession: string | null = null
): AsyncGenerator<StreamEvent> {
  const clientTools: any[] =
    Array.isArray(body.tools) && body.tool_choice?.type !== "none" ? body.tools.filter((t: any) => t?.name) : []

  const env = buildCliEnv()
  if (SESSION_MODE) env.CLAUDE_CONFIG_DIR = SESSION_CONFIG_DIR
  const maxTokens = Number(body.max_tokens)
  if (Number.isFinite(maxTokens) && maxTokens > 0) {
    env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = String(Math.min(Math.floor(maxTokens), DEFAULT_MAX_OUTPUT_TOKENS))
  }
  const model = resolveModel(body.model, forceAlias)

  const messages = Array.isArray(body.messages) ? body.messages : []
  const system = systemText(body.system)
  // Title generation and similar helpers carry the same caller session header but have no tools and
  // a single message. They must not create or overwrite the real conversation's transcript state.
  const helperRequest = clientTools.length === 0 && messages.length <= 1
  const track = callerSession !== null && !helperRequest
  const effectiveSession = track ? callerSession : null
  if (effectiveSession) {
    const pending = pendingTurns.get(effectiveSession)
    if (pending) {
      const started = Date.now()
      await Promise.race([pending, new Promise<void>((resolve) => setTimeout(resolve, PENDING_TURN_TIMEOUT_MS))])
      claudeLog("proxy.session.waited", { key: effectiveSession.slice(0, 12), ms: Date.now() - started })
    }
  }
  const decision = resolveSessionForRequest(effectiveSession, messages)
  const resume = decision.resume
  // A known conversation resumes its own transcript, which is what keeps tokens flat as it grows.
  const useResume = resume !== null
  const reset = decision.resetReason !== null

  // The pool is keyed by conversation, so a resuming turn can use a warm process too: that process
  // already sits on the same transcript. Only aliased retries skip it, since they change the shape.
  // A reset conversation must not reuse a process that resumes the transcript we just rejected.
  const shape = warmShape(model, env, clientTools)
  const key = effectiveSession ? warmKey(shape, effectiveSession) : ""
  if (reset && key) discardWarmKey(key)
  const usePool = WARM_POOL_SIZE > 0 && !forceAlias && effectiveSession !== null && !reset

  const warm = usePool ? takeWarm(key, messages.length) : null
  const client = warm ? warm.client : clientTools.length > 0 ? buildClientTools(clientTools) : null

  const nameToAlias = client?.nameToAlias ?? new Map<string, string>()
  // On a resume the transcript already holds the history, so only the new turn is sent. Otherwise
  // the whole conversation goes in: it becomes the transcript the next turn resumes from.
  const prompt = useResume ? buildSessionPrompt(messages, nameToAlias) : buildPrompt(messages, nameToAlias)
  // The warm replacement is a bet, not a computation: it cannot know how many blocks the reply will
  // have, so it expects the caller to come back carrying this turn's user message plus the reply.
  const expectedNextMessageCount = messages.length + 2

  claudeLog("proxy.anthropic.request", {
    model,
    requested: body.model,
    stream: body.stream === true,
    messages: body.messages?.length,
    tools: client?.allowed.length ?? 0,
    warm: usePool ? (warm ? "hit" : "miss") : "off",
    session: effectiveSession ? (useResume ? "resume" : reset ? `reset:${decision.resetReason}` : "new") : "off"
  })

  let response: Query
  if (warm) {
    // The warm CLI has its own AbortController: forward the request's abort to it.
    const kill = () => warm.abort.abort()
    if (abort.signal.aborted) kill()
    else abort.signal.addEventListener("abort", kill, { once: true })
    warm.send(prompt)
    response = warm.query
  } else {
    const options = buildOptions(model, env, client, abort, track, system) as any
    if (useResume) {
      options.resume = resume
    }
    response = query({ prompt, options })
  }
  let releasePending: (() => void) | null = null
  if (effectiveSession) {
    let release!: () => void
    const done = new Promise<void>((resolve) => (release = resolve))
    pendingTurns.set(effectiveSession, done)
    releasePending = () => {
      if (pendingTurns.get(effectiveSession!) === done) pendingTurns.delete(effectiveSession!)
      release()
    }
  }
  // Start the replacement after this request's own CLI, so it never competes with it for CPU on a miss.
  // It resumes the same conversation, so the next turn lands on a live transcript instead of a blank one.
  let started = false
  // The caller gets its assistant message as soon as `message_stop` arrives, but the CLI only
  // flushes the persisted transcript - and only the `result` carries its real session id - once the
  // turn has finished. Returning early or aborting the process here would leave nothing to resume,
  // which is exactly the `tool_use` case (every proxy turn ends that way). So the loop keeps
  // consuming the CLI without emitting anything after the first message, then remembers the id.
  let done = false
  let completed = false
  let result: string | null = null

  try {
    for await (const message of response) {
      if (message.type === "result") {
        // A warm process can report a fresh id for the same conversation; the caller's own id wins
        // so consecutive turns never fork the transcript.
        result = warm?.resume ?? (message as any).session_id ?? result
        done = true
        if (started) break
        const r = message as any
        const detail = r.result || (Array.isArray(r.errors) ? r.errors.join("; ") : "") || r.subtype || "unknown error"
        throw new Error(`Claude Code returned no message: ${detail}`)
      }
      if ((message as any).parent_tool_use_id) continue

      if (message.type === "stream_event") {
        let event = message.event as StreamEvent
        if (event.type === "message_start") {
          if (started) {
            done = true
            continue
          }
          started = true
          const u = (event as any).message?.usage ?? {}
          claudeLog("proxy.anthropic.usage", {
            input: u.input_tokens ?? null,
            cache_read: u.cache_read_input_tokens ?? null,
            cache_creation: u.cache_creation_input_tokens ?? null,
            output: u.output_tokens ?? null
          })
        }
        if (!started || done) continue

        if (event.type === "content_block_start" && event.content_block?.type === "tool_use") {
          event = {
            ...event,
            content_block: { ...event.content_block, name: originalToolName(event.content_block.name, client) }
          }
        }
        if (event.type === "message_delta") {
          event = { ...event, usage: { output_tokens: 0, ...(event.usage ?? {}) } }
        }

        yield event
        if (event.type === "message_stop") {
          completed = true
          // Stop emitting, but keep draining the CLI: only its `result` carries the session id and
          // only after that does it flush the transcript we need for the next turn.
          done = true
        }
      } else if (message.type === "assistant" && !started) {
        // No partial events were produced. The CLI reports upstream failures as a synthetic assistant
        // message ("API Error: 400 {...}"): surface those as real HTTP errors, not as model output.
        const blocks: any[] = Array.isArray(message.message?.content) ? message.message.content : []
        const text = blocks
          .filter((b) => b?.type === "text")
          .map((b) => b.text ?? "")
          .join("")
        const apiError = /^API Error:\s*(\d{3})\s*([\s\S]*)$/.exec(text.trim())
        if (apiError) throw upstreamError(Number(apiError[1]), apiError[2] ?? "")

        // Otherwise replay the full message (e.g. other synthetic CLI messages).
        started = true
        for (const ev of eventsFromAssistantMessage(message.message, client)) {
          yield ev
        }
        completed = true
        done = true
      }
    }
    if (!started) throw new Error("Claude Code returned no message")
  } catch (error) {
    // Aborting after we got what we needed is expected and not an error.
    if (!(started && abort.signal.aborted)) throw error
  } finally {
    // Always let the CLI finish writing its transcript and report the session id, even when the
    // caller already got its message. Killing it here is what left nothing to resume from.
    if (track || result !== null) {
      await new Promise((resolve) => setTimeout(resolve, SESSION_FLUSH_MS))
    }
    const id = result
    if (track && id && completed) {
      claudeLog("proxy.session.remember", { sid: id.slice(0, 8), messages: messages.length })
      rememberSession(effectiveSession!, {
        id,
        messageCount: messages.length,
        updatedAt: Date.now()
      })
      // Spawn the replacement only now, holding the id this turn actually produced. Scheduling it
      // before the turn ran always resumed the previous id, so the pool forked the conversation.
      if (usePool) {
        topUpWarm(key, model, env, clientTools, system, id, expectedNextMessageCount)
      }
    } else if (track && !id) {
      // No session id means there is nothing to resume; a later turn must replay the history.
      deleteSession(effectiveSession!)
    }
    abort.abort()
    releasePending?.()
  }
}

/** Retries once with the CLI alias when an exact model id was rejected before any output. */
async function* runTurnWithFallback(
  body: any,
  abort: AbortController,
  callerSession: string | null = null
): AsyncGenerator<StreamEvent> {
  const exact = resolveModel(body.model) !== aliasForModel(String(body.model ?? ""))
  const first = runTurn(body, abort, false, callerSession)
  let started = false
  try {
    for await (const ev of first) {
      started = true
      yield ev
    }
    return
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    // The remembered session may be gone from disk (TTL cleanup, manual removal, reinstall). Forget
    // it and replay the full history instead of failing the caller's turn.
    if (!started && callerSession && /No conversation found/i.test(message)) {
      deleteSession(callerSession)
      claudeLog("proxy.session.stale", { key: callerSession.slice(0, 12) })
      // Keep the caller key: the retry replays the full history and records the new transcript
      // under the same conversation, so the following turn can resume again.
      yield* runTurn(body, new AbortController(), false, callerSession)
      return
    }
    if (started || !exact || !/model/i.test(message) || abort.signal.aborted) throw error
    console.error("[claude-max-proxy] exact model rejected, retrying with alias:", body.model, "-", message.slice(0, 200))
  }
  yield* runTurn(body, new AbortController(), true, callerSession)
}

// ---------------------------------------------------------------------------------------
// Non-streaming: fold the event stream into a Messages API response
// ---------------------------------------------------------------------------------------

async function collectMessage(events: AsyncGenerator<StreamEvent>, fallbackModel: string): Promise<Record<string, any>> {
  let message: Record<string, any> = {
    id: `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    model: fallbackModel,
    content: [],
    stop_reason: null,
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: 0 }
  }
  const blocks: Record<number, any> = {}

  for await (const ev of events) {
    switch (ev.type) {
      case "message_start":
        message = { ...message, ...ev.message, content: [] }
        break
      case "content_block_start":
        blocks[ev.index] = { ...ev.content_block, _json: "" }
        break
      case "content_block_delta": {
        const block = blocks[ev.index]
        if (!block) break
        const d = ev.delta
        if (d.type === "text_delta") block.text = (block.text ?? "") + d.text
        else if (d.type === "input_json_delta") block._json += d.partial_json
        else if (d.type === "thinking_delta") block.thinking = (block.thinking ?? "") + d.thinking
        else if (d.type === "signature_delta") block.signature = d.signature
        break
      }
      case "content_block_stop": {
        const block = blocks[ev.index]
        if (!block) break
        if (block.type === "tool_use") {
          try {
            block.input = block._json ? JSON.parse(block._json) : {}
          } catch {
            block.input = {}
          }
        }
        delete block._json
        message.content[ev.index] = block
        break
      }
      case "message_delta":
        message.stop_reason = ev.delta?.stop_reason ?? message.stop_reason
        message.stop_sequence = ev.delta?.stop_sequence ?? null
        message.usage = { ...message.usage, ...(ev.usage ?? {}) }
        break
    }
  }
  message.content = message.content.filter(Boolean)
  return message
}

// ---------------------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------------------

function errorStatus(message: string): 429 | 500 {
  return /rate.?limit|usage limit|hit your limit|too many requests|overloaded/i.test(message) ? 429 : 500
}

export function createProxyServer(config: Partial<ProxyConfig> = {}) {
  const finalConfig = { ...DEFAULT_PROXY_CONFIG, ...config }
  const app = new Hono()

  app.use("*", cors())

  app.get("/", (c) => {
    return c.json({
      status: "ok",
      service: "claude-max-proxy",
      version: "1.1.0",
      format: "anthropic",
      endpoints: ["/v1/messages", "/messages"]
    })
  })

  const handleMessages = async (c: Context) => {
    try {
      const body = await c.req.json()
      // Anthropic semantics: streaming is opt-in. litellm omits `stream` on non-streaming calls.
      const stream = body.stream === true
      const abort = new AbortController()
      c.req.raw.signal?.addEventListener("abort", () => abort.abort())

      const callerMessages = Array.isArray(body.messages) ? body.messages : []
      const events = runTurnWithFallback(body, abort, sessionKey(callerMessages, systemText(body.system), c))

      if (!stream) {
        const message = await collectMessage(events, String(body.model ?? ""))
        return c.json(message)
      }

      // Peek the first event so startup failures become a real HTTP error instead of an empty 200 stream.
      const first = await events.next()
      const encoder = new TextEncoder()
      const sse = (ev: StreamEvent) => encoder.encode(`event: ${ev.type}\ndata: ${JSON.stringify(ev)}\n\n`)

      const readable = new ReadableStream({
        async start(controller) {
          const heartbeat = setInterval(() => {
            try {
              controller.enqueue(sse({ type: "ping" }))
            } catch {
              clearInterval(heartbeat)
            }
          }, 10_000)
          try {
            if (!first.done) controller.enqueue(sse(first.value))
            for await (const ev of events) controller.enqueue(sse(ev))
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error)
            console.error("[claude-max-proxy] stream error:", message)
            try {
              controller.enqueue(sse({ type: "error", error: { type: "api_error", message } }))
            } catch {}
          } finally {
            clearInterval(heartbeat)
            try {
              controller.close()
            } catch {}
          }
        },
        cancel() {
          abort.abort()
          void events.return?.(undefined)
        }
      })

      return new Response(readable, {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache",
          Connection: "keep-alive"
        }
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error"
      const status = error instanceof ProxyError ? error.status : errorStatus(message)
      const errorType = error instanceof ProxyError ? error.errorType : "api_error"
      console.error("[claude-max-proxy] request error:", status, message)
      return c.json({ type: "error", error: { type: errorType, message } }, status as any)
    }
  }

  app.post("/v1/messages", handleMessages)
  app.post("/messages", handleMessages)

  return { app, config: finalConfig }
}

export async function startProxyServer(config: Partial<ProxyConfig> = {}) {
  const { app, config: finalConfig } = createProxyServer(config)

  const server = Bun.serve({
    port: finalConfig.port,
    hostname: finalConfig.host,
    // Bun's default (10s) drops connections that are idle while the model is thinking.
    idleTimeout: 255,
    fetch: app.fetch
  })

  console.log(`Claude Max Proxy (Anthropic API) running at http://${finalConfig.host}:${finalConfig.port}`)
  startSessionSweeper()

  return server
}
