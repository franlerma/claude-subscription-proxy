import { query } from "@anthropic-ai/claude-agent-sdk"
import type { Query, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk"
import { claudeLog } from "../logger"
import { claudeExecutable } from "./models"
import { SYSTEM_PROMPT_DYNAMIC_BOUNDARY, renderBlocks } from "./prompt"
import { buildCliEnv, type Env, SESSION_MODE, SESSION_CONFIG_DIR, WARM_POOL_TTL_MS } from "./env"
import { upstreamError, isRateLimitText } from "./errors"

// ---------------------------------------------------------------------------------------
// Haiku summarizer (history compaction)
//
// A dedicated haiku process compacts a cold, oversized history before it is handed to the turn's
// model. It shares the caller's subscription, so it must stay first-party: it runs the `claude_code`
// preset with the summary instruction appended after the dynamic boundary, exactly like a normal
// turn. `persistSession: false` keeps it out of the session store; `tools: []` runs nothing on the
// host. One process is kept warm waiting for work; concurrent summaries spawn as many extra
// processes as needed, and each is discarded after one summary, leaving a single warm process behind.
// ---------------------------------------------------------------------------------------

const SUMMARIZE_INSTRUCTION = [
  "You are compressing a conversation so it can continue in a fresh context.",
  "From the recent history below, produce a short summary of what the user is trying to achieve right now, plus the key decisions and file paths from the recent turns.",
  "Be as concise, fast and token-efficient as possible: a few short lines, no structure headings, no preamble, nothing that is not essential."
].join(" ")

// The whole cold history is far too expensive to summarise (and the main model far too expensive to
// feed it to). Instead send haiku only the recent tail. The main model re-reads the code on its
// next turn. Keep the tail tiny: the last ~8 messages, capped at ~12k
// chars (~3k tokens), which is enough for the current intent and the immediate tool results.
const SUMMARIZE_TAIL_MESSAGES = 8
const SUMMARIZE_TAIL_CHARS = 12_000

/** The recent history tail, oldest-first, capped by message count and char budget. */
function recentHistory(messages: any[]): string {
  const blocks: string[] = []
  let chars = 0
  // Walk backwards so the most recent messages always fit within the budget.
  for (let i = messages.length - 1; i >= 0 && blocks.length < SUMMARIZE_TAIL_MESSAGES; i--) {
    const m = messages[i]
    const role = m?.role === "assistant" ? "assistant" : "user"
    const block = `<${role}>\n${renderBlocks(m?.content, new Map())}\n</${role}>`
    if (chars + block.length > SUMMARIZE_TAIL_CHARS && blocks.length > 0) break
    blocks.push(block)
    chars += block.length
  }
  return blocks.reverse().join("\n")
}

interface Summarizer {
  query: Query
  abort: AbortController
  send: (prompt: string) => void
  /** Accumulates the CLI's stderr so failures surface the real reason, not just the exit code. */
  stderr: string[]
}

let warmSummarizer: Summarizer | null = null
let warmSummarizerTimer: ReturnType<typeof setTimeout> | null = null

// One haiku process per summary. It gets only the recent history tail - no project access - and
// interpolates a short summary from it. Kept warm so the CLI start-up is paid once, not per summary.
function spawnSummarizer(env: Env): Summarizer {
  const abort = new AbortController()
  const stderr: string[] = []
  let release!: (prompt: string) => void
  const gate = new Promise<string>((resolve) => (release = resolve))
  async function* input(): AsyncGenerator<SDKUserMessage> {
    const prompt = await gate
    if (!prompt) return
    yield { type: "user", message: { role: "user", content: prompt }, parent_tool_use_id: null, session_id: "" }
  }
  const options: Record<string, any> = {
    model: "haiku",
    pathToClaudeCodeExecutable: claudeExecutable,
    systemPrompt: {
      type: "preset" as const,
      preset: "claude_code" as const,
      append: `${SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n${SUMMARIZE_INSTRUCTION}`
    },
    // No tools: the summarizer only interpolates a short summary from the recent history it is given.
    // Exploring the project was too slow and made the summary unpredictable. The main model does its
    // own reading on the next turn without being conditioned by a pre-baked summary.
    tools: [],
    settingSources: [],
    maxTurns: 1,
    includePartialMessages: false,
    persistSession: false,
    abortController: abort,
    env,
    stderr: (data: string) => stderr.push(data)
  }
  const q = query({ prompt: input(), options })
  return { query: q, abort, send: release, stderr }
}

function discardSummarizer(entry: Summarizer) {
  try {
    entry.query.close()
  } catch {}
  try {
    entry.abort.abort()
  } catch {}
  try {
    entry.send("") // sentinel: ends the input stream and lets the CLI finish
  } catch {}
}

function ensureWarmSummarizer(env: Env) {
  if (warmSummarizer) return
  try {
    warmSummarizer = spawnSummarizer(env)
    // Without a TTL the idle haiku CLI lived for the whole proxy lifetime, holding a process (and
    // its cached prefix) for a compaction that may never come.
    if (warmSummarizerTimer) clearTimeout(warmSummarizerTimer)
    warmSummarizerTimer = setTimeout(() => {
      const entry = warmSummarizer
      warmSummarizer = null
      warmSummarizerTimer = null
      if (entry) discardSummarizer(entry)
    }, WARM_POOL_TTL_MS)
    warmSummarizerTimer.unref?.()
  } catch (error) {
    claudeLog("proxy.haiku.warm_failed", { error: error instanceof Error ? error.message : String(error) })
  }
}

async function runSummarizer(entry: Summarizer, prompt: string): Promise<string> {
  entry.send(prompt)
  let text = ""
  for await (const message of entry.query) {
    if (message.type === "assistant") {
      // A spent plan surfaces as a synthetic assistant message with an `error` field, or as the exact
      // literal "You've hit your limit ... (UTC)" text. Only those are rate limits — the summary's own
      // text may legitimately mention "usage" or "rate" and must not be misread as one.
      const err = (message as any).error
      if (err === "rate_limit" || err === "billing_error") {
        throw upstreamError(429, err)
      }
      const blocks: any[] = Array.isArray(message.message?.content) ? message.message.content : []
      text = blocks
        .filter((b: any) => b?.type === "text")
        .map((b: any) => b.text ?? "")
        .join("")
      if (/^You've hit your limit\b/.test(text.trim()) || /^API Error:\s*429\b/.test(text.trim())) {
        throw upstreamError(429, text.trim())
      }
    }
  }
  return text.trim()
}

/**
 * Produces a compact context summary with haiku from the recent history tail only, instead of the
 * whole cold history.
 * Throws on failure: the summary and the turn share the same subscription and endpoint, so a failed
 * summary means the main call would fail too, and replaying the full history would only burn more.
 */
export async function summarizeHistory(messages: any[]): Promise<string> {
  const env = buildCliEnv()
  if (SESSION_MODE) env.CLAUDE_CONFIG_DIR = SESSION_CONFIG_DIR
  const tail = recentHistory(messages)
  const inChars = tail.length

  console.log(
    `[claude-subscription-proxy] haiku summary: starting, ${messages.length} messages total, recent ${inChars} chars`
  )

  let entry: Summarizer | null = null
  try {
    entry = warmSummarizer ?? spawnSummarizer(env)
    warmSummarizer = null
    if (warmSummarizerTimer) {
      clearTimeout(warmSummarizerTimer)
      warmSummarizerTimer = null
    }

    const summary = await runSummarizer(entry, tail)
    if (!summary) throw new Error("haiku summary returned no text")
    const ratio = inChars > 0 ? Math.round((1 - summary.length / inChars) * 100) : 0
    console.log(
      `[claude-subscription-proxy] haiku summary: ${messages.length} messages total, ` +
        `${inChars} -> ${summary.length} chars (${ratio}% smaller)`
    )
    claudeLog("proxy.haiku.summarize", {
      messages: messages.length,
      inChars,
      outChars: summary.length,
      reducedPct: ratio
    })
    return summary
  } catch (error) {
    // A failed summary means the account is spent or unreachable; the main call shares that fate, so
    // abort the turn instead of replaying the full history against the same wall. Always surface the
    // CLI's own stderr: the exit code alone hides the real reason (429, etc.).
    const stderr = entry?.stderr.join("").trim() ?? ""
    if (stderr) console.error(`[claude-subscription-proxy] haiku CLI stderr:\n${stderr.slice(0, 1500)}`)
    if (isRateLimitText(stderr)) {
      const rl = upstreamError(429, stderr.slice(0, 1000))
      console.error("[claude-subscription-proxy] haiku summary failed:", rl.message)
      claudeLog("proxy.haiku.summarize_failed", { error: rl.message })
      throw rl
    }
    console.error("[claude-subscription-proxy] haiku summary failed:", error instanceof Error ? error.message : String(error))
    claudeLog("proxy.haiku.summarize_failed", { error: error instanceof Error ? error.message : String(error) })
    throw error
  } finally {
    if (entry) discardSummarizer(entry)
    ensureWarmSummarizer(env)
  }
}
