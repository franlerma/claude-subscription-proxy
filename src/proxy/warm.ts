import { createHash } from "crypto"
import { query } from "@anthropic-ai/claude-agent-sdk"
import type { Query, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk"
import { claudeLog } from "../logger"
import { buildOptions, type Env, WARM_POOL_SIZE, WARM_POOL_TTL_MS } from "./env"
import { buildClientTools, type ClientTools } from "./tools"

export interface WarmEntry {
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
  /** Accumulated CLI stderr, so a failed turn can report the real reason. */
  stderr: string[]
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

export function warmShape(model: string, env: Env, clientTools: any[]): string {
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
export function warmKey(shape: string, sessionKey: string): string {
  return createHash("sha1").update(JSON.stringify([shape, sessionKey])).digest("hex")
}

function discardWarm(entry: WarmEntry) {
  clearTimeout(entry.timer)
  // Close the query before aborting: aborting first makes the SDK's control-request writer
  // throw "Operation aborted" on an already-closed stdin, which crashes the whole process.
  try {
    entry.query.close()
  } catch {}
  try {
    entry.abort.abort()
  } catch {}
  try {
    entry.send("")
  } catch {}
}

export function discardWarmKey(key: string) {
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
  const stderr: string[] = []

  let release!: (prompt: string) => void
  const gate = new Promise<string>((resolve) => (release = resolve))
  async function* input(): AsyncGenerator<SDKUserMessage> {
    const prompt = await gate
    if (!prompt) return // discarded before use
    yield { type: "user", message: { role: "user", content: prompt }, parent_tool_use_id: null, session_id: "" }
  }

  // A warm process resumes the conversation's transcript when it has one. Without this the pool
  // serves blank processes, which defeat prompt caching entirely.
  const options = buildOptions(model, env, client, abort, resume !== null, callerSystem, stderr) as any
  if (resume) options.resume = resume
  const q = query({ prompt: input(), options })
  const entry: WarmEntry = {
    key,
    query: q,
    client,
    abort,
    resume,
    expectedMessageCount,
    stderr,
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
export function takeWarm(key: string, callerMessageCount: number): WarmEntry | null {
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

export function topUpWarm(
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
