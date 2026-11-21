import { query } from "@anthropic-ai/claude-agent-sdk"
import type { Query } from "@anthropic-ai/claude-agent-sdk"
import { claudeLog } from "../logger"
import { ProxyError, upstreamError } from "./errors"
import type { StreamEvent } from "./errors"
import { resolveModel, aliasForModel, DEFAULT_MAX_OUTPUT_TOKENS, thinkingBudget } from "./models"
import { buildClientTools, originalToolName, type ClientTools } from "./tools"
import { systemText, buildPrompt, buildSessionPrompt, buildCompactedPrompt } from "./prompt"
import {
  buildCliEnv,
  buildOptions,
  WARM_POOL_SIZE,
  SESSION_MODE,
  SESSION_CONFIG_DIR,
  SESSION_FLUSH_MS,
  COMPACT_HISTORY_TOKENS,
  COMPACT_KEEP_MESSAGES,
  HELPER_MODEL,
  estimateHistoryTokens
} from "./env"
import {
  resolveSessionForRequest,
  rememberSession,
  deleteSession,
  pendingTurns,
  PENDING_TURN_TIMEOUT_MS
} from "./session"
import { warmShape, warmKey, discardWarmKey, takeWarm, topUpWarm } from "./warm"
import { summarizeHistory } from "./summarizer"

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

  const messages = Array.isArray(body.messages) ? body.messages : []
  const system = systemText(body.system)
  // Title generation and similar helpers carry the same caller session header but have no tools and
  // a single message. They must not create or overwrite the real conversation's transcript state.
  const helperRequest = clientTools.length === 0 && messages.length <= 1
  // Helpers never need the conversation's model: route them to a cheap alias so a title costs a
  // haiku call instead of a Sonnet/Opus one. The caller's `thinking` budget is forwarded as-is.
  const model = resolveModel(helperRequest ? HELPER_MODEL : body.model, forceAlias)
  const maxThinkingTokens = thinkingBudget(body.thinking)
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
  const reset = decision.resetReason !== null

  // History compaction. The prompt cache is per model, so a cold history (one with no Claude
  // transcript to resume) or a mid-conversation model switch would re-send everything with no chance
  // of a cache read. Summarise that history once with haiku and keep only the newest messages
  // verbatim. A live transcript in the same model is never compacted: resuming reads the cached
  // prefix, so growth there is already cheap.
  const modelSwitch =
    decision.resume !== null && decision.state?.model !== undefined && decision.state.model !== model
  const coldHistory =
    decision.resume === null && !reset && estimateHistoryTokens(messages) > COMPACT_HISTORY_TOKENS
  const compact =
    COMPACT_HISTORY_TOKENS > 0 && messages.length > COMPACT_KEEP_MESSAGES + 1 && (modelSwitch || coldHistory)
  // A compacted turn starts a fresh transcript, so it never resumes the previous one.
  const useResume = decision.resume !== null && !compact
  const resume = useResume ? decision.resume : null

  // The pool is keyed by conversation, so a resuming turn can use a warm process too: that process
  // already sits on the same transcript. Only aliased retries skip it, since they change the shape.
  // A reset conversation must not reuse a process that resumes the transcript we just rejected.
  const shape = warmShape(model, env, clientTools, maxThinkingTokens)
  const key = effectiveSession ? warmKey(shape, effectiveSession) : ""
  if (reset && key) discardWarmKey(key)
  const usePool = WARM_POOL_SIZE > 0 && !forceAlias && effectiveSession !== null && !reset && !compact

  const warm = usePool ? takeWarm(key, messages.length) : null
  const client = warm ? warm.client : clientTools.length > 0 ? buildClientTools(clientTools) : null

  const nameToAlias = client?.nameToAlias ?? new Map<string, string>()
  // On a resume the transcript already holds the history, so only the new turn is sent. Otherwise
  // the whole conversation goes in: it becomes the transcript the next turn resumes from. A compacted
  // turn sends the haiku summary plus the newest messages verbatim.
  let prompt: string
  if (compact) {
    const summary = await summarizeHistory(messages.slice(0, messages.length - COMPACT_KEEP_MESSAGES))
    prompt = summary
      ? buildCompactedPrompt(summary, messages.slice(-COMPACT_KEEP_MESSAGES), nameToAlias)
      : buildPrompt(messages, nameToAlias)
  } else {
    prompt = useResume ? buildSessionPrompt(messages, nameToAlias) : buildPrompt(messages, nameToAlias)
  }
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
    session: effectiveSession ? (compact ? "compact" : useResume ? "resume" : reset ? `reset:${decision.resetReason}` : "new") : "off"
  })

  let response: Query
  const coldStderr: string[] = []
  const stderrBuf = () => (warm ? warm.stderr : coldStderr).join("").trim()
  if (warm) {
    // The warm CLI has its own AbortController: forward the request's abort to it.
    const kill = () => warm.abort.abort()
    if (abort.signal.aborted) kill()
    else abort.signal.addEventListener("abort", kill, { once: true })
    warm.send(prompt)
    response = warm.query
  } else {
    const options = buildOptions(model, env, client, abort, track, system, coldStderr, maxThinkingTokens) as any
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
  let usage: Record<string, any> = {}
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
          usage = { ...u }
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
          usage = { ...usage, ...(event.usage ?? {}) }
          // One line per turn, unconditional: cache_read vs cache_creation is the only signal that
          // tells a healthy cached prefix from a cold replay, and it must be visible in journald.
          console.log(
            `[claude-subscription-proxy] usage model=${model} input=${usage.input_tokens ?? 0} ` +
              `cache_read=${usage.cache_read_input_tokens ?? 0} ` +
              `cache_creation=${usage.cache_creation_input_tokens ?? 0} output=${usage.output_tokens ?? 0}`
          )
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

        // The CLI reports a spent plan / 5-hour window as a plain text message, not an API Error
        // line. Surface it as HTTP 429 so the caller stops instead of showing it as model output.
        if (/hit your limit|resets \d|5.?hour|usage limit|rate.?limit|too many requests/i.test(text)) {
          throw upstreamError(429, text.trim())
        }

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
    if (started && abort.signal.aborted) {
      // expected
    } else {
      // The CLI wraps upstream failures as a bare exit code; the real cause (e.g. a 429) is only on
      // its stderr. Always print it, then surface a rate limit as HTTP 429 instead of a generic 500.
      const stderr = stderrBuf()
      if (stderr) console.error(`[claude-subscription-proxy] CLI stderr:\n${stderr.slice(0, 1500)}`)
      if (/rate.?limit|429|usage limit|hit your limit|too many requests|overloaded/i.test(stderr)) {
        throw upstreamError(429, stderr.slice(0, 1000))
      }
      if (stderr && error instanceof Error && /process exited|no message/i.test(error.message)) {
        throw new Error(`${error.message}: ${stderr.slice(0, 1000)}`)
      }
      throw error
    }
  } finally {
    // Always let the CLI finish writing its transcript and report the session id, even when the
    // caller already got its message. Killing it here is what left nothing to resume from.
    // A `result` already proves the transcript was flushed, so only wait when it is still missing.
    if (result === null && track) {
      await new Promise((resolve) => setTimeout(resolve, SESSION_FLUSH_MS))
    }
    // A turn that already emitted its message but whose CLI ended without a `result` (the process
    // was killed, or the stream closed early) still owns the transcript it resumed from. Deleting
    // the session there is what forced the next turn to replay the whole history cold - the exact
    // case the cache cannot survive. Keep the id we resumed from instead of throwing it away.
    const resumedId = warm?.resume ?? resume
    const id = result ?? (completed ? resumedId : null)
    if (track && id) {
      claudeLog("proxy.session.remember", {
        sid: id.slice(0, 8),
        messages: messages.length,
        completed,
        viaResult: result !== null
      })
      rememberSession(effectiveSession!, {
        id,
        messageCount: messages.length,
        updatedAt: Date.now(),
        model
      })
      // Spawn the replacement only now, holding the id this turn actually produced. Scheduling it
      // before the turn ran always resumed the previous id, so the pool forked the conversation.
      // Only a `result` proves the transcript was flushed; without it a warm process could resume a
      // half-written transcript, so the pool is left cold for one turn instead.
      if (usePool && result !== null) {
        topUpWarm(key, model, env, clientTools, system, id, expectedNextMessageCount, maxThinkingTokens)
      }
    } else if (track && started) {
      // The turn emitted output but did not yield a session id to remember: either it never
      // finished, or it was a compacted turn whose new transcript is unknown. The mapping would
      // otherwise keep pointing at the pre-compaction transcript, so drop it and let a later turn
      // replay the history. A failure before the first token (a 429, a CLI that never started)
      // leaves the previous transcript intact - keep it, both because it is still valid and because
      // dropping it turns the next turn into a cold replay.
      deleteSession(effectiveSession!)
    }
    abort.abort()
    releasePending?.()
  }
}

/** Retries once with the CLI alias when an exact model id was rejected before any output. */
export async function* runTurnWithFallback(
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
    // A spent plan / overloaded upstream can mention "model" in its text, but retrying with the
    // alias would just be a second doomed call against the same subscription. Only a genuine
    // model-id rejection (and never a rate limit) is worth a retry.
    if (error instanceof ProxyError && error.status === 429) throw error
    if (/rate.?limit|usage limit|hit your limit|too many requests|overloaded/i.test(message)) throw error
    console.error("[claude-subscription-proxy] exact model rejected, retrying with alias:", body.model, "-", message.slice(0, 200))
  }
  yield* runTurn(body, new AbortController(), true, callerSession)
}
