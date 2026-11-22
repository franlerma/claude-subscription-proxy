import { Hono } from "hono"
import type { Context } from "hono"
import type { ProxyConfig } from "./types"
import { DEFAULT_PROXY_CONFIG } from "./types"
import { claudeLog } from "../logger"
import { ProxyError, isRateLimitText } from "./errors"
import { version as PROXY_VERSION } from "../../package.json"
import type { StreamEvent } from "./errors"
import { runTurnWithFallback } from "./turn"
import { sessionKey, startSessionSweeper } from "./session"
import { systemText } from "./prompt"

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
  return isRateLimitText(message) ? 429 : 500
}

export function createProxyServer(config: Partial<ProxyConfig> = {}) {
  const finalConfig = { ...DEFAULT_PROXY_CONFIG, ...config }
  const app = new Hono()

  app.get("/", (c) => {
    return c.json({
      status: "ok",
      service: "claude-subscription-proxy",
      version: PROXY_VERSION,
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

      // Once the client is gone we keep draining `events` instead of closing the generator: the turn
      // still has to read the CLI's `result` to remember the session. `runTurn` aborts the CLI itself
      // when the disconnect arrives before `message_stop`.
      let cancelled = false
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
            if (!first.done && !cancelled) controller.enqueue(sse(first.value))
            for await (const ev of events) {
              if (!cancelled) controller.enqueue(sse(ev))
            }
          } catch (error) {
            if (!cancelled) {
              const message = error instanceof Error ? error.message : String(error)
              console.error("[claude-subscription-proxy] stream error:", message)
              try {
                controller.enqueue(sse({ type: "error", error: { type: "api_error", message } }))
              } catch {}
            }
          } finally {
            clearInterval(heartbeat)
            try {
              controller.close()
            } catch {}
          }
        },
        cancel() {
          cancelled = true
          abort.abort()
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
      console.error("[claude-subscription-proxy] request error:", status, message)
      return c.json({ type: "error", error: { type: errorType, message } }, status as any)
    }
  }

  app.post("/v1/messages", handleMessages)
  app.post("/messages", handleMessages)

  return { app, config: finalConfig }
}

/**
 * The SDK aborts its control-request writer with "Operation aborted" whenever a turn (or a warm
 * process) is discarded. That is expected cleanup, not a fatal condition: without these handlers the
 * rejection surfaces as an uncaught exception and kills the whole proxy, dropping every session and
 * warm process. Keep the process alive; only log genuinely unexpected errors.
 */
function installAbortGuards() {
  const isAbort = (reason: unknown) =>
    reason instanceof Error
      ? /aborted|abort/i.test(reason.message)
      : /aborted|abort/i.test(String(reason))

  process.on("unhandledRejection", (reason) => {
    if (isAbort(reason)) return
    console.error("[claude-subscription-proxy] unhandled rejection:", reason)
    claudeLog("proxy.unhandled_rejection", { error: String(reason) })
  })

  process.on("uncaughtException", (error) => {
    if (isAbort(error)) return
    console.error("[claude-subscription-proxy] uncaught exception, exiting:", error)
    claudeLog("proxy.uncaught_exception", { error: error instanceof Error ? error.message : String(error) })
    process.exit(1)
  })
}

export async function startProxyServer(config: Partial<ProxyConfig> = {}) {
  installAbortGuards()
  const { app, config: finalConfig } = createProxyServer(config)

  const server = Bun.serve({
    port: finalConfig.port,
    hostname: finalConfig.host,
    // Bun's default (10s) drops connections that are idle while the model is thinking.
    idleTimeout: 255,
    fetch: app.fetch
  })

  console.log(`Claude Subscription Proxy (Anthropic API) running at http://${finalConfig.host}:${finalConfig.port}`)
  startSessionSweeper()

  return server
}
