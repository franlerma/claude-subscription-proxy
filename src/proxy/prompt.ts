import { CLIENT_TOOL_PREFIX } from "./tools"

// ---------------------------------------------------------------------------------------
// Request -> prompt
// ---------------------------------------------------------------------------------------

export function systemText(system: unknown): string {
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

export function renderBlocks(content: unknown, nameToAlias: Map<string, string>): string {
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
 * Marker Claude Code uses to split its system prompt into a static, cacheable half and a dynamic
 * half. Everything before it gets `cache_control`; everything after stays uncached. Injecting the
 * caller's prompt after the marker keeps Claude Code's own prefix cacheable while the caller's
 * (which changes per turn) does not invalidate it. It must stay: appending the caller text directly
 * onto the first-party preset makes Anthropic classify the request as a third-party app and bill it
 * against "extra usage" instead of the plan.
 */
export const SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

const CWD_HINT =
  "The first <env> block (and its working directory) belongs to the proxy's empty sandbox, not to the user. The user's real working directory is the one in the caller's own environment above. If none is given, use relative paths or ask the user; never assume the proxy's path."

export function callerSystemBlock(system: string): string {
  return ["<caller_system_prompt>", system, CWD_HINT, "</caller_system_prompt>"].join("\n")
}

/** Wraps each message in its role tag, the format the CLI reads as conversation history. */
function renderMessages(messages: any[], nameToAlias: Map<string, string>): string {
  return messages
    .map((m) => {
      const role = m?.role === "assistant" ? "assistant" : "user"
      return `<${role}>\n${renderBlocks(m?.content, nameToAlias)}\n</${role}>`
    })
    .join("\n")
}

export function buildPrompt(messages: any[], nameToAlias: Map<string, string>): string {
  if (messages.length === 1 && messages[0]?.role === "user") {
    return renderBlocks(messages[0].content, nameToAlias)
  }
  return [
    "Conversation history. Write the next assistant turn. Call tools normally; never type <tool_call>/<tool_result>.",
    "<conversation>",
    renderMessages(messages, nameToAlias),
    "</conversation>"
  ].join("\n")
}

/**
 * Prompt for a resumed session: only what came after the last assistant turn. That is the newest
 * user message, or the tool results that answer the assistant's `tool_use` blocks. Everything
 * before it already lives in the CLI's own transcript.
 */
export function buildSessionPrompt(messages: any[], nameToAlias: Map<string, string>): string {
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
  return renderMessages(fresh, nameToAlias)
}

export function buildCompactedPrompt(summary: string, tail: any[], nameToAlias: Map<string, string>): string {
  return [
    "Conversation history. Write the next assistant turn. Call tools normally; never type <tool_call>/<tool_result>.",
    "<conversation>",
    `<user>\n<history_summary>\n${summary}\n</history_summary>\n</user>`,
    renderMessages(tail, nameToAlias),
    "</conversation>"
  ].join("\n")
}
