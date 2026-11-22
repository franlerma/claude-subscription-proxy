import { execSync } from "child_process"
import { fileURLToPath } from "url"
import { join, dirname } from "path"
import { existsSync } from "fs"

export const DEFAULT_MAX_OUTPUT_TOKENS = 32_000

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

export const claudeExecutable = resolveClaudeExecutable()

// ---------------------------------------------------------------------------------------
// Model mapping
// ---------------------------------------------------------------------------------------

export function aliasForModel(model: string): "sonnet" | "opus" | "haiku" {
  if (model.includes("opus")) return "opus"
  if (model.includes("haiku")) return "haiku"
  return "sonnet"
}

/** Exact `claude-*` ids are passed through; anything else falls back to the CLI alias. */
export function resolveModel(model: unknown, forceAlias = false): string {
  const name = typeof model === "string" ? model : ""
  if (!name) return "sonnet"
  if (!forceAlias && process.env.CLAUDE_PROXY_MODEL_PASSTHROUGH !== "0" && /^claude-[a-z0-9._-]+$/i.test(name)) {
    return name
  }
  return aliasForModel(name)
}

/**
 * True when two resolved model strings share a prompt cache. A dated snapshot ("claude-x-20250929"),
 * its undated id and "-latest" are one model, and a bare CLI alias stands for its family. Different
 * versions or families stay different: those are real switches. Comparing with `!==` treated an exact
 * id and its alias (e.g. after the alias fallback) as a switch, which compacted and threw away a
 * perfectly good cached transcript.
 */
export function sameModel(a: string, b: string): boolean {
  const normalize = (m: string) => m.toLowerCase().replace(/-(\d{8}|latest)$/, "")
  const isAlias = (m: string) => m === "sonnet" || m === "opus" || m === "haiku"
  const x = normalize(a)
  const y = normalize(b)
  if (x === y) return true
  if (isAlias(x) || isAlias(y)) return aliasForModel(x) === aliasForModel(y)
  return false
}

/**
 * Maps the Anthropic `thinking` parameter to the CLI's `maxThinkingTokens`. Returning `undefined`
 * leaves the CLI's own default untouched; only an explicit `disabled` (0) or `enabled` budget
 * overrides it. Without this the caller's thinking intent was simply dropped and the CLI's default
 * budget applied to every turn.
 */
export function thinkingBudget(thinking: unknown): number | undefined {
  if (!thinking || typeof thinking !== "object") return undefined
  const t = thinking as { type?: unknown; budget_tokens?: unknown }
  if (t.type === "disabled") return 0
  if (t.type === "enabled") {
    const n = Number(t.budget_tokens)
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined
  }
  return undefined
}
