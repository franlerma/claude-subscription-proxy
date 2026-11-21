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
