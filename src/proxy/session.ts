import type { Context } from "hono"
import { createHash, randomUUID } from "crypto"
import { join } from "path"
import { homedir } from "os"
import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  rmdirSync,
  statSync,
  symlinkSync,
  writeFileSync
} from "fs"
import { claudeLog } from "../logger"
import { renderBlocks } from "./prompt"
import {
  SESSION_MODE,
  SESSION_TTL_MS,
  SESSION_MAX,
  SESSION_FLUSH_MS,
  SESSION_CONFIG_DIR,
  SESSION_FINGERPRINT_MESSAGES
} from "./env"

/**
 * Everything the proxy knows about one caller conversation. `messageCount` describes where the
 * assistant turn the transcript holds should sit in the caller's history on the next request, which
 * is what lets a later request prove that resuming is safe.
 */
export interface SessionState {
  id: string
  messageCount: number
  updatedAt: number
  /** Model the transcript was last served with, to detect a model switch mid-conversation. */
  model?: string
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
export const pendingTurns = new Map<string, Promise<void>>()
export const PENDING_TURN_TIMEOUT_MS = Math.max(60_000, SESSION_FLUSH_MS + 60_000)

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
export function sessionKey(messages: any[], system: string, c?: Context): string | null {
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
        updatedAt: parsed.updated,
        model: typeof parsed.model === "string" ? parsed.model : undefined
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
        updated: state.updatedAt,
        model: state.model ?? null
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
        updatedAt: entry.updated,
        model: typeof entry.model === "string" ? entry.model : undefined
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

export function deleteSession(key: string) {
  const state = sessions.get(key)
  if (state?.timer) clearTimeout(state.timer)
  sessions.delete(key)
  try {
    rmSync(sessionFilePath(key), { force: true })
  } catch {}
}

export function rememberSession(key: string, state: SessionState) {
  if (!state.id) return
  const previous = sessions.get(key)
  if (previous?.timer) clearTimeout(previous.timer)
  // Re-insert so Map order stays recency order: `set` on an existing key keeps its old position,
  // which would evict an active conversation before an idle one.
  sessions.delete(key)
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

export function resolveSessionForRequest(key: string | null, messages: any[]): ResumeDecision {
  if (!key || !SESSION_MODE) return { resume: null, state: null, resetReason: null }
  loadSessions()
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

export function startSessionSweeper() {
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
  loadSessions()
  const interval = setInterval(sweepSessions, Math.min(SESSION_TTL_MS, 300_000))
  interval.unref?.()
  sweepSessions()
}
