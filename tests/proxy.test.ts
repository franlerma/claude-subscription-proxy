import { describe, expect, test } from "bun:test"
import { assistantFingerprint, canonicalJson, canonicalizeSchema } from "../src/proxy/server"

describe("canonicalJson", () => {
  test("sorts object keys recursively", () => {
    expect(canonicalJson({ b: 1, a: { d: 2, c: 3 } })).toBe('{"a":{"c":3,"d":2},"b":1}')
  })

  test("keeps array order", () => {
    expect(canonicalJson([3, 1, 2])).toBe("[3,1,2]")
  })
})

describe("canonicalizeSchema", () => {
  test("sorts properties but not enums", () => {
    const schema = { type: "object", properties: { z: { type: "string" }, a: { enum: ["z", "a"] } } }
    expect(canonicalizeSchema(schema)).toEqual({
      properties: { a: { enum: ["z", "a"] }, z: { type: "string" } },
      type: "object"
    })
  })
})

describe("assistantFingerprint", () => {
  test("is stable across JSON key order and ignored blocks", () => {
    const a = [
      { type: "thinking", thinking: "secret" },
      { type: "tool_use", id: "t1", name: "read", input: { z: 1, a: { y: 2, x: 3 } }, cache_control: { type: "ephemeral" } }
    ]
    const b = [
      { type: "tool_use", name: "read", id: "t1", input: { a: { x: 3, y: 2 }, z: 1 } },
      { type: "redacted_thinking", data: "other" }
    ]
    expect(assistantFingerprint(a)).toBe(assistantFingerprint(b))
  })

  test("changes when tool input changes", () => {
    const a = [{ type: "tool_use", id: "t1", name: "read", input: { path: "a" } }]
    const b = [{ type: "tool_use", id: "t1", name: "read", input: { path: "b" } }]
    expect(assistantFingerprint(a)).not.toBe(assistantFingerprint(b))
  })
})
