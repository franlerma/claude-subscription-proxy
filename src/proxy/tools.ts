import { createSdkMcpServer, tool } from "@anthropic-ai/claude-agent-sdk"
import { z } from "zod"

export const CLIENT_MCP_SERVER = "client"
export const CLIENT_TOOL_PREFIX = `mcp__${CLIENT_MCP_SERVER}__`

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

export interface ClientTools {
  server: ReturnType<typeof createSdkMcpServer>
  allowed: string[]
  nameToAlias: Map<string, string>
  aliasToName: Map<string, string>
}

/**
 * Dedupe by name, then sort by code point. The tool list is the first part of the API request:
 * reordering it invalidates the entire prompt cache even though the definitions are unchanged.
 */
export function buildClientTools(tools: Array<{ name: string; description?: string; input_schema?: any }>): ClientTools {
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

export function originalToolName(name: string, client: ClientTools | null): string {
  if (!name.startsWith(CLIENT_TOOL_PREFIX)) return name
  const alias = name.slice(CLIENT_TOOL_PREFIX.length)
  return client?.aliasToName.get(alias) ?? alias
}
