export type StreamEvent = { type: string; [key: string]: any }

/** Error that carries the HTTP status / Anthropic error type to return to the caller. */
export class ProxyError extends Error {
  constructor(
    message: string,
    readonly status: number = 500,
    readonly errorType: string = "api_error"
  ) {
    super(message)
  }
}

/** Turns the CLI's "API Error: 400 {json}" text into a ProxyError with the upstream status and message. */
export function upstreamError(status: number, rest: string): ProxyError {
  try {
    const parsed = JSON.parse(rest)
    const message = parsed?.error?.message ?? rest
    return new ProxyError(String(message), status, parsed?.error?.type ?? "api_error")
  } catch {
    return new ProxyError(rest || `Upstream error ${status}`, status)
  }
}
