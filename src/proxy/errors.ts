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

// A bare "429" is not enough: stderr and error text routinely contain digits (ids, counters, traces),
// and a false positive makes the caller back off and retry a failure that was never a rate limit.
// Only count 429 when it is written as a status ("API Error: 429", "HTTP 429", "status code 429").
const RATE_LIMIT_PATTERN =
  /rate.?limit|usage limit|hit your limit|too many requests|overloaded|(?:api error|http|status(?: code)?)\W{0,4}429\b/i

/** True when CLI/upstream text reports a rate limit, quota, or overload (maps to HTTP 429). */
export function isRateLimitText(text: string): boolean {
  return RATE_LIMIT_PATTERN.test(text)
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
