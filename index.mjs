// Magpie plugin: OpenAI's Decisions API as a System One decision provider.
//
// Magpie's routing groups hold either conversation models or decision models (providers with `decide`),
// and a `decide` provider is asked System One at `<base>/systemone`. OpenAI speaks Decisions at
// `/v1/decisions`, so without this plugin `openai/gpt-6-luna` cannot sit in a group beside
// `typesafe/jev-latest`. The plugin answers that System One request through its own `fetch`, calls OpenAI
// and translates both ways, so a group `[openai-decide/gpt-6-luna, typesafe/jev-latest]` gives clients the
// unchanged `POST /v1/systemone` with the gateway's own quota/credit/outage fallback.
//
// Failures are passed on as failures: OpenAI's own 401/402/429/5xx keep their status, a timeout is 504 and
// an answer that is not a complete, valid System One answer is 502. Nothing is ever turned into `unknown`.

import { DecisionError, OPENAI_MODEL, fromDecisionResponse, toDecisionRequest } from "./translate.mjs";

export const PROVIDER = "openai-decide";
const BASE_URL = "https://api.openai.com/v1";
const DEFAULT_UPSTREAM = "https://api.openai.com/v1/decisions";
const DEFAULT_TIMEOUT_MS = 8_000;

const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
const failure = (status, type, message) => json(status, { error: { code: null, message, param: null, type } });

/** Sentinel used to keep the provider signed in (so a decision group stays a decision group)
 * while no real OpenAI key is configured. When the stored key is missing or equals this
 * placeholder, the plugin answers 503 locally without calling OpenAI, so the gateway falls
 * through to the next group member (typesafe) with zero upstream calls. */
export const PLACEHOLDER_KEY = "openai-decide-not-configured";
const isPlaceholder = (key) => !key || key === PLACEHOLDER_KEY;

export async function server(_input, options = {}) {
  const upstream = String(options.upstream ?? DEFAULT_UPSTREAM);
  const timeoutMs = Number.isFinite(Number(options.timeoutMs)) && Number(options.timeoutMs) > 0 ? Number(options.timeoutMs) : DEFAULT_TIMEOUT_MS;

  async function decide(request, key) {
    let body;
    let decisionRequest;
    try {
      body = await request.json();
      decisionRequest = toDecisionRequest(body);
    } catch (error) {
      if (error instanceof DecisionError) return failure(error.status, error.type, error.message);
      return failure(400, "invalid_request_error", "expected a JSON System One request");
    }
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(timeoutMs)].filter(Boolean));
    let response;
    try {
      response = await fetch(upstream, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
        body: JSON.stringify(decisionRequest),
        signal,
      });
    } catch (error) {
      if (request.signal?.aborted) throw error;
      if (signal.aborted) return failure(504, "timeout", `OpenAI Decisions did not answer within ${timeoutMs} ms`);
      return failure(502, "upstream_unreachable", `OpenAI Decisions is unreachable: ${error instanceof Error ? error.message : String(error)}`);
    }
    const text = await response.text();
    if (!response.ok) {
      // Keep OpenAI's status and body: the gateway tells quota, credit and key errors apart by them.
      return new Response(text, { status: response.status, headers: { "content-type": response.headers.get("content-type") ?? "application/json" } });
    }
    try {
      return json(200, fromDecisionResponse(JSON.parse(text), body));
    } catch (error) {
      if (error instanceof DecisionError) return failure(error.status, error.type, error.message);
      return failure(502, "invalid_response", "OpenAI Decisions answered with something that is not JSON");
    }
  }

  const models = { [OPENAI_MODEL]: { name: "GPT-6 Luna (Decisions)", decides: true } };

  return {
    async config(cfg) {
      cfg.provider ??= {};
      cfg.provider[PROVIDER] ??= { name: "OpenAI Decisions", npm: "@ai-sdk/openai-compatible", api: BASE_URL, decide: true, models };
    },
    auth: {
      provider: PROVIDER,
      single: true,
      methods: [{ type: "api", label: "OpenAI API key (Decisions)" }],
      async loader(getAuth) {
        const auth = await getAuth();
        if (!auth || auth.type !== "api" || !auth.key) return {};
        return {
          baseURL: BASE_URL,
          apiKey: auth.key,
          async fetch(input, init) {
            const request = new Request(input, init);
            const path = new URL(request.url).pathname;
            if (request.method !== "POST" || !path.endsWith("/systemone")) return failure(404, "not_found_error", "only POST /systemone is served");
            const current = await getAuth();
            if (!current || current.type !== "api" || !current.key) return failure(401, "authentication_error", "no OpenAI API key is signed in");
            if (isPlaceholder(current.key)) return failure(503, "not_configured", "openai-decide has no OpenAI key configured; falling through to the next group member");
            return decide(request, current.key);
          },
        };
      },
    },
    provider: {
      id: PROVIDER,
      async models(provider) {
        return { ...provider.models, ...models };
      },
    },
  };
}

export const OpenAIDecisionsPlugin = server;
export default { id: PROVIDER, server };
