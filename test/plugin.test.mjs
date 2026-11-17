import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { server } from "../index.mjs";
import { fromDecisionResponse, toDecisionRequest } from "../translate.mjs";

const systemOne = {
  model: "typesafe/jev-latest",
  state: { company: "Acme Pumps GmbH", text: "Manufacturer of centrifugal pumps." },
  questions: {
    kind: { type: "choice", instructions: "Kind?", criteria: { manufacturer: "makes goods", service: "sells services", unknown: "unclear" } },
    fit: { type: "noul", instructions: "Is it a manufacturer?" },
    depth: { type: "score", instructions: "How deep?", criteria: ["none", "weak", "strong"] },
  },
};

const usage = { input_tokens: 40, input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 }, output_tokens: 0, output_tokens_details: { reasoning_tokens: 0 }, total_tokens: 40 };
const okDecision = {
  model: "gpt-6-luna",
  answers: [
    { type: "choice", name: "kind", choice: "manufacturer", confidence: 0.97, probabilities: [{ value: "manufacturer", probability: 0.97 }, { value: "service", probability: 0.02 }, { value: "unknown", probability: 0.01 }] },
    { type: "predicate", name: "fit", probability: 0.91 },
    { type: "score", name: "depth", score: 1.9, confidence: 0.9, probabilities: [{ label: "none", probability: 0.01, value: 0 }, { label: "weak", probability: 0.09, value: 1 }, { label: "strong", probability: 0.9, value: 2 }] },
  ],
  usage,
};

describe("translation", () => {
  test("System One request becomes an OpenAI decision request in question order", () => {
    const request = toDecisionRequest(systemOne);
    expect(request.model).toBe("gpt-6-luna");
    expect(request.input).toEqual([{ role: "user", content: [{ type: "input_json", data: systemOne.state }] }]);
    expect(request.questions.map((q) => [q.name, q.type])).toEqual([["kind", "choice"], ["fit", "predicate"], ["depth", "score"]]);
    expect(request.questions[0].choices).toEqual([{ value: "manufacturer", description: "makes goods" }, { value: "service", description: "sells services" }, { value: "unknown", description: "unclear" }]);
    expect(request.questions[2].levels).toEqual([{ label: "none" }, { label: "weak" }, { label: "strong" }]);
  });

  test("JSON state is sent with sorted keys, so key order never changes the input", () => {
    const a = toDecisionRequest({ ...systemOne, state: { b: 1, a: { d: [{ y: 1, x: 2 }], c: 3 } } }).input;
    const b = toDecisionRequest({ ...systemOne, state: { a: { c: 3, d: [{ x: 2, y: 1 }] }, b: 1 } }).input;
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    expect(JSON.stringify(a[0].content[0].data)).toBe('{"a":{"c":3,"d":[{"x":2,"y":1}]},"b":1}');
  });

  test("text state stays text and unusable questions are rejected before any call", () => {
    expect(toDecisionRequest({ ...systemOne, state: "plain" }).input).toBe("plain");
    expect(toDecisionRequest({ ...systemOne, state: 7 }).input).toBe("7");
    expect(() => toDecisionRequest({ ...systemOne, questions: { q: { type: "choice", instructions: "x", criteria: { only: "one" } } } })).toThrow("two criteria");
    expect(() => toDecisionRequest({ ...systemOne, questions: {} })).toThrow("at least one");
  });

  test("the answer keeps the System One shape clients validate", () => {
    const response = fromDecisionResponse(okDecision, systemOne);
    expect(response.model).toBe("gpt-6-luna");
    expect(response.usage).toEqual({ input_tokens: 40, output_tokens: 0 });
    expect(response.answers.fit).toEqual({ type: "noul", noul: 0.91 });
    expect(response.answers.depth.legend).toEqual({ "0": "none", "1": "weak", "2": "strong" });
    // The System One contract a client relies on: one answer per question, every criterion has a
    // probability, the probabilities sum to 1 (±0.02) and the chosen one leads.
    const kind = response.answers.kind;
    expect(kind.choice).toBe("manufacturer");
    expect(Object.keys(kind.probabilities).sort()).toEqual(["manufacturer", "service", "unknown"]);
    expect(Math.abs(Object.values(kind.probabilities).reduce((sum, p) => sum + p, 0) - 1)).toBeLessThanOrEqual(0.02);
    expect(Math.max(...Object.values(kind.probabilities))).toBe(kind.probabilities[kind.choice]);
  });

  test("an incomplete or inconsistent answer is a failure, never an answer", () => {
    const withAnswers = (answers) => ({ ...okDecision, answers });
    expect(() => fromDecisionResponse(withAnswers(okDecision.answers.slice(0, 2)), systemOne)).toThrow("count");
    expect(() => fromDecisionResponse(withAnswers([{ type: "refusal", name: "kind" }, ...okDecision.answers.slice(1)]), systemOne)).toThrow("refused");
    expect(() => fromDecisionResponse(withAnswers([{ ...okDecision.answers[0], choice: "bogus" }, ...okDecision.answers.slice(1)]), systemOne)).toThrow("listed choice");
    expect(() => fromDecisionResponse(withAnswers([{ ...okDecision.answers[0], probabilities: [{ value: "manufacturer", probability: 0.5 }, { value: "service", probability: 0.2 }, { value: "unknown", probability: 0.1 }] }, ...okDecision.answers.slice(1)]), systemOne)).toThrow("probabilities");
    expect(() => fromDecisionResponse({ ...okDecision, usage: undefined }, systemOne)).toThrow("usage");
    expect(() => fromDecisionResponse(withAnswers([okDecision.answers[1], okDecision.answers[0], okDecision.answers[2]]), systemOne)).toThrow();
  });
});

describe("plugin fetch", () => {
  let mode = "ok";
  let seen = null;
  let upstream;
  beforeAll(() => {
    upstream = Bun.serve({
      port: 0,
      async fetch(request) {
        seen = { authorization: request.headers.get("authorization"), body: await request.json() };
        if (mode === "quota") return Response.json({ error: { message: "You exceeded your current quota", type: "insufficient_quota", code: "insufficient_quota" } }, { status: 429 });
        if (mode === "auth") return Response.json({ error: { message: "Incorrect API key provided", type: "invalid_request_error", code: "invalid_api_key" } }, { status: 401 });
        if (mode === "garbage") return new Response("<html>oops</html>", { status: 200 });
        if (mode === "empty") return Response.json({ ...okDecision, answers: [] });
        if (mode === "hang") { await Bun.sleep(1500); return Response.json(okDecision); }
        return Response.json(okDecision);
      },
    });
  });
  afterAll(() => upstream.stop(true));

  async function fetcher(timeoutMs = 8000, key = "sk-test") {
    const hooks = await server({}, { upstream: `http://127.0.0.1:${upstream.port}/v1/decisions`, timeoutMs });
    const loaded = await hooks.auth.loader(async () => (key ? { type: "api", key } : undefined));
    return { hooks, loaded };
  }
  const call = (loaded, body = systemOne, path = "/v1/systemone") =>
    loaded.fetch(`https://api.openai.com${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

  test("registers a System One decision provider with its model", async () => {
    const { hooks } = await fetcher();
    const cfg = {};
    await hooks.config(cfg);
    expect(cfg.provider?.["openai-decide"]).toMatchObject({ decide: true, models: { "gpt-6-luna": { decides: true } } });
    expect(hooks.auth.provider).toBe("openai-decide");
  });

  test("answers in System One shape and sends the key and the decision request upstream", async () => {
    mode = "ok";
    const { loaded } = await fetcher();
    const response = await call(loaded);
    expect(response.status).toBe(200);
    expect((await response.json()).answers.kind).toMatchObject({ choice: "manufacturer" });
    expect(seen?.authorization).toBe("Bearer sk-test");
    expect(seen?.body.model).toBe("gpt-6-luna");
    expect(seen?.body.questions).toHaveLength(3);
  });

  test("OpenAI's own failures keep their status so the gateway can fall back", async () => {
    const { loaded } = await fetcher();
    mode = "quota";
    const quota = await call(loaded);
    expect(quota.status).toBe(429);
    expect(JSON.stringify(await quota.json())).toContain("insufficient_quota");
    mode = "auth";
    expect((await call(loaded)).status).toBe(401);
  });

  test("garbage, empty answers and a hung upstream are failures, not answers", async () => {
    mode = "garbage";
    expect((await call((await fetcher()).loaded)).status).toBe(502);
    mode = "empty";
    expect((await call((await fetcher()).loaded)).status).toBe(502);
    mode = "hang";
    const started = Date.now();
    const hung = await call((await fetcher(200)).loaded);
    expect(hung.status).toBe(504);
    expect(Date.now() - started).toBeLessThan(1200);
  });

  test("other paths, bad requests and a missing key are refused without calling OpenAI", async () => {
    mode = "ok";
    seen = null;
    const { loaded } = await fetcher();
    expect((await call(loaded, systemOne, "/v1/models")).status).toBe(404);
    expect((await call(loaded, { questions: {} })).status).toBe(400);
    expect(seen).toBeNull();
    const none = await fetcher(8000, null);
    expect(none.loaded).toEqual({});
  });
});

describe("placeholder key", () => {
  test("a placeholder or empty key returns 503 without any upstream call", async () => {
    let calls = 0;
    const auth = { type: "api", key: "openai-decide-not-configured" };
    const srv = await server({}, {});
    const loaded = await srv.auth.loader(async () => auth);
    const res = await loaded.fetch("https://api.openai.com/v1/systemone", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "gpt-6-luna", state: { c: "x" }, questions: { k: { type: "choice", instructions: "i", criteria: { a: "a", b: "b" } } } }),
    });
    expect(res.status).toBe(503);
    expect(calls).toBe(0);
  });
});
