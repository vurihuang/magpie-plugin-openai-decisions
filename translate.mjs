// System One <-> OpenAI Decisions translation. Pure functions, no I/O.
//
// System One (TypeSafe Jev) request:  { model, state, questions: { <id>: { type: "choice"|"noul"|"score", instructions, criteria } } }
// OpenAI Decisions request:           { model, input, questions: [{ type: "choice"|"predicate"|"score", name, instructions, choices|levels }] }
//
// The answer must stay a valid System One answer for any client that checks it: every criterion
// has a probability, the probabilities sum to 1 (±0.02) and the chosen one leads. Anything else is a
// technical failure (`DecisionError`), never an answer, so the gateway can fall back to another model.

export const OPENAI_MODEL = "gpt-6-luna";

export class DecisionError extends Error {
  constructor(status, type, message) {
    super(message);
    this.status = status;
    this.type = type;
  }
}

const isRecord = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const isProbability = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;

// Object keys sorted at every level, as magpie's own System One -> Decisions route sends them: the same
// evidence then gets the same input (and the same answer) whatever order its keys were built in.
const sortKeys = (v) => (Array.isArray(v) ? v.map(sortKeys) : isRecord(v) ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v);

/** state -> OpenAI `input`: text stays text, JSON travels as an `input_json` part. */
export function toInput(state) {
  if (typeof state === "string") return state;
  const jsonData = isRecord(state) || (Array.isArray(state) && state.every(isRecord));
  if (!jsonData) return JSON.stringify(state ?? null);
  return [{ role: "user", content: [{ type: "input_json", data: sortKeys(state) }] }];
}

export function toDecisionRequest(body) {
  if (!isRecord(body) || !isRecord(body.questions)) throw new DecisionError(400, "invalid_request_error", "questions must be an object");
  const ids = Object.keys(body.questions);
  if (ids.length < 1) throw new DecisionError(400, "invalid_request_error", "at least one question is required");
  const questions = ids.map((id) => {
    const q = body.questions[id];
    if (!isRecord(q) || typeof q.instructions !== "string") throw new DecisionError(400, "invalid_request_error", `question ${id} needs instructions`);
    const type = q.type ?? "choice";
    if (type === "noul") return { type: "predicate", name: id, instructions: q.instructions };
    if (type === "choice") {
      if (!isRecord(q.criteria) || Object.keys(q.criteria).length < 2) throw new DecisionError(400, "invalid_request_error", `question ${id} needs at least two criteria`);
      return { type: "choice", name: id, instructions: q.instructions, choices: Object.entries(q.criteria).map(([value, description]) => ({ value, ...(typeof description === "string" && description ? { description } : {}) })) };
    }
    if (type === "score") {
      if (!Array.isArray(q.criteria) || q.criteria.length < 2) throw new DecisionError(400, "invalid_request_error", `question ${id} needs at least two levels`);
      return { type: "score", name: id, instructions: q.instructions, levels: q.criteria.map((label) => ({ label: String(label) })) };
    }
    throw new DecisionError(400, "invalid_request_error", `question ${id} has unsupported type ${String(type)}`);
  });
  return { model: OPENAI_MODEL, input: toInput(body.state), questions };
}

/** OpenAI Decision -> System One response for the request that produced it. */
export function fromDecisionResponse(decision, body) {
  const bad = (why) => new DecisionError(502, "invalid_response", `OpenAI Decisions answer rejected: ${why}`);
  const ids = Object.keys(body.questions);
  if (!isRecord(decision) || !Array.isArray(decision.answers) || decision.answers.length !== ids.length) throw bad("answer count differs from question count");
  const answers = {};
  for (const [n, id] of ids.entries()) {
    const a = decision.answers[n];
    const q = body.questions[id];
    const type = q.type ?? "choice";
    if (!isRecord(a)) throw bad(`answer ${id} is not an object`);
    if (a.type === "refusal") throw bad(`question ${id} was refused`);
    if (a.name !== undefined && a.name !== null && a.name !== id) throw bad(`answer ${n} is for ${String(a.name)}, not ${id}`);
    if (type === "noul") {
      if (a.type !== "predicate" || !isProbability(a.probability)) throw bad(`answer ${id} is not a probability`);
      answers[id] = { type: "noul", noul: a.probability };
    } else if (type === "choice") {
      const keys = Object.keys(q.criteria);
      if (a.type !== "choice" || typeof a.choice !== "string" || !keys.includes(a.choice) || !Array.isArray(a.probabilities) || !isProbability(a.confidence)) throw bad(`answer ${id} is not a listed choice`);
      const probabilities = {};
      for (const p of a.probabilities) {
        if (!isRecord(p) || typeof p.value !== "string" || !keys.includes(p.value) || !isProbability(p.probability) || p.value in probabilities) throw bad(`answer ${id} has an unlisted or repeated probability`);
        probabilities[p.value] = p.probability;
      }
      const total = Object.values(probabilities).reduce((s, v) => s + v, 0);
      if (Object.keys(probabilities).length !== keys.length || Math.abs(total - 1) > 0.02 || Object.values(probabilities).some((v) => v > probabilities[a.choice] + 0.001)) throw bad(`answer ${id} probabilities do not describe its choice`);
      answers[id] = { type: "choice", choice: a.choice, confidence: a.confidence, probabilities };
    } else {
      const levels = q.criteria;
      if (a.type !== "score" || !Number.isFinite(a.score) || !isProbability(a.confidence) || !Array.isArray(a.probabilities) || a.probabilities.length !== levels.length) throw bad(`answer ${id} is not a score`);
      const probabilities = {};
      for (const p of a.probabilities) {
        if (!isRecord(p) || !Number.isInteger(p.value) || p.value < 0 || p.value >= levels.length || !isProbability(p.probability) || String(p.value) in probabilities) throw bad(`answer ${id} has an invalid level`);
        probabilities[String(p.value)] = p.probability;
      }
      answers[id] = { type: "score", score: a.score, confidence: a.confidence, legend: Object.fromEntries(levels.map((label, i) => [String(i), String(label)])), probabilities };
    }
  }
  const usage = decision.usage;
  if (!isRecord(usage) || !Number.isInteger(usage.input_tokens) || !Number.isInteger(usage.output_tokens) || typeof decision.model !== "string") throw bad("usage or model is missing");
  return { model: decision.model, answers, usage: { input_tokens: usage.input_tokens, output_tokens: usage.output_tokens } };
}
