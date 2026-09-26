# 02.03 — Structured Outputs & Schemas

> **Module 2: Prompting & Context** · Subtopic 3 of 4
> **Prerequisites:** 02.01 (prompt anatomy), JSON Schema, Pydantic (Python) and/or Jackson + Bean Validation (Java), 01.02 §6 (the idea of token masking).
> **Outcome:** you design schemas that make models *more accurate* (not just parseable), use each provider's strict-output features correctly, and build a validation-and-repair layer that turns "usually valid JSON" into a typed contract your services can depend on. This covers streaming, evaluation, and schema evolution.

> **Scope split with 02.04:** this subtopic is the **application and API layer** (schemas, providers, validation, evolution). 02.04 is the **decoding engine** underneath (automata, grammars, masks, distortion).

---

## 1. The Reliability Ladder

```
 GUARANTEE ▲
           │  5. Strict schema (constrained decoding) + semantic validation + repair
           │  4. Strict schema (constrained decoding)         ── syntactically valid by construction*
           │  3. Tool/function calling (non-strict)            ── usually valid, can miss fields/types
           │  2. "JSON mode"                                   ── valid JSON, any shape
           │  1. Prompt-only ("return only JSON")              ── best effort
           └──────────────────────────────────────────────────►  EFFORT / COUPLING TO PROVIDER
   * except refusals and max_tokens truncation — always check the stop reason
```

**Syntactic validity ≠ correctness.** Constrained decoding guarantees the output *parses and matches types*. It does not guarantee that `effective_date` is the right date, that `sponsor` exists, or that `end_date ≥ start_date`. Level 5 — adding a semantic validation layer — is the production target.

---

## 2. Provider APIs (verified September 2026 — re-check before building)

### 2.1 Anthropic (Claude API)

- **JSON outputs:** `output_config={"format": {"type": "json_schema", "schema": {...}}}`. The older `output_format` parameter is deprecated, and no beta header is needed anymore.
- **Strict tool use:** add `"strict": true` to a tool definition, and its `input` is guaranteed to match `input_schema`.
- **SDK helper:** `client.messages.parse(..., output_format=PydanticModel)` → `response.parsed_output`. The Java SDK accepts plain classes through `outputConfig(Class<T>)`.
- **Schema subset:** no `minimum`/`maximum`/`minLength`/`maxLength`/`pattern` and no recursive schemas. The SDKs move unsupported constraints into descriptions and validate them client-side. `additionalProperties: false` is required on objects (the SDKs add it). **Required properties are generated before optional ones**, regardless of declaration order.
- **Caveats:**
  - The first request with a new schema pays **grammar-compilation latency**. Compiled grammars are cached for 24 h from last use; changing only names or descriptions doesn't invalidate them.
  - Changing the format invalidates the **prompt cache**.
  - Enum and const **casing is not guaranteed**, so compare case-insensitively.
  - Complexity limits apply: at most 20 strict tools per request, 24 optional parameters, and 16 union-typed parameters in total.
  - Structured outputs are incompatible with citations and with message prefilling.
  - `stop_reason: "refusal"` or `"max_tokens"` can yield output that doesn't match the schema.

```python
from pydantic import BaseModel, Field
from anthropic import Anthropic


class BillSummary(BaseModel):
    evidence: list[str] = Field(description="Verbatim quotes supporting the fields below")
    bill_id: str = Field(description="Identifier like HB-123 or SB-45")
    amended_sections: list[str] = Field(description="Code sections amended, e.g. '2-18-303'")
    effective_date: str | None = Field(description="ISO date, or null if not stated")


client = Anthropic()
resp = client.messages.parse(
    model="claude-opus-5-5",
    max_tokens=2048,
    messages=[{"role": "user", "content": f"Extract the bill metadata.\n<bill>{bill_text}</bill>"}],
    output_format=BillSummary,
)
if resp.stop_reason in ("refusal", "max_tokens"):
    raise RuntimeError(f"unusable structured output: {resp.stop_reason}")
summary: BillSummary = resp.parsed_output
```

### 2.2 OpenAI

- **Responses API:** `text={"format": {"type": "json_schema", "name": "...", "schema": {...}, "strict": True}}`. **Chat Completions:** `response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}`.
- **SDK helper:** `client.responses.parse(model=..., input=..., text_format=PydanticModel)` → `response.output_parsed`.
- **Schema rules in strict mode:**
  - **Every property must be listed in `required`.** Express optional values as a union with `null`, e.g. `"type": ["string", "null"]`.
  - `additionalProperties: false` is mandatory.
  - Recursive schemas via `$ref` are supported.
  - Nesting-depth and size limits apply.
- **Refusals** are surfaced as a separate refusal item or field. Check for them before parsing.

### 2.3 Open models (self-hosted)

vLLM, SGLang, TensorRT-LLM, and llama.cpp accept a JSON Schema, regex, or grammar through their OpenAI-compatible `response_format` and engine-specific structured-output options. Backends include XGrammar, llguidance, and Outlines (02.04). Supported JSON Schema keywords **differ by backend**, so test your exact schema on your exact engine version. Many open backends *do* support `pattern` and numeric bounds.

### 2.4 Portability rule

Write schemas in the **intersection** of what your providers support: objects, arrays, enums, string/number/boolean/null, required fields, `additionalProperties: false`, and shallow nesting. Enforce everything else (ranges, regex, cross-field rules) in your **own validation layer** (§4), which is provider-independent and testable.

---

## 3. Schema Design for LLMs

A schema is a prompt. It controls **what** the model generates, **in what order**, and **with what vocabulary**.

### 3.1 Field order is generation order

Autoregressive generation factorises the output left to right. For a reasoning field $r$ and a decision field $a$:

$$
\underbrace{p(r)\,p(a \mid r)}_{\texttt{"reasoning"}\ \text{first}}
\qquad\text{vs.}\qquad
\underbrace{p(a)\,p(r \mid a)}_{\texttt{"answer"}\ \text{first}}
$$

With the answer first, the "reasoning" becomes a **post-hoc rationalisation** of a choice already made with no intermediate computation. Put `evidence` or `analysis` fields *before* verdicts, scores, and classifications. On Anthropic, required fields are emitted before optional ones, so make the reasoning field **required**.

**Caveat:** heavy format constraints can reduce reasoning quality (Tam et al., 2024). For hard reasoning, either let the model reason freely (or use a reasoning model's hidden thinking) and then produce the structured output, or use a two-step chain (reason → extract).

### 3.2 Design rules

| Rule | Why | Example |
|---|---|---|
| **Descriptions are instructions** | The model reads them | `"effective_date": {"description": "ISO-8601; null if not stated; never infer from session year"}` |
| **Enums over free text** | Closed vocabulary → no normalisation bugs | `"status": {"enum": ["introduced", "in_committee", "passed", "vetoed"]}` |
| **Explicit unknowns** | Stops fabrication | `["string", "null"]` + a description saying *when* to use null |
| **Distinguish null from absent** | "Not stated" ≠ "not applicable" | an `"applicability": "not_applicable"` enum rather than overloading null |
| **Evidence fields** | Grounding + auditability | `evidence: [{quote, location}]` before the extracted value |
| **Flat over deep** | Deep nesting raises error rates and grammar complexity | ≤ 3–4 levels |
| **Bounded arrays** | Prevents runaway lists | describe the max count; enforce `maxItems` in validation |
| **Discriminated unions** | Heterogeneous outputs parse deterministically | `{"type": "amendment", ...} \| {"type": "repeal", ...}` with a `type` const |
| **IDs, not copies** | Reference retrieved items by ID (02.02) | `"cited_ids": ["HB-123:s3"]` |
| **Semantic names** | `sponsor_last_name` beats `field_7` | the name is part of the prompt |
| **Numbers as numbers** | Avoids locale/format ambiguity | `"amount_usd": {"type": "number"}`; spell out the unit in the name |

### 3.3 A schema built with these rules

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["analysis", "classification", "confidence", "cited_ids"],
  "properties": {
    "analysis":       {"type": "string", "description": "2-4 sentences of reasoning grounded in the cited items. Written BEFORE the classification."},
    "classification": {"type": "string", "enum": ["fiscal", "criminal", "education", "health", "other"]},
    "confidence":     {"type": "string", "enum": ["high", "medium", "low"], "description": "low if evidence is indirect or conflicting"},
    "cited_ids":      {"type": "array", "items": {"type": "string"}, "description": "IDs of <item>s used; must be non-empty unless classification is 'other'"}
  }
}
```

A categorical confidence field is used instead of a float. Models are poorly calibrated when emitting numeric probabilities as text.

---

## 4. The Validation-and-Repair Layer

### 4.1 Architecture

```
 prompt + schema ─► LLM (strict mode) ─► stop_reason check ─► parse ─► SCHEMA validate ─► SEMANTIC validate ─► OK
                         ▲                     │ refusal/           │ fail           │ fail            │ fail
                         │                     ▼ truncation         ▼                ▼                 ▼
                         └──────── repair prompt: original output + precise errors (≤ N retries) ◄──────┘
                                                                                    │ still failing
                                                                                    ▼
                                                                   fallback: stronger model / human queue / typed error
```

**Semantic validators** encode business rules that JSON Schema can't express or that providers don't support:
- **Cross-field rules:** `end_date ≥ start_date`; totals equal the sum of their parts.
- **Referential integrity:** `cited_ids ⊆ manifest IDs`; the bill ID exists in the database.
- **Grounding:** every `evidence.quote` is a verbatim substring of the source (after whitespace normalisation).
- **Ranges and patterns** unsupported by the provider.

### 4.2 Reference implementation (provider-agnostic)

```python
import json
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    def __init__(self, msg: str, attempts: list[dict]):
        super().__init__(msg)
        self.attempts = attempts


def structured_call(
    llm: Callable[[list[dict]], tuple[str, str]],     # messages -> (text, stop_reason)
    messages: list[dict],
    model: type[T],
    semantic_checks: list[Callable[[T], str | None]] = (),
    max_repairs: int = 2,
) -> T:
    attempts, msgs = [], list(messages)
    for attempt in range(max_repairs + 1):
        text, stop = llm(msgs)
        if stop in ("refusal", "max_tokens"):
            raise StructuredOutputError(f"unrecoverable stop_reason={stop}", attempts)
        errors: list[str] = []
        obj = None
        try:
            obj = model.model_validate_json(text)
        except ValidationError as e:
            errors = [f"{'/'.join(map(str, er['loc'])) or '<root>'}: {er['msg']}" for er in e.errors()]
        if obj is not None:
            errors = [m for chk in semantic_checks if (m := chk(obj))]
            if not errors:
                return obj
        attempts.append({"attempt": attempt, "output": text, "errors": errors})
        msgs = msgs + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Your output failed validation:\n- " + "\n- ".join(errors)
             + "\nReturn the corrected JSON only. Keep all correct fields unchanged."},
        ]
    raise StructuredOutputError("validation failed after repairs", attempts)


def quotes_are_verbatim(source: str) -> Callable[[BaseModel], str | None]:
    norm = lambda s: " ".join(s.split()).lower()
    src = norm(source)

    def check(obj) -> str | None:
        bad = [q for q in getattr(obj, "evidence", []) if norm(q) not in src]
        return f"evidence quotes not found verbatim in source: {bad[:3]}" if bad else None
    return check
```

**Repair prompts must be precise.** Send JSON-pointer paths and the rule that failed, not "invalid output, try again". Log every attempt: the repair rate is a key quality metric, and a rising repair rate is an early warning of prompt or model drift.

**Libraries:** **Instructor** (Python/TS/Go/…) wraps this pattern around most providers. **Pydantic AI** and **Outlines** offer typed outputs. On the Java side, see §5.

---

## 5. Structured Outputs in Java / Spring

The same contract in a JVM service: records as schemas, Bean Validation for semantic rules, and a JSON Schema generated from the type so there is a single source of truth.

```java
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import jakarta.validation.constraints.*;
import java.util.List;

@JsonPropertyOrder({"evidence", "billId", "amendedSections", "effectiveDate"})   // generation order
public record BillSummary(
    @NotEmpty
    @JsonPropertyDescription("Verbatim quotes that support the fields below")
    List<String> evidence,

    @Pattern(regexp = "^(HB|SB|HJ|SJ)-\\d{1,4}$")
    @JsonPropertyDescription("Bill identifier like HB-123")
    String billId,

    @JsonPropertyDescription("Code sections amended, e.g. 2-18-303")
    List<@Pattern(regexp = "^\\d+-\\d+-\\d+$") String> amendedSections,

    @JsonPropertyDescription("ISO-8601 date, or null if not stated in the text")
    String effectiveDate
) {}
```

- **Schema generation:** `com.github.victools:jsonschema-generator` (with its Jackson module) produces the JSON Schema from the record. Provider-unsupported keywords such as `pattern` stay enforced by Bean Validation after parsing.
- **Spring AI:** `chatClient.prompt().user(u -> u.text(template).param("bill", text)).call().entity(BillSummary.class)` derives the format instructions from the type and deserialises the result. Where the underlying model supports native structured output, enable it in the model options so decoding is constrained rather than just prompted.
- **Anthropic Java SDK:** pass the class to `outputConfig(BillSummary.class)` on the message request builder to get constrained decoding and a typed result.
- **Validation:** run `jakarta.validation.Validator.validate(summary)` and, on violations, perform the repair loop from §4 with the violation messages (`propertyPath + message`).

> Pin the Spring AI and SDK versions and check their docs. APIs in this space have changed across minor releases.

---

## 6. Streaming Structured Output

UIs want progressive rendering, but a JSON prefix isn't valid JSON. The options are:

1. **Partial-JSON parsing:** repair the prefix into the best-effort object so far. `pydantic_core.from_json(buf, allow_partial=True)` and `partial-json-parser` do this. The reference below shows the idea.
2. **Streamed tool-input deltas:** providers stream tool-argument fragments that you accumulate and parse partially.
3. **Line-delimited records (JSONL):** for lists, instruct one object per line (or use a schema of `items`) and emit each item when its line closes. This is the simplest robust option.

```python
import json


def parse_partial_json(buf: str):
    """Best-effort parse of a truncated JSON prefix (educational; O(n^2) worst case).

    Closes open strings/containers and drops a trailing incomplete key/value.
    The last numeric value may be truncated (e.g. 12 of 123).
    """
    def close(s: str) -> str:
        stack, in_str, esc = [], False, False
        for ch in s:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        if in_str:
            s += "\\" if esc else ""
            s += '"'
        s = s.rstrip()
        while s and s[-1] in ",:":
            s = s[:-1].rstrip()
        return s + "".join(reversed(stack))

    for i in range(len(buf), 0, -1):
        try:
            return json.loads(close(buf[:i]))
        except json.JSONDecodeError:
            continue
    return None
```

**Streaming caveats:** validate only the *final* object. Never trigger side effects from partial values. Enum values can appear truncated mid-stream, so render them only once their field closes.

---

## 7. Evaluating Extraction Quality

Validity rate is necessary but insufficient. Measure **field-level accuracy**.

- **Scalar fields:** exact match after normalisation (dates to ISO, whitespace, case for enums).
- **List fields:** set-based precision and recall:

$$
P = \frac{|\hat S \cap S|}{|\hat S|},\quad R = \frac{|\hat S \cap S|}{|S|},\quad F_1 = \frac{2PR}{P+R}
$$

- **Null handling:** count "predicted value where gold is null" as a **hallucination**, and "predicted null where gold has a value" as an **omission**. Track these two separately, because they have different costs.
- **Grounding rate:** the fraction of evidence quotes found verbatim in the source.

```python
def field_scores(pred: dict, gold: dict, norm=lambda v: str(v).strip().lower()):
    """Per-field metrics. Lists → set P/R/F1; scalars → exact match with null accounting."""
    out = {}
    for key, g in gold.items():
        p = pred.get(key)
        if isinstance(g, list) or isinstance(p, list):
            ps, gs = {norm(x) for x in (p or [])}, {norm(x) for x in (g or [])}
            tp = len(ps & gs)
            prec = tp / len(ps) if ps else float(not gs)
            rec = tp / len(gs) if gs else float(not ps)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            out[key] = {"precision": prec, "recall": rec, "f1": f1}
        else:
            out[key] = {
                "exact": (p is None and g is None) or (p is not None and g is not None and norm(p) == norm(g)),
                "hallucination": g is None and p is not None,
                "omission": g is not None and p is None,
            }
    return out
```

### Long documents

For documents longer than the effective context (01.04), extract per chunk with the same schema. Then **merge** in code: union the lists, deduplicate entities by normalised key, and resolve scalar conflicts by evidence strength. Finally, optionally run a reconciliation call over the merged candidates. Keep chunk-level provenance on every value.

---

## 8. Schema Evolution and Governance

Structured outputs are **API contracts** between an LLM and your services. Govern them as such.

- **Version every schema** (`bill_summary.v3`) and include the version in logs and stored outputs.
- **Prefer additive changes.** Adding optional (nullable) fields is backward compatible. Renames, type changes, and enum removals are breaking.
- **Enum changes are prompt changes.** Adding an enum value changes model behaviour on *existing* inputs, so re-run the evals.
- **Store a schema registry** (a Git repo or a schema registry service). Generate language bindings (Pydantic, Java records, TypeScript) from one source.
- **Operational cost of changes:** each new schema triggers grammar compilation on first use and invalidates the prompt cache (§2.1). Roll changes out deliberately and warm them up.
- **Contract tests in CI:** golden inputs → the output must validate, and the field-level metrics must not regress.

---

## 9. Production Challenges & How to Solve Them

| Challenge | Symptom | Fix |
|---|---|---|
| **Truncated JSON** | Parse errors on long outputs | Check `stop_reason == max_tokens`; raise `max_tokens`; bound arrays; split into chunked extraction |
| **Refusals inside strict mode** | Output doesn't match the schema | Branch on the refusal stop reason; return a typed error, never a fabricated object |
| **Valid but wrong values** | High validity, low accuracy | Evidence-first fields; semantic validators; field-level evals; verifier pass |
| **Hallucinated optional fields** | Values where the gold is null | Explicit null semantics in descriptions; hallucination metric; grounding check |
| **Reasoning quality drops under strict format** | Lower accuracy than free-text | Reason → extract two-step chain; put analysis fields first; use reasoning models' thinking |
| **First-call latency spikes** | P99 TTFT jumps after deploys | Warm up new schemas; keep schemas stable; avoid per-request dynamic schemas |
| **Schema too complex** | 400 errors, compile timeouts | Flatten; split into multiple calls; reduce unions and optional fields |
| **Provider keyword gaps** | `pattern`/`minimum` silently ignored or rejected | Portable schema subset + your own validation layer (§2.4) |
| **Enum casing drift** | `"Passed"` vs `"passed"` | Case-insensitive comparison and normalisation in the parser |
| **Silent schema drift across services** | Consumers break | Registry + versioning + contract tests |

---

## 10. Hands-On Projects

### Project 1 — Grounded Document Extraction Pipeline with Field-Level Evaluation

**User stories**
- *As a legislative-services analyst*, I want bill metadata (ID, sponsors, amended code sections, effective date, appropriation amounts, fiscal-note flag) extracted with verbatim evidence, so that attorneys can verify each value in seconds.
- *As the engineering lead*, I want per-field accuracy, hallucination, and omission rates, so that we know which fields can be automated and which need review.

**Acceptance criteria**
1. The schema follows §3 (evidence-first, explicit nulls, enums, bounded arrays) and is defined once, generating both Pydantic and Java bindings.
2. The pipeline handles documents longer than the context window via chunk → extract → merge (§7), with provenance per value.
3. The validation layer includes verbatim-quote grounding, code-section pattern checks, date sanity rules, and a repair loop with ≤ 2 retries.
4. A gold set of ≥ 100 annotated documents. The report includes per-field exact match / F1, hallucination and omission rates, grounding rate, repair rate, cost, and latency.
5. A routing rule sends documents with any low-confidence or failed-validation field to a review queue. The report shows the automation rate at ≥ 98% field precision.

**Step-by-step**
1. Annotate the gold set with a simple labelling UI (or JSON files plus a guideline document).
2. Define the schema, then generate the Pydantic model and the Java record (datamodel-code-generator / jsonschema2pojo, or hand-written with contract tests).
3. Implement `structured_call` (§4.2) with the Anthropic or OpenAI parse helpers, plus `quotes_are_verbatim`.
4. Implement chunking, per-chunk extraction, and merging with conflict resolution.
5. Implement `field_scores` (§7) and aggregate the metrics per field.
6. Tune: descriptions, field order, and examples. Compare strict vs non-strict mode, and one-step vs reason→extract. Document the results.

---

### Project 2 — Cross-Provider Structured-Output Conformance Suite

**User stories**
- *As a platform engineer*, I want to know how each provider and engine handles our schemas (validity, accuracy, latency, unsupported keywords), so that we can switch providers without surprises.

**Acceptance criteria**
1. A suite of ≥ 15 schemas of increasing difficulty: flat, nested, enums, nullable, discriminated unions, arrays of objects, recursive (where supported), and schemas using `pattern` or numeric bounds.
2. Targets: Anthropic strict outputs, OpenAI strict outputs, and a self-hosted open model on vLLM or SGLang with two backends (e.g. XGrammar and llguidance).
3. For each (schema, target), records: request acceptance or rejection (with the error), schema-validity rate over 50 inputs, semantic accuracy on a labelled subset, first-call vs warm latency, and output tokens.
4. Auto-generates a compatibility matrix in Markdown, published in the repo README.
5. A portability linter flags schema features outside the common subset (§2.4).

**Step-by-step**
1. Write the schemas and matching input sets with gold outputs.
2. Implement a `Target` interface per provider (request builder, response parser, stop-reason mapping).
3. Implement the run loop with timing (TTFT, total) and store raw results in SQLite.
4. Implement the linter by walking the schema AST and checking keywords against per-target capability tables.
5. Generate the matrix and a narrative of findings (e.g. which keywords are silently ignored vs rejected).

---

### Project 3 — Spring Boot Structured-Output Service with a Schema Registry and Streaming UI

**User stories**
- *As a Java backend engineer*, I want a reusable service that turns any registered schema plus a document into a validated, typed result, so that product teams don't reimplement LLM parsing.
- *As a front-end developer*, I want partial results streamed to a React UI as fields are completed.

**Acceptance criteria**
1. A Spring Boot 3 service with endpoints `POST /schemas` (register a versioned JSON Schema), `POST /extract/{schema}/{version}` (sync), and `GET /extract/stream` (SSE with partial objects).
2. Uses a provider's native structured outputs through Spring AI or the provider's Java SDK. Validation uses a JSON Schema validator (e.g. networknt) plus pluggable semantic rules. The repair loop follows §4.
3. Streaming emits partial objects (a partial-JSON parse of accumulated deltas) and a final validated object. Side effects occur only on final validation.
4. Observability: Micrometer metrics for validity, repair rate, refusals, truncations, and latency per schema version. Every request carries a trace ID.
5. Contract tests: golden inputs per schema version, with a backward-compatibility check on registering a new version (additive-only unless `--breaking` is set).

**Step-by-step**
1. Scaffold the Spring Boot app with Spring AI (or the Anthropic Java SDK), Jackson, the networknt json-schema-validator, and Micrometer.
2. Implement the schema registry (Postgres table: `name, version, schema_json, created_at, status`) and a compatibility checker (diff the property sets and types).
3. Implement the extraction service: build the prompt → call → stop-reason check → validate → repair loop → typed result or problem+json error.
4. Implement SSE streaming with a Java port of `parse_partial_json` (or Jackson's non-blocking parser) and emit partial updates.
5. Build a small React page that renders fields progressively.
6. Add the Micrometer dashboards and the contract tests in CI.

---

## 11. Foundational Papers & Reading (exact titles)

- Tam et al., 2024 — *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models*
- Geng et al., 2025 — *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models*
- Shorten et al., 2024 — *StructuredRAG: JSON Response Formatting with Large Language Models*
- Geng et al., 2023 — *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*
- Willard & Louf, 2023 — *Efficient Guided Generation for Large Language Models*
- Schick et al., 2023 — *Toolformer: Language Models Can Teach Themselves to Use Tools*
- Patil et al., 2023 — *Gorilla: Large Language Model Connected with Massive APIs*
- Xu et al., 2024 — *Large Language Models for Generative Information Extraction: A Survey*
- Provider documentation: Anthropic *Structured outputs*; OpenAI *Structured model outputs*; vLLM / SGLang structured-output guides; JSON Schema 2020-12 specification

## 12. Essential Tooling

| Tool | Role |
|---|---|
| **Pydantic v2** (`model_validate_json`, `pydantic_core.from_json(allow_partial=True)`) | Python schema + validation + partial parsing |
| **Instructor**, **Pydantic AI** | Typed LLM outputs with retries across providers |
| **Anthropic / OpenAI SDK parse helpers** | Native strict outputs with typed results |
| **Spring AI** (`.entity(Class)`), **Anthropic Java SDK** (`outputConfig(Class)`) | JVM structured outputs |
| **victools jsonschema-generator**, **networknt json-schema-validator**, **Jakarta Bean Validation** | Java schema generation and validation |
| **datamodel-code-generator**, **jsonschema2pojo**, **quicktype** | Generate bindings from one schema source |
| **jsonschema** (Python), **ajv** (JS) | Standard validators for contract tests |
| **Zod** + `zodOutputFormat` / `zodTextFormat` | TypeScript schemas for provider SDKs |
