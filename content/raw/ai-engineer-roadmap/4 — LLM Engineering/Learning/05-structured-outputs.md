# Structured Outputs — Master Study Guide

> **Track:** LLM Engineering · **Module:** 05
> **Prerequisites:** Modules 01–04.
> **Time budget:** ~6–8 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every real LLM application eventually needs the model to return **structured data** — JSON, XML, a fixed grammar — that downstream code can parse without heroic string manipulation. Get this right and your app is reliable; get it wrong and you'll spend engineering weeks writing regex parsers and retry loops.

Modern providers (OpenAI, Anthropic, Google, and every open-model server) support **structured decoding** — a constrained-generation mode that guarantees the output matches a specified schema. This is not "please return JSON"; it's a grammar-constrained decoder that literally can't produce invalid output.

**Fundamental principles you must own:**

1. **Structured decoding is grammar-constrained.** The model's token probabilities are masked so only valid continuations remain possible.
2. **Prefer structured output over prompt-based JSON.** Prompt "return JSON" produces JSON *usually*; structured output produces JSON *always*.
3. **JSON Schema is the interoperable format.** Learn it — every provider uses it or a close cousin.
4. **Function calling (Module 06) is a specialized case** of structured output where the schema describes a callable function.
5. **Even with structured output, validate.** Test that the schema matches your Pydantic model exactly.
6. **Use Pydantic to define schemas in Python.** Less bug-prone than hand-writing JSON Schema.

If you retain nothing else: **JSON-out-of-LLM is a solved problem — use structured decoding, not string parsing.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 Why "Return JSON" Prompts Fail (Enough)

**Prompt-only JSON:**

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract {name, age} from: 'Brian is 42'. Return JSON."}],
)
print(resp.choices[0].message.content)
```

Typical failures:
- Wrapping the JSON in markdown code fences (` ```json ` … ` ``` `).
- Adding prose before/after ("Sure! Here's the extracted JSON:").
- Truncating mid-object due to `max_tokens`.
- Inconsistent field names, missing fields, added fields.
- Extra commas, unquoted keys, single quotes.
- Hallucinating fields that weren't asked for.

At scale, even a 1% failure rate produces daily incident reports. **Don't parse LLM JSON with regex** — use grammar-constrained decoding.

**JSON mode (OpenAI, older API):**

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[..., {"role": "user", "content": "Return JSON."}],
    response_format={"type": "json_object"},
)
```

JSON mode guarantees the output is **syntactically valid JSON** but doesn't enforce any schema. Fields can still be wrong, missing, or extra.

**Structured Output with schema** (modern default):

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
)
```

`strict=True` guarantees the output matches the schema exactly. This uses constrained decoding under the hood.

---

### 2.2 Defining Schemas — Pydantic + Structured Output

**Pydantic is the ergonomic Python interface.** Both OpenAI and Anthropic (via the `instructor` library or native support) accept Pydantic models.

```python
from pydantic import BaseModel, Field
from typing import Literal

class Address(BaseModel):
    street: str
    city: str
    state: Literal["MT", "CA", "NY", "TX", "..."]
    zip: str = Field(pattern=r"^\d{5}$")

class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    email: str | None = None
    role: Literal["employee", "manager", "contractor"]
    address: Address
```

**OpenAI native structured output with Pydantic:**

```python
from openai import OpenAI
client = OpenAI()

resp = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract structured person data."},
        {"role": "user",   "content": "Brian, 42, VP of Engineering, based in Helena, MT 59601, at 123 Main St."},
    ],
    response_format=Person,
)
person = resp.choices[0].message.parsed   # → Person instance, already validated!
```

If parsing fails, `parsed` is `None`, and `refusal` may indicate the model refused.

**Anthropic + instructor:**

```python
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())
person = client.messages.create(
    model="claude-3-5-sonnet-latest",
    response_model=Person,
    messages=[{"role": "user", "content": "..."}],
    max_tokens=500,
)
```

**Or Anthropic's native tool-calling for structured output:**

Anthropic doesn't have a dedicated "structured output" mode; the idiomatic approach is a tool that describes the output schema and force the model to call it:

```python
tools = [{
    "name": "record_person",
    "description": "Record extracted person information.",
    "input_schema": Person.model_json_schema(),
}]

resp = client.messages.create(
    model="claude-3-5-sonnet-latest",
    tools=tools,
    tool_choice={"type": "tool", "name": "record_person"},
    messages=[{"role": "user", "content": "..."}],
    max_tokens=500,
)
person = Person(**resp.content[0].input)
```

**Google Gemini:** supports JSON Schema via `response_schema` and `response_mime_type="application/json"`.

**Open models via `outlines`, `guidance`, `llama.cpp`'s grammar support** — Pydantic → JSON schema → grammar-constrained decoding.

---

### 2.3 What Structured Decoding Guarantees (and What It Doesn't)

**Guarantees:**
- Output is syntactically valid JSON (or the target format).
- Output matches the specified schema (field names, types, required fields).
- No extra fields (if `additionalProperties: False`).
- Enum values are in the enum.
- Numbers respect min/max where the model supports it.

**Doesn't guarantee:**
- **Correctness of content.** The model can still fill "age": 999 if you say `ge=0, le=1500` — schema constraints on numeric ranges aren't always enforced at decode time.
- **Semantic accuracy.** JSON says `{"name": "Brian", "age": 42}` but Brian might not be 42.
- **Sensible refusals.** The model may still produce a "best guess" for a field it doesn't know.
- **Nested consistency.** If you have a discriminated union with a "type" field, the corresponding shape may not always be right.

**Design implications:**

- Add **soft validation** on top — Pydantic's field validators, business rule checks.
- For unknown values, prefer nullable fields (`Optional[str]`) and instruct the model to leave them null.
- Test with adversarial inputs — outputs might be constrained to schema but semantically wrong.

**Performance:**
- Structured decoding adds ~5–15% latency overhead vs raw generation.
- For long structured outputs (>1000 tokens), the overhead compounds.
- Weighing: usually worth it for reliability.

---

### 2.4 Schema Design Patterns

**Discriminated unions** — model a set of alternatives with a "type" field:

```python
from typing import Union, Literal
from pydantic import BaseModel, Field

class TextBlock(BaseModel):
    type: Literal["text"]
    content: str

class ImageBlock(BaseModel):
    type: Literal["image"]
    url: str
    caption: str

class Document(BaseModel):
    blocks: list[Union[TextBlock, ImageBlock]] = Field(discriminator="type")
```

The `discriminator="type"` tells Pydantic (and downstream JSON Schema tooling) to route on the "type" field. Common pattern for extraction pipelines.

**Optional vs required:**
- Required: must be present in output.
- Optional (`str | None`): the model can output `null` when unknown.

**Explicit "unknown" instead of null.** Sometimes better to add a sentinel:

```python
class Field(BaseModel):
    value: str
    confidence: Literal["high", "medium", "low", "unknown"]
```

More expressive than nullable — the model can flag low confidence rather than pretending to know.

**Reasoning field first.** For extraction with reasoning, include a scratchpad in the schema **before** the answer:

```python
class Answer(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about the answer.")
    final_answer: str = Field(description="The concise final answer.")
```

The model generates left-to-right, so `reasoning` is populated first, and `final_answer` benefits from that context. This is CoT baked into the schema.

**Avoid over-nesting.** Deep nesting confuses models. Flatten when possible.

**Use descriptions.** Pydantic `Field(description="...")` becomes the JSON Schema "description" — models read and use these. Treat descriptions as prompts.

```python
class Product(BaseModel):
    name: str = Field(description="Product name as it appears on the label; strip trademark symbols.")
    price_usd: float = Field(description="Price in USD; if listed in another currency, convert using today's approximate rate.")
```

---

### 2.5 When Structured Output Fails and What to Do

**Failure mode 1: model refuses.** The model produces a `refusal` field (OpenAI) or a text explanation. Handle:

```python
resp = client.beta.chat.completions.parse(..., response_format=Person)
message = resp.choices[0].message
if message.refusal:
    logger.warning("Model refused: %s", message.refusal)
    # fallback: send back to human, try a different prompt, etc.
else:
    person = message.parsed
```

**Failure mode 2: parsed=None.** Rare with `strict=True` but happens. Retry with a corrective prompt, escalate to a stronger model, or return an error.

**Failure mode 3: content is wrong even though schema valid.** The hardest one. Solutions:
- **Add a self-check step**: after generation, ask the model "does this output correctly answer the question? If not, revise."
- **Multi-sample and vote** — generate $N$ outputs, pick the most common.
- **Combine with retrieval** — force the model to cite sources.
- **Post-hoc validation** — business rules that reject impossible outputs.

**Failure mode 4: schema too complex; model gives up on some fields.** Simplify. Consider splitting into multiple calls:

```
Call 1: Extract "entities" (list of names/dates/orgs).
Call 2: For each entity, extract detailed info.
```

**Streaming + structured output.** OpenAI supports streaming parsed responses — partial JSON is emitted as it generates. Useful for showing UI-side progress on long structured tasks. `client.beta.chat.completions.stream(...)`. Handle partials with care.

**Structured output + tools.** Function/tool calling (Module 06) is essentially "structured output" for calling functions. Same technology, different framing.

---

## 3. Mental Models & Analogies

### 3.1 The "Form-Filling Kiosk" Model

Structured output turns the LLM from a **free-form writer** into a **form-filling kiosk**. Instead of giving open-ended prose, you hand it a form with specific fields ("Name: ___", "Age: ___", "Email: ___"), and it fills in each field one by one. Grammar-constrained decoding is the kiosk's software: at every keystroke, it only offers valid options — you can only type digits in the "age" box.

Consequences:
- Downstream code trusts the form structure — always parseable.
- The model can't invent extra fields.
- The model must fit its response into the form's shape — no rambling.
- Complex forms with 30 fields tax the model; simpler forms extract more reliably.

This model also explains **why the descriptions in `Field(description=...)` matter**: they're the labels the model reads to figure out what belongs in each blank. Fill the labels with clear instructions and the form gets filled correctly.

### 3.2 The "Grammar Is a Fence" Model

Free-form generation is the model wandering a huge field of possible token sequences. Structured decoding is a **fence** — at each step, the model sees only tokens compatible with the schema so far. If we're inside `"age":`, only digits or `null` are behind the fence.

The model still has plenty of room to think (choosing values), but it can't wander into `"age": "not sure I remember exactly"`. The fence isn't a suggestion; it's mathematically enforced by masking logits.

This mental model also demystifies **latency overhead**: constructing and enforcing the grammar at each step costs compute. But since the model would otherwise produce longer output (parsing prose, apologies, prose around JSON), constrained decoding often ends up cheaper *in total tokens*.

![IMG-STRUCT-01](/4%20—%20LLM%20Engineering/images/IMG-STRUCT-01.jpg)

> **Caption:** Prompt-only JSON produces best-effort text; structured decoding produces a schema-valid object by construction.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "Structured Output = Correct Output"

Structured output guarantees schema conformance, not semantic correctness. The model may still fill fields with wrong or hallucinated values. Always: (a) validate with business rules, (b) test on labeled examples, (c) log outputs for later review.

### 4.2 "I'll Just Regex the JSON Out of the Response"

You will spend cumulative days maintaining that regex. Every provider updates their model, and the surrounding prose changes. Every unusual input pattern breaks the parser. Use structured output APIs. When you can't (older provider or unsupported model), use `instructor` or `outlines`. Regex-parsing LLM output is a smell.

### 4.3 "Bigger Schemas Are Fine — Just Model Everything"

Complex schemas (deep nesting, many required fields, big enums) tax the model's attention. Practical limits:
- Keep hierarchy shallow (≤ 3 levels).
- Keep field count reasonable (≤ 20 top-level fields).
- Split large tasks into multiple calls if one call struggles.
- Descriptions are your friend, but excessive verbosity in descriptions also hurts.

Test complex schemas on realistic inputs; iterate.

---

## 5. Self-Assessment Bank (Structured Outputs)

### Questions

**Q1 (Short answer).** What does "structured output" or "grammar-constrained decoding" guarantee that a "please return JSON" prompt does not?

**Q2 (Multiple choice).** OpenAI's `response_format={"type": "json_object"}` guarantees:
- (a) Valid JSON syntax.
- (b) Correct JSON schema.
- (c) Semantically correct output.
- (d) All of the above.

**Q3 (Short answer).** Why is Pydantic a good way to specify schemas in Python for structured LLM outputs?

**Q4 (Multiple choice).** For putting reasoning + final answer into the same structured output, the best pattern is:
- (a) Put `final_answer` first, then `reasoning`.
- (b) Put `reasoning` first, then `final_answer`.
- (c) Put them in random order.
- (d) Don't include reasoning.

**Q5 (Short answer).** Describe the discriminated-union pattern and one situation where it helps.

**Q6 (Multiple choice).** With `strict=True` structured output on OpenAI, what happens if the model can't answer?
- (a) It always answers, even guessing.
- (b) It may set a `refusal` field on the message; `parsed` will be None.
- (c) The API errors out.
- (d) Nothing; you just get the schema with nulls.

**Q7 (Short answer).** Give one downside of structured decoding.

**Q8 (Multiple choice).** For Anthropic Claude, structured output is idiomatically done via:
- (a) A dedicated `response_format` argument.
- (b) A tool definition that describes the output schema, forced with `tool_choice`.
- (c) Prompt engineering only.
- (d) Not supported.

**Q9 (Short answer).** Name three field-level design patterns that help LLMs produce good structured output.

**Q10 (Multiple choice).** For very complex extraction tasks with a large nested schema, the recommended approach is:
- (a) Larger model.
- (b) Split into multiple calls with simpler schemas each; compose the results.
- (c) Longer temperature.
- (d) Larger `max_tokens`.

---

### Answer Key & Detailed Explanations

**A1.** Grammar-constrained decoding guarantees the model can *only* produce tokens that keep the output valid under the schema — no invalid JSON, no missing required fields, no extra top-level keys, no invalid enum values. A prompt-only "please return JSON" produces JSON usually but has a nonzero failure rate that grows with volume.

**A2. (a).** `json_object` mode guarantees only that the output is valid JSON. It doesn't enforce a schema — fields can be missing, extra, or wrong types. For schema enforcement, use `json_schema` with `strict=True`.

**A3.** Pydantic models express schemas as type-hinted Python classes with validators. They generate a JSON Schema via `model_json_schema()` for the LLM API and validate parsed responses against the same schema. Same source of truth for schema + validation + downstream types — no duplication, no drift.

**A4. (b).** Reasoning first, then final answer. Left-to-right generation means the model produces `reasoning` before `final_answer`, letting the reasoning act as a scratchpad that informs the final answer. Reversing puts the final answer before its own justification — the model has to commit before thinking.

**A5.** Discriminated union: a type with a "type" field distinguishing several alternative shapes. Example: `Block` = `TextBlock | ImageBlock | TableBlock`. Helps when the schema has genuine alternatives (e.g., different message types, different event kinds), each with its own required fields.

**A6. (b).** OpenAI's `strict=True` structured output includes a `refusal` mechanism. If the model refuses (safety, uncertainty), it populates `message.refusal` with a string explanation; `message.parsed` will be `None`. Handle this in code — don't assume every call yields a valid object.

**A7.** Downsides: (a) small latency overhead (constrained decoding is slower per token); (b) doesn't guarantee semantic correctness (only syntactic); (c) very complex schemas can degrade quality (model devotes more attention to structure than content); (d) sometimes limits the model's ability to express uncertainty naturally.

**A8. (b).** Anthropic's idiomatic pattern is to define a tool whose `input_schema` is your desired output schema, and force the model to call it with `tool_choice={"type": "tool", "name": "..."}`. The tool's inputs become your structured output. `instructor` and other libraries handle this abstraction transparently.

**A9.** Any three of: **use descriptions on every field** (they're prompts), **put reasoning fields before answer fields**, **use `Literal` / enums for constrained values**, **prefer `None` over "unknown" for missing data**, **discriminated unions for alternatives**, **shallow hierarchies over deep nesting**, **explicit confidence fields when the model should express uncertainty**, **regexp constraints on strings (zip codes, IDs)**.

**A10. (b).** Complex schemas overwhelm models. Break into pipeline: (1) extract entities; (2) for each entity, extract details; (3) compose the final structure. Each call is simpler, more reliable, and often cheaper in total than one mega-schema call.

---

## 6. Practice Prompts

1. **Migration exercise.** Take a prompt-only JSON extractor from an existing project. Migrate it to `response_format=json_schema` with `strict=True`. Measure failure-rate before/after on 100 sample inputs.
2. **Pydantic schema.** Design a Pydantic model for extracting invoice data (invoice number, date, line items, totals). Include validators. Test on a small labeled invoice set.
3. **Discriminated union.** Build a schema for a message log entry with types: `text`, `image`, `system_event`, `tool_call`. Test that the model produces the right shape based on input.
4. **Reasoning-first pattern.** For a hard extraction task, compare two schemas: `{answer}` vs `{reasoning, answer}`. Measure accuracy improvement.
5. **Multi-call pipeline.** For a 10-field extraction, split into 3 calls (entities → details → composition). Compare to a single-call baseline for accuracy and cost.

---

## 7. References

- OpenAI Structured Outputs: [https://platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs).
- Anthropic Tool Use: [https://docs.anthropic.com/en/docs/build-with-claude/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use).
- `instructor` library: [https://python.useinstructor.com/](https://python.useinstructor.com/).
- `outlines`: [https://github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines).
- JSON Schema: [https://json-schema.org/](https://json-schema.org/).
- Pydantic: [https://docs.pydantic.dev](https://docs.pydantic.dev).
