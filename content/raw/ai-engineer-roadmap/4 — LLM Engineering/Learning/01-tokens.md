# Tokens & Tokenization for LLM Engineers — Master Study Guide

> **Track:** LLM Engineering · **Module:** 01
> **Prerequisites:** Month 3 module 11 (LLMs) covered the concept; here we go operational.
> **Time budget:** ~5–7 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** Every LLM interaction is denominated in **tokens** — not characters, not words. Your API bill, your context window, your latency, your quality — every one is a function of tokens. Yet most engineers treat tokens as a mystery ("about 4 characters"). Fluency here saves money and hours of debugging.

Concretely, this module makes you the person who:
- Predicts token counts before making a call.
- Explains why "abcabcabc" and "abc abc abc" tokenize differently.
- Diagnoses tokenization-related bugs (mysterious API failures, prompt truncation, weird outputs on non-English text).
- Chooses tokenizers deliberately when working with open models.

**Fundamental principles you must own:**

1. **Tokens are subword units, not words or characters.** They come from BPE-family algorithms trained on a specific corpus.
2. **Different models tokenize the same string differently.** GPT-4, Claude, and Gemini each have their own vocabularies.
3. **Whitespace matters.** `" Hello"` and `"Hello"` often produce different tokens.
4. **Non-Latin scripts pay a token tax.** Chinese, Japanese, Arabic often 3-5× more tokens per character than English.
5. **Special tokens** (BOS, EOS, chat template tokens) count in your budget.
6. **You bill on input + output tokens.** Every message in a conversation is re-billed in the input.

If you retain nothing else: **know how to count tokens before you send them, and you won't be surprised.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 How Tokenizers Work (BPE Quickly)

**Byte-Pair Encoding (BPE)** in a nutshell:

1. Start with a base vocabulary of individual bytes (or characters).
2. Scan the training corpus; find the most frequent adjacent pair of tokens.
3. Merge that pair into a new token; add it to the vocabulary.
4. Repeat until vocabulary reaches target size (e.g., 100k tokens).

**Result:** frequent words become single tokens; rare words split into pieces.

**Variants:**
- **Byte-level BPE (GPT-2/3/4, GPT-3.5, GPT-4)** — operates on raw UTF-8 bytes; handles all Unicode.
- **WordPiece (BERT)** — similar but uses `##` prefix for subwords.
- **SentencePiece (T5, LLaMA)** — treats whitespace as a character (`▁`).
- **Unigram** (also SentencePiece) — probabilistic; different training procedure.

**Vocabulary sizes** (approximate, 2026):

- GPT-2: 50,257
- GPT-3.5 / GPT-4 (`cl100k_base`): 100,277
- GPT-4o (`o200k_base`): 200,019
- Claude (Anthropic tokenizer): ~65,000
- LLaMA-3: 128,000
- Gemini: ~256,000 (via SentencePiece)

**Larger vocabularies** = fewer tokens per sentence = shorter sequences at the cost of larger embedding matrices.

**Special tokens** — reserved IDs for BOS (`<|begin_of_text|>`), EOS (`<|endoftext|>`), padding, chat roles, etc. Almost every modern chat model uses a chat template with special tokens:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
```

Those `<|im_start|>` tokens count against your budget.

---

### 2.2 Counting Tokens in Practice

**OpenAI models — `tiktoken`:**

```python
import tiktoken

# For GPT-4o, GPT-4-turbo, GPT-4o-mini
enc = tiktoken.get_encoding("o200k_base")

# For GPT-4, GPT-3.5-turbo, text-embedding-3-*
enc = tiktoken.get_encoding("cl100k_base")

n = len(enc.encode("Hello, world!"))    # 4
tokens = enc.encode("The quick brown fox")
# [791, 4062, 14198, 39935]
print([enc.decode([t]) for t in tokens])
# ['The', ' quick', ' brown', ' fox']  ← note the leading spaces
```

**Anthropic — `anthropic.Anthropic().count_tokens()` (older) or count via the messages API's `input_tokens` usage field.** As of 2024+ Anthropic recommends the beta counting endpoint:

```python
client.beta.messages.count_tokens(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hi"}],
)
```

**HuggingFace models — `AutoTokenizer`:**

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
ids = tok.encode("Hello, world!")
n = len(ids)
```

**Rough English-to-token conversions:**

- 1 token ≈ 4 characters English
- 1 token ≈ 0.75 words English
- 100 tokens ≈ 75 words ≈ half a paragraph
- 1000 tokens ≈ 750 words ≈ a short article

But these are **very rough**. Always count exactly for real budgets.

**Counting chat conversations.** The chat template adds overhead. OpenAI recommends this formula for `gpt-3.5-turbo`:

```
tokens = 3  # every reply is primed with <|start|>assistant<|message|>
for message in messages:
    tokens += 3  # every message has <|start|>{role/name}<|end|>
    tokens += num_tokens(message["content"])
    if "name" in message:
        tokens += 1
```

For up-to-date rules, use the API's returned `usage.prompt_tokens` and `usage.completion_tokens` as ground truth.

---

### 2.3 Tokenization Traps

**English is cheap; other languages aren't.**

```python
enc = tiktoken.get_encoding("cl100k_base")
len(enc.encode("Hello, how are you?"))          # 6 tokens
len(enc.encode("こんにちは、元気ですか?"))         # ~14 tokens (2× characters)
len(enc.encode("你好,你怎么样?"))                  # ~10 tokens
```

If your app serves non-English users, this affects both cost and context. GPT-4o's `o200k_base` was designed specifically to reduce this tax and produces roughly 2× fewer tokens on Asian languages than `cl100k_base`.

**Whitespace is part of the token.**

```
"Hello" → [' Hello']   (with leading space)
"Hello" (start of string) → ['Hello']  (no leading space)
```

Common bug: fine-tuning data where responses have inconsistent leading whitespace produces subtly different tokens and confuses the model.

**Numbers tokenize poorly.**

```python
enc.encode("1234567")
# May produce ['12', '345', '67'] — three tokens for a single number
```

This is why LLMs are bad at arithmetic — the model doesn't see "1234567" as one thing; it sees ambiguous chunks. Newer models (Claude 3, GPT-4o) tokenize digits individually to improve math.

**URLs / code blob artifacts.** Long unique strings (URLs, base64) tokenize into many small tokens. `<script>` tags, weird Unicode, emojis — each has quirks. Test with your actual data.

**Repeated substrings.** Sometimes 100 identical characters become "one big token" (`"aaaaaaaaaa..."` → 1–2 tokens). Sometimes not. Depends on merges.

**Counting characters via LLMs.** LLMs can't reliably count letters in a word ("how many r's in strawberry"). This isn't a reasoning failure — the model literally can't see the letters. If you need character-level operations, do them outside the LLM.

---

### 2.4 Prompt Caching (2024+)

Modern APIs (OpenAI, Anthropic, Gemini) support **prompt caching** — reuse a large static prefix (system prompt, long instructions, retrieved documents) across requests at a fraction of the token cost.

**Anthropic prompt caching** (as of 2024):
```python
client.messages.create(
    model="claude-3-5-sonnet-latest",
    system=[
        {"type": "text", "text": "long system prompt here...",
         "cache_control": {"type": "ephemeral"}},
    ],
    messages=[...],
)
```

Cached tokens cost ~10% of normal tokens on subsequent reads within the cache TTL (~5 min for ephemeral). Cache writes cost ~1.25× normal.

**OpenAI prompt caching** (automatic, no code change) — caches prefixes ≥ 1024 tokens; reads cost ~50% of normal.

**Design implication.** Put stable content (system prompts, retrieved docs) at the **start** of your prompt; variable content (user's latest message) at the **end**. Cache hits happen only on prefix matches — a single differing token at position 1 invalidates the entire cache.

**Semantic caching** (a different beast) — cache prior *responses* keyed by embedding similarity of the input. Different problem, different tools (e.g., GPTCache, langchain's cache). See Module 10.

---

### 2.5 Practical Token Budgeting

**Before every API call, know:**
- Your input tokens (system + history + user message + retrieved context).
- Your output token budget (`max_tokens`).
- Your model's context window (see Module 09).

Example budget for a chat app with RAG:

| Segment | Tokens |
|---------|--------|
| System prompt | ~500 |
| Retrieved context (5 chunks × 400 tokens) | ~2,000 |
| Conversation history (last 5 turns × ~150) | ~750 |
| Current user message | ~200 |
| **Total input** | **~3,450** |
| Output max_tokens | 1,000 |
| **Total round-trip** | **~4,450** |

Against Claude 3.5 Sonnet's 200k context: fine. Against a hypothetical 4k-context model: about at the limit.

**Truncation strategies** when budget is tight:
1. **Summarize old turns** — replace older messages with a summary.
2. **Sliding window** — drop oldest turns.
3. **Retrieval instead of history** — offload knowledge to a vector store; only recent conversation stays inline.
4. **Compress documents** — pre-summarize before injection.

**Cost estimation function:**

```python
def estimate_cost(input_tokens, output_tokens,
                  price_in_per_1k=0.003, price_out_per_1k=0.015):
    return (input_tokens/1000)*price_in_per_1k + (output_tokens/1000)*price_out_per_1k
```

For 10 million monthly requests averaging 3k input + 500 output at Claude 3.5 Sonnet prices ($3/M input, $15/M output):

$$\text{cost} = 10^7 \cdot \left(\frac{3000}{10^6} \cdot 3 + \frac{500}{10^6} \cdot 15\right) = 10^7 \cdot (0.009 + 0.0075) = \$165{,}000/\text{month}$$

That's the kind of math that should live in your head, not a spreadsheet after the fact.

---

## 3. Mental Models & Analogies

### 3.1 The "Postal Weight Class" Model

Sending a message to an LLM API is like mailing a package by weight class, not size. The service quotes prices per gram (per token), not per page. A wordy sentence and a compact bullet list can weigh the same. English is aluminum foil (lightweight per character); Japanese and Chinese are lead sheets — same physical shape, much heavier.

**Consequences:**
- The address label ("system prompt") gets weighed every time.
- Every conversation message you attach gets weighed again on each new request — the postal service re-weighs the whole envelope.
- **Prompt caching** = the courier remembers your last envelope's prefix and charges you only for the changes at the end.

This model makes cost intuition robust. If you catch yourself thinking "I'll just include the whole document in every request" — that's like mailing your entire filing cabinet with each letter. Retrieval + caching solves it.

### 3.2 The "Foreign Alphabet Keyboard" Model

Imagine you're typing on a keyboard that has **65,000 keys**. Each key represents a token. To type "hello world" you press two keys (`hello`, ` world`). To type "hello-world" you might press four keys (`hello`, `-`, `world`, or however BPE learned to split hyphens). To type "こんにちは" you press many keys because there are fewer common Japanese multi-character keys.

Now the keyboard's designers made choices — the GPT-4 keyboard's designers included many keys for English tech vocabulary; the Claude keyboard included different keys. Same string, different key sequence, sometimes different length.

This model demystifies:
- **Why non-English is expensive** (fewer common keys for those scripts).
- **Why URLs cost so much** (they use rare character combinations).
- **Why you can't count letters** (the model reads keys, not letters).
- **Why fine-tuning tokenization matters** (you can add specialized keys for your domain).

![IMG-TOK-01](/4%20—%20LLM%20Engineering/images/IMG-TOK-01.jpg)

> **Caption:** Same string, different tokenizers, different token counts and cost.
> **Placement:** Section 2.1.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "One Token Is About Four Characters"

True on average for English prose. Wildly wrong for code, JSON, non-English text, URLs, base64 blobs, or numbers. Never budget from character count; count tokens directly. Also: **the same content can produce a different token count** depending on whether it starts with a space, whether it's mid-sentence, or which model's tokenizer is used.

### 4.2 "Once I Pay for a Token, I Own It"

You don't. In a chat, every prior message is **re-billed** on every new request (input tokens). If your conversation is 20 turns deep with 500 tokens each, that's 10,000 input tokens every time the user sends a new message. Long chats get exponentially expensive without truncation or summarization. Prompt caching mitigates this only for the *stable prefix* portion.

### 4.3 "The LLM Sees the Same Text I Do"

It doesn't. It sees token IDs. "Strawberry" might be a single token; the model can't count characters inside it. "1234" might be `[123, 4]` — the model doesn't see individual digits reliably. When you ask an LLM to reverse a string or count occurrences of a letter, you're asking it to reason about characters it literally can't see. Use structured decoding or offload to a code-executing tool.

---

## 5. Self-Assessment Bank (Tokens)

### Questions

**Q1 (Short answer).** In one paragraph, explain how BPE builds its vocabulary.

**Q2 (Multiple choice).** Approximately how many tokens does "The quick brown fox jumps over the lazy dog" tokenize to in `cl100k_base`?
- (a) 3
- (b) 9
- (c) 20
- (d) 40

**Q3 (Short answer).** Why does the same English word tokenize differently at the start of a string vs mid-string?

**Q4 (Multiple choice).** Which of these is a special token in a chat template?
- (a) `strawberry`
- (b) `<|im_start|>`
- (c) `Hello`
- (d) `,`

**Q5 (Short answer).** Explain the "token tax" on non-English languages and one modeling implication.

**Q6 (Multiple choice).** OpenAI prompt caching typically triggers on:
- (a) The last 100 characters of your prompt.
- (b) A stable prefix ≥ 1024 tokens; cached read costs ~50% of normal.
- (c) The output tokens.
- (d) Only on function calls.

**Q7 (Short answer).** You have a chat app that keeps growing conversation history. Cost per user is climbing linearly with conversation length. Name three approaches to bound it.

**Q8 (Multiple choice).** Why can't a modern LLM reliably answer "how many R's are in strawberry"?
- (a) It's a general reasoning failure.
- (b) The tokenizer merges the word into one or two tokens; the model doesn't see individual characters.
- (c) There's a bug in the API.
- (d) The training data was wrong.

**Q9 (Short answer).** In one sentence: what does `tok.encode(" Hello")` return that's different from `tok.encode("Hello")`?

**Q10 (Multiple choice).** In a chat conversation, when the user sends message 20, the input tokens billed are:
- (a) Only message 20's tokens.
- (b) All 20 messages' tokens plus the system prompt (minus any cached prefix).
- (c) The previous response tokens only.
- (d) A random sample.

---

### Answer Key & Detailed Explanations

**A1.** BPE starts with a base vocabulary of single characters (or bytes). It counts the most frequent adjacent pair of tokens in the training corpus and merges them into a new token added to the vocabulary. It repeats this greedy merging until reaching a target vocabulary size (~50k–200k). Common substrings (like "ing", "tion", "the") become single tokens; rare words split into pieces.

**A2. (b).** ~9 tokens. Common English tokenizes cleanly.

**A3.** Because tokenizers include whitespace as part of the token. `"Hello"` at position 0 has no leading space; the same word after a preceding token typically comes with a leading space. `enc.encode("Hello")` might be `[9906]`; `enc.encode(" Hello")` might be `[13225]` — different token IDs entirely.

**A4. (b).** `<|im_start|>` is a special token used in chat templates to mark the start of a role's turn. It's reserved in the vocabulary and counts against your token budget.

**A5.** English tokenizes ~4 characters per token; Chinese, Japanese, Korean, Arabic, and many other scripts often tokenize at 1–2 characters per token. Cost per character can be 3–5× higher, and context windows fill up much faster. Modeling implication: non-English users experience worse quality on tasks near the context limit, and can be more expensive to serve. Newer tokenizers (`o200k_base`, Gemini's) reduce this gap.

**A6. (b).** OpenAI's prompt cache kicks in on prefixes ≥ 1024 tokens; cached prefix reads are ~50% of the normal input price. Cache is invalidated if the prefix changes.

**A7.** (1) **Truncation** — drop or summarize old turns. (2) **Rolling summary** — replace older messages with a running summary token-block. (3) **Retrieval** — offload knowledge to a vector store; only recent turns stay inline. (4) **Prompt caching** — keep a stable prefix and cache it (reduces cost of the fixed portion, not the growing history). (5) **Session compression** — periodically compress the whole session with a summarization pass.

**A8. (b).** Tokenizers combine "strawberry" into one (or a small number of) tokens. The model's internal representation is the token ID, not the character sequence. It cannot literally see the R's. Fixes: use a tool that operates on characters, or reason step-by-step through characters using explicit prompting (which sometimes helps by forcing character-level extraction).

**A9.** `tok.encode("Hello")` typically returns a single token ID for the raw word; `tok.encode(" Hello")` returns a different single token ID for the "space-Hello" combined token — the leading space is part of the token in BPE-family tokenizers.

**A10. (b).** LLM APIs are stateless. Every request re-sends the entire conversation history, and every request bills the input tokens for that history. Prompt caching can reduce the cost of the stable prefix portion but not the growing tail of conversation.

---

## 6. Practice Prompts

1. **Tokenizer explorer.** Write a small script that takes a string and prints its tokens (with decoded text of each) for GPT-4o, GPT-3.5, Claude, and LLaMA-3. Compare.
2. **Language tax.** Take the same English sentence, translate it to Chinese, Japanese, Spanish, and Swahili (with Google Translate). Count tokens in each. Chart the difference.
3. **Chat cost predictor.** Build a function `estimate_conversation_cost(messages, model)` that returns projected input tokens, output tokens, and dollar cost. Test on a real conversation.
4. **Cache design.** Given a system prompt of 1500 tokens and 100 daily users each with 20-turn conversations, compute the cost with vs. without prompt caching.
5. **Token counting an unknown model.** Given a HuggingFace model ID, write a script that downloads its tokenizer and counts tokens for a sample string.

---

## 7. References

- OpenAI: [How to count tokens with tiktoken](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).
- Anthropic: [Prompt caching guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
- Andrej Karpathy, ["Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE) — canonical explainer.
- HuggingFace: [Summary of tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary).
- Simon Willison, ["Understanding GPT tokenizers"](https://simonwillison.net/2023/Jun/8/gpt-tokenizers/).
