# HuggingFace — Master Study Guide

> **Track:** LLM Engineering · **Module:** 11
> **Prerequisites:** Modules 01–10, PyTorch (Month 3 Module 12).
> **Time budget:** ~10–12 hours.

---

## 1. Executive Summary & Core Concepts

**Why this matters.** HuggingFace is the **GitHub of models**. Nearly every open-weight model (LLaMA, Mistral, Qwen, Gemma, DeBERTa, sentence-transformers, Whisper, CLIP, Stable Diffusion, ...) is distributed via the HuggingFace Hub. The `transformers`, `datasets`, `accelerate`, `peft`, and `evaluate` libraries are the de-facto tools for fine-tuning, evaluating, and serving open models. Fluency here is what separates "I can call an API" from "I can bring open-source AI in-house."

**Fundamental principles you must own:**

1. **The Hub is a git-based model registry** — every model, dataset, and space is a git repo.
2. **`transformers`** unifies model loading, tokenization, generation, and training across architectures.
3. **`datasets`** wraps efficient columnar (Arrow) datasets with streaming, mapping, filtering, splitting.
4. **`accelerate`** abstracts multi-GPU / distributed / mixed-precision setup.
5. **`peft`** implements LoRA, QLoRA, and other parameter-efficient methods.
6. **`transformers` inference is production-grade** for many tasks; for high-throughput serving, use `text-generation-inference` (TGI), `vLLM`, or `sglang`.

If you retain nothing else: **HuggingFace is what turns "I'll use an open-weight model" from a research task into a Tuesday afternoon.**

---

## 2. Deep-Dive Breakdown — Top 5 Sub-topics

### 2.1 The Hub, Models, and Tokenizers

**Hub basics.** Every entity (model, dataset, space) lives at `https://huggingface.co/{owner}/{name}`. You can `git clone` any model. Public models are free; some require accepting a license (LLaMA, gated Mistral versions).

**Loading a model + tokenizer:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="bfloat16",   # or "auto"
    device_map="auto",         # spreads across available GPUs
)
```

**Auto-classes.** `AutoModel`, `AutoTokenizer`, `AutoConfig`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification`, etc. Look at the config and pick the right architecture automatically.

**Task classes:**
- `AutoModelForCausalLM` — GPT-style (decoder-only, autoregressive).
- `AutoModelForSequenceClassification` — classification head on top.
- `AutoModelForTokenClassification` — NER, POS.
- `AutoModelForQuestionAnswering` — extractive QA.
- `AutoModelForSeq2SeqLM` — T5, BART.
- `AutoModelForMaskedLM` — BERT-style MLM.

**Tokenizer methods:**

```python
# Encode
inputs = tokenizer("Hello, world!", return_tensors="pt")
# inputs = {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}

# Decode
tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Batch with padding
tokenizer(["a", "long sentence here"], padding=True, truncation=True, max_length=512, return_tensors="pt")

# Chat template
messages = [{"role": "user", "content": "Hi"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**Chat templates.** Every instruction-tuned model has its own chat format (special tokens, role markers). `apply_chat_template` renders it correctly per model — never hand-format chat prompts.

**Model cards.** Each model has a `README.md` describing training data, intended use, limitations, and a config. **Always read the model card** before deploying — many models have restricted licenses or specific use-case guidance.

---

### 2.2 Generation: `.generate()` and Its Parameters

The `.generate()` method is HF's inference workhorse:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Key parameters:**

- `max_new_tokens` — cap on generated tokens. **Always set this.**
- `do_sample` — `False` = greedy; `True` = probabilistic sampling.
- `temperature` — softmax temperature. Lower = more focused; higher = more diverse.
- `top_k` — sample from top-k tokens by probability.
- `top_p` — nucleus sampling; smallest set whose cumulative probability ≥ p.
- `repetition_penalty` — >1 discourages repeated tokens.
- `num_beams` — beam search width; deterministic when >1 (use for translation).
- `eos_token_id` — stop generation.
- `stopping_criteria` — custom stop conditions.
- `streamer` — a `TextStreamer` object to emit tokens as they're generated (see below).

**Streaming:**

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=256)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
for chunk in streamer:
    print(chunk, end="", flush=True)
thread.join()
```

**Batching for inference:**

```python
batched = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    outs = model.generate(**batched, max_new_tokens=100)
decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
```

**Left-padding for causal LMs.** For decoder-only models generating in batch, pad on the **left** so all sequences end at the same position: `tokenizer.padding_side = "left"`.

---

### 2.3 The `pipeline` Abstraction

For quick prototyping, `pipeline` wraps everything:

```python
from transformers import pipeline

# Zero-shot classification
clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
clf("This dish is amazing!", candidate_labels=["food review", "movie review", "code"])

# Text generation
gen = pipeline("text-generation", model="gpt2", device="cuda")
gen("Once upon a time,", max_new_tokens=50)

# Summarization
summ = pipeline("summarization", model="facebook/bart-large-cnn")
summ(long_text, max_length=150, min_length=30)

# Named entity recognition
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")
ner("Brian is a software engineer in Helena, Montana.")

# Embeddings via sentence-transformers is easier than pipeline; see Module 02.

# Chat with a specific model
chat = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device="cuda")
messages = [{"role": "user", "content": "Explain BPE in one sentence."}]
chat(messages)
```

Pipelines handle tokenization, forwarding, and decoding for you. Good for quick tests; move to explicit `model.generate()` when you need control.

---

### 2.4 `datasets` — Data at Scale

**HuggingFace `datasets`** is a memory-efficient, streaming-capable dataset library backed by Apache Arrow.

**Loading:**

```python
from datasets import load_dataset

# From the Hub
ds = load_dataset("imdb")
# DatasetDict({train: Dataset(features..., num_rows: 25000), test: Dataset(...)})

train = ds["train"]
print(train[0])   # {'text': '...', 'label': 1}

# Local
ds = load_dataset("json", data_files="data/*.jsonl")
ds = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Streaming (doesn't load into RAM)
ds = load_dataset("c4", "en", split="train", streaming=True)
for row in ds.take(100):
    print(row["text"][:100])
```

**Transformations:**

```python
# Map: apply a function to each row (batched for speed)
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train = train.map(preprocess, batched=True, remove_columns=["text"])

# Filter
short = train.filter(lambda ex: len(ex["input_ids"]) < 100)

# Select subset
small = train.select(range(1000))

# Split
splits = train.train_test_split(test_size=0.1)

# Shuffle
train = train.shuffle(seed=42)

# Format for PyTorch
train.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(train, batch_size=32, shuffle=True)
```

**Save and push:**

```python
ds.save_to_disk("./my_dataset")
ds = load_from_disk("./my_dataset")

# Push to Hub
ds.push_to_hub("my-username/my-dataset", private=True)
```

**Efficient tokenization.** `.map(batched=True)` processes many rows per Python call — 10–100× faster than row-by-row. Always use it.

---

### 2.5 Fine-Tuning: Trainer, PEFT, and TRL

**The `Trainer` API** handles training loops:

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    bf16=True,
    dataloader_num_workers=4,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=my_compute_metrics,
)
trainer.train()
```

The `Trainer` handles gradient accumulation, mixed precision, distributed training, checkpointing, evaluation loops — all for you. Enormous productivity gain.

**PEFT (Parameter-Efficient Fine-Tuning).** LoRA and cousins:

```python
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # LLaMA-style
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()   # "trainable params: X (~1% of total)"

# Now train with Trainer as usual
```

**QLoRA** — LoRA on a 4-bit quantized base:

```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B",
    quantization_config=bnb,
    device_map="auto",
)
model = get_peft_model(model, peft_config)
# ... train
```

QLoRA lets you fine-tune huge models on single 48GB GPUs — one of the most impactful open-source contributions of the last few years.

**TRL (Transformer Reinforcement Learning)** — HF's library for **SFT**, **DPO**, **PPO/RLHF**, and **KTO**:

```python
from trl import SFTTrainer

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_ds,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
)
sft_trainer.train()

# DPO
from trl import DPOTrainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,   # will use model with adapters disabled
    tokenizer=tokenizer,
    args=args,
    train_dataset=preference_dataset,   # rows: {'prompt', 'chosen', 'rejected'}
    peft_config=peft_config,
    beta=0.1,
)
dpo_trainer.train()
```

**Serving fine-tuned models:**
- **Text Generation Inference (TGI)** — HF's optimized inference server. Batching, streaming, PagedAttention.
- **vLLM** — the popular alternative; supports LoRA adapters natively.
- **llama.cpp / GGUF** — CPU + consumer-GPU inference; convert HF models to GGUF.
- **Ollama** — llama.cpp wrapper for easy local runs.

---

## 3. Mental Models & Analogies

### 3.1 The "GitHub for Models" Model

Every operation you'd do with code on GitHub, HF supports for models and data:
- Clone → `AutoModel.from_pretrained(id)`.
- Push → `.push_to_hub()`.
- Branches → revisions (git commits at specific versions).
- Issues → the "Discussions" tab.
- Actions → Spaces (hosted apps).
- Private repos → optional paid feature.

The Hub isn't just a download server — it's a full version-control ecosystem for AI artifacts. This is why "which model version were you using?" has a real answer: it's a git SHA.

### 3.2 The "Universal Wrapper" Model

`transformers` is a **universal wrapper** — it hides architecture-specific loading, tokenization, generation behind a single interface. You write code once against `AutoModel...`, and it works with LLaMA today and next month's release without changes.

This is only possible because HF has *manually implemented* each architecture inside the library — every new model family requires a modeling file. That's how they've stayed the standard for six years and counting. From your side: you get to pretend all decoder-only LLMs behave the same. From HF's side: they do the tedious work of unifying the interface.

Corollary: **when you use a very-recent model, use recent `transformers`** — architecture support lags a version or two behind release.

---

## 4. Common Pitfalls & Misconceptions

### 4.1 "I'll Use `pipeline` in Production"

`pipeline` is great for prototyping. In production:
- No control over batching.
- Overhead per call.
- Awkward for streaming.
- Awkward for LoRA adapter switching.

For production, use `AutoModel.from_pretrained()` + explicit `generate()` calls, or better, deploy via **TGI**, **vLLM**, or **sglang** which optimize batching and KV cache.

### 4.2 "device_map='auto' Solves All Multi-GPU Issues"

`device_map="auto"` uses `accelerate` to distribute layers across available GPUs. It works but:
- The default split may not be balanced (memory or compute).
- Slow inter-GPU communication if the model is too fragmented.
- Doesn't beat proper tensor/pipeline parallelism (Megatron, DeepSpeed) at scale.

For serious multi-GPU serving, learn tensor parallelism (vLLM handles this automatically).

### 4.3 "LoRA + Fine-Tuning ≈ Same Quality as Full Fine-Tuning"

Often yes, occasionally no. LoRA is enough for:
- Style / persona adaptation.
- Task-specific instruction following.
- Domain-specific vocabulary tightening.

Full fine-tuning may be needed for:
- Learning genuinely new abilities not in pretraining.
- Very large domain shifts.
- When quality gap of a few % matters.

Always A/B against a full-FT (or a stronger base) baseline before shipping.

---

## 5. Self-Assessment Bank (HuggingFace)

### Questions

**Q1 (Short answer).** What is the HuggingFace Hub, and how do you load a model + tokenizer for a causal LM?

**Q2 (Multiple choice).** Which class loads a decoder-only LLM (like LLaMA-3-8B-Instruct)?
- (a) `AutoModelForSequenceClassification`
- (b) `AutoModelForCausalLM`
- (c) `AutoModelForMaskedLM`
- (d) `AutoModelForSeq2SeqLM`

**Q3 (Short answer).** Why should you always use `tokenizer.apply_chat_template` instead of building the prompt string manually?

**Q4 (Multiple choice).** For batched generation with a decoder-only LLM, the tokenizer's `padding_side` should be:
- (a) "right"
- (b) "left"
- (c) either — no difference
- (d) not padded

**Q5 (Short answer).** What does `.map(batched=True)` do to speed up dataset preprocessing?

**Q6 (Multiple choice).** In `TrainingArguments`, `gradient_accumulation_steps=4` and `per_device_train_batch_size=8` gives an effective batch size of:
- (a) 8
- (b) 12
- (c) 32 (per device)
- (d) 32 × world_size

**Q7 (Short answer).** In one sentence, what does PEFT / LoRA let you do that full fine-tuning can't (practically)?

**Q8 (Multiple choice).** QLoRA differs from LoRA in that:
- (a) It uses a larger rank.
- (b) It runs on a 4-bit quantized base model, enabling much larger models to fit in memory.
- (c) It removes the LoRA update.
- (d) It only works for BERT.

**Q9 (Short answer).** Why is HuggingFace `pipeline` a great prototyping tool but often not the right production interface?

**Q10 (Multiple choice).** For serving a LoRA-fine-tuned LLaMA in production with high throughput and streaming:
- (a) Use `pipeline("text-generation")`.
- (b) Use `transformers.generate()` in a Python loop.
- (c) Use vLLM or Text Generation Inference (TGI) with LoRA adapter support.
- (d) Convert to ONNX.

---

### Answer Key & Detailed Explanations

**A1.** The Hub is a git-based registry for models, datasets, and spaces. Every entity is a repo at `huggingface.co/{owner}/{name}`. Load a causal LM:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto")
```

**A2. (b).** `AutoModelForCausalLM` is for autoregressive decoder-only LLMs (GPT, LLaMA, Mistral, Qwen). SequenceClassification (a) is for classification heads; MaskedLM (c) is for BERT-style; Seq2SeqLM (d) is for T5, BART.

**A3.** Every instruction-tuned model has its own chat template (role tokens, special separators). Applying the wrong format tanks quality silently. `apply_chat_template` reads the tokenizer's template config and formats correctly per model. Hand-building prompts is a common source of silent bugs.

**A4. (b).** Left-padding for causal LMs during batched inference: sequences are aligned at the right (end), so each sequence's next-token position is the same. Right-padding would leave the model generating from a pad token — bad results.

**A5.** `.map(fn, batched=True)` passes many rows at once (default 1000) to `fn`, which processes them in a single vectorized pass. Row-by-row calls are slow due to Python overhead; batched processing is 10–100× faster, especially for tokenization which is optimized C code that benefits from vectorization.

**A6. (c).** Per device: 8 × 4 = 32 effective batch. With `world_size` GPUs in DDP, total is 32 × world_size.

**A7.** LoRA / QLoRA enables fine-tuning **much larger models on much less GPU memory** by freezing the base model and only training small low-rank updates — often trainable-param count drops 100–1000×. This makes 70B model fine-tuning feasible on single-GPU consumer hardware.

**A8. (b).** QLoRA loads the base model in 4-bit precision (via `bitsandbytes` NF4), then trains LoRA adapters in bf16/fp16. The 4-bit base saves ~4× memory, letting a 70B model fit on a 48GB GPU that couldn't hold it in fp16. LoRA rank is orthogonal.

**A9.** `pipeline` is a high-level convenience wrapper — tokenize, forward, decode — with no control over batching, dynamic KV cache, streaming, or multi-adapter switching. Fine for quick prototypes. Production needs the finer control offered by `model.generate()` directly, or preferably a purpose-built serving stack (TGI, vLLM, sglang).

**A10. (c).** vLLM and TGI both support LoRA adapters, high-throughput batching, PagedAttention, and streaming out of the box. `pipeline` and naive `generate` loops are single-threaded and slow. ONNX is overkill and doesn't handle KV cache well for LLMs.

---

## 6. Practice Prompts

1. **Hub tour.** Clone the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model; run a single-generation and a batched-generation inference. Time both.
2. **Data pipeline.** Load the IMDB dataset, tokenize with `.map(batched=True)`, split, and produce a `DataLoader` ready for training. Time the pipeline.
3. **LoRA fine-tune.** LoRA-fine-tune TinyLlama on a small instruction dataset (Alpaca or your own JSONL). Confirm trainable-params is 1–2%. Compare pre/post fine-tune outputs.
4. **QLoRA on a 7B model.** Fine-tune `mistralai/Mistral-7B-Instruct-v0.3` with QLoRA on a task-specific dataset. Run inference and save the adapter.
5. **vLLM serving.** Serve the fine-tuned model with vLLM on GPU with streaming enabled. Load-test with concurrent requests.

---

## 7. References

- HuggingFace docs: [transformers](https://huggingface.co/docs/transformers), [datasets](https://huggingface.co/docs/datasets), [peft](https://huggingface.co/docs/peft), [trl](https://huggingface.co/docs/trl), [accelerate](https://huggingface.co/docs/accelerate).
- HuggingFace Course: [huggingface.co/course](https://huggingface.co/course).
- vLLM: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm).
- Text Generation Inference: [github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference).
- Dettmers et al., ["QLoRA"](https://arxiv.org/abs/2305.14314) (2023).
