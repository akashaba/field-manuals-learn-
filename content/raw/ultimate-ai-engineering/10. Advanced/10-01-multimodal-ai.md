# 10.01 — Multimodal AI (Vision, Documents, Audio)

> **Module 10: Advanced** · Subtopic 1 of 4
> **Prerequisites:** 01.01 (attention), 01.03 (tokenization), 03.01 (embeddings, contrastive learning), 04.x (RAG), 06.01 (evals), 07.03 (prompt injection), 08.02 (cost engineering).
> **Outcome:** you can explain how vision-language models turn pixels into tokens, predict and control image-token cost, and build document pipelines that route each page to the cheapest method that preserves its meaning. You can also retrieve over visually rich documents with late interaction, transcribe and evaluate audio (committee hearings), and secure multimodal inputs.

> **Status check (September 2026).**
> - Frontier APIs accept images and PDFs natively.
> - **Image cost is per patch, and the patch rules differ by provider and model.**
>   - Anthropic's docs now specify $\lceil w/28\rceil \times \lceil h/28\rceil$ visual tokens.
>   - Anthropic images are downscaled to a **1568 px long edge / 1568-token cap**, or **2576 px / 4784 tokens** for high-resolution models (Claude 4.7 and later).
>   - OpenAI uses 32-px patches with a per-model multiplier on newer models, and a tile rule on GPT-4o-class models.
> - **Re-verify the rules for your exact model before budgeting.** The estimator below takes a provider profile as input.

---

## 1. How Vision-Language Models See

```
 image ──► [resize / tile to the model's resolution policy]
             │
             ▼
        VISION ENCODER (ViT): split into p×p patches → N = ⌈H/p⌉·⌈W/p⌉ patch embeddings → transformer
             │                   (CLIP/SigLIP-pretrained; often 14–16 px patches, then 2×2 merge ≈ 28 px effective)
             ▼
        PROJECTOR / CONNECTOR: MLP (LLaVA) · Q-Former/resampler to fixed K tokens (BLIP-2) · pixel-shuffle merge
             │
             ▼
        LLM DECODER: image tokens interleaved with text tokens; 2-D / multimodal RoPE keeps spatial layout (Qwen2-VL M-RoPE)
 Alternatives: cross-attention into the LM (Flamingo) · early fusion, one tokenizer for all modalities (Chameleon-style)
```

**Contrastive pretraining** (CLIP, SigLIP) aligns the image and text encoders. For a batch of $B$ pairs with normalised embeddings $u_i$ (image) and $v_i$ (text):

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2B}\sum_{i=1}^{B}\left[-\log\frac{e^{u_i^\top v_i/\tau}}{\sum_j e^{u_i^\top v_j/\tau}} \;-\;\log\frac{e^{u_i^\top v_i/\tau}}{\sum_j e^{u_j^\top v_i/\tau}}\right].
$$

SigLIP replaces the softmax with **pairwise sigmoid losses**, which removes the need for huge global batches.

```python
import numpy as np

def clip_loss(img_emb, txt_emb, tau=0.07):
    """Symmetric InfoNCE: matched (image_i, text_i) pairs are positives, all other pairs negatives."""
    I = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    T = txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)
    logits = I @ T.T / tau
    def ce(l):
        l = l - l.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(l).sum(axis=1)) - np.diag(l)))
    return (ce(logits) + ce(logits.T)) / 2, logits

rng = np.random.default_rng(0)
B, d = 256, 64
img = rng.standard_normal((B, d))
la, logits = clip_loss(img, img + 1.2 * rng.standard_normal((B, d)))     # aligned pairs
lr_, _ = clip_loss(img, rng.standard_normal((B, d)))                        # unrelated pairs
r_at_1 = float(np.mean(logits.argmax(axis=1) == np.arange(B)))
print(f"CLIP loss aligned={la:.3f} unrelated={lr_:.3f} (ln B={np.log(B):.3f}); image->text R@1={r_at_1:.2f}")
assert la < 1.0 < lr_ and r_at_1 > 0.9
```

**What this means for engineers:**
- **Resolution is a quality and cost knob.** Small text such as bill line numbers, footnotes, and fiscal table cells needs pixels. Downscaling loses them silently.
- **Spatial reasoning is weaker than recognition.** Counting, reading order in multi-column layouts, and table cell alignment fail more often than "what does this say".
- **VLM OCR is generative.** It can **hallucinate plausible text**, whereas classic OCR fails visibly with garbled characters. Treat VLM transcription of numbers as untrusted until it is verified (§4, §5).

---

## 2. Image Token Economics

For patch-based models (verified formula for Anthropic; OpenAI's newer models use $p=32$ times a model multiplier):

$$
\text{tokens}(w,h) = \Big\lceil \tfrac{w'}{p} \Big\rceil \cdot \Big\lceil \tfrac{h'}{p} \Big\rceil \cdot m,\qquad (w',h') = \text{downscale}(w,h \mid \text{long-edge cap},\ \text{token cap}).
$$

```python
import math

PROFILES = {
    # Anthropic docs (Sep 2026): ceil(w/28)*ceil(h/28), downscaled to fit long-edge and token caps
    "claude-standard": dict(kind="patch", patch=28, mult=1.0, max_long=1568, max_tokens=1568),
    "claude-highres":  dict(kind="patch", patch=28, mult=1.0, max_long=2576, max_tokens=4784),
    # Legacy GPT-4o-class "high" detail: fit 2048^2, shortest side -> 768, 85 + 170 per 512-px tile
    "gpt4o-tile":      dict(kind="tile", base=85, per_tile=170),
}

def image_tokens(w, h, profile):
    """Estimated input tokens for one image; pass a dict for other providers (patch, mult, caps from their docs)."""
    p = PROFILES[profile] if isinstance(profile, str) else profile
    if p["kind"] == "tile":
        s = min(1.0, 2048 / max(w, h)); w, h = w * s, h * s
        s = min(1.0, 768 / min(w, h)); w, h = w * s, h * s
        return p["base"] + p["per_tile"] * math.ceil(w / 512) * math.ceil(h / 512)
    s = min(1.0, p["max_long"] / max(w, h)); w, h = w * s, h * s
    n = math.ceil(w / p["patch"]) * math.ceil(h / p["patch"])
    while p.get("max_tokens") and n > p["max_tokens"]:
        s = math.sqrt(p["max_tokens"] / n) * 0.999
        w, h = w * s, h * s
        n = math.ceil(w / p["patch"]) * math.ceil(h / p["patch"])
    return math.ceil(n * p["mult"])

cases = {"letter page @150 DPI": (1275, 1650), "letter page @200 DPI": (1700, 2200),
         "1080p screenshot": (1920, 1080), "phone photo 12MP": (4032, 3024), "thumbnail": (256, 256)}
print(f"{'image':22s}" + "".join(f"{k:>17s}" for k in PROFILES))
for name, (w, h) in cases.items():
    print(f"{name:22s}" + "".join(f"{image_tokens(w, h, k):17d}" for k in PROFILES))
assert image_tokens(256, 256, "claude-standard") == 100
assert image_tokens(1700, 2200, "claude-standard") <= 1568 and image_tokens(1700, 2200, "claude-highres") <= 4784
assert image_tokens(1024, 1024, "gpt4o-tile") == 85 + 170 * 4
```

| Image | Claude standard | Claude high-res | GPT-4o tile rule |
|---|---|---|---|
| Letter page @150 DPI | 1,496 | 2,714 | 765 |
| Letter page @200 DPI | 1,496 (downscaled) | 4,758 | 765 |
| 1080p screenshot | 1,560 | 2,691 | 1,105 |
| 12 MP phone photo | 1,564 | 4,740 | 765 |

**The comparison that drives architecture:**
- A dense born-digital bill page is about **500–800 text tokens** via its text layer.
- As an image, the same page costs **1.5k–4.8k tokens**, 2–7× more, *and* is less exact.
- So **never send born-digital pages as images by default.** Use pixels only where the pixels carry meaning the text layer lacks (§3).
- High-resolution mode roughly **triples** the cost of a page. Enable it per page, for pages with small text or dense tables, rather than globally.

Cost control levers (see 08.02):
- **Crop to the region of interest** (a table or a signature block) before sending.
- **Resize deliberately.** 150 DPI is usually enough for 10–12 pt text, and footnotes need more.
- **Cache image prefixes** (provider prompt caching covers images in stable prefixes).
- **Batch APIs** for backfills.
- **Grayscale conversion** doesn't reduce patch count; don't expect savings from it.

---

## 3. Document AI: Route Each Page to the Cheapest Faithful Method

Legislative documents mix page types, and the right tool differs by type:

| Page type | Example | What carries the meaning | Method |
|---|---|---|---|
| Born-digital prose | Bill body text, statutes | Text layer | **PDF text extraction** (exact, cheapest) |
| **Amendment markup** | ~~stricken~~ and underlined new text in amended bills | **Visual formatting**, which plain text extraction drops | Vector-graphics analysis of PDF drawing ops (lines over text runs) **or a VLM** on that page |
| Tables | Fiscal-note revenue/expenditure tables | Row/column structure | **Layout model** (Docling, Azure Document Intelligence Layout) → Markdown/HTML tables |
| Scans | Historical session laws, signed letters | Pixels only | **OCR** (+ layout); VLM for low-quality scans with human review |
| Charts/figures | Fiscal projections | Visual encoding | **VLM** with a chart-specific extraction prompt and numeric verification |

Amendment markup is where naive pipelines fail silently. "The county ~~shall~~ may adopt…" becomes "The county shall may adopt…" after text extraction, and **the model reads both words as current law**. Detect strike/underline runs from the PDF's drawing operations, or render those pages for a VLM, and emit an explicit representation (`[DELETE: shall] [INSERT: may]`) before any downstream LLM sees the text.

```python
from dataclasses import dataclass

@dataclass
class Page:
    num: int
    text_chars: int            # characters in the PDF text layer (0 = scanned)
    image_frac: float          # fraction of page area covered by images
    tables: int                # table regions from a layout model
    markup_spans: int          # strike/underline runs detected in PDF drawing ops (amendment markup)
    charts: int = 0

COST_TOKENS = {"text": 700, "ocr": 900, "layout": 1200, "vlm": 1600 + 400}   # rough per-page input incl. prompt

def route_page(p: Page):
    """Cheapest pipeline that preserves the page's meaning. Order matters: markup and charts need pixels."""
    if p.markup_spans > 0:
        return "vlm", "amendment strike/underline carries meaning; text layer drops it"
    if p.charts > 0:
        return "vlm", "chart values only exist visually"
    if p.text_chars < 200 and p.image_frac > 0.5:
        return "ocr", "scanned page, no usable text layer"
    if p.tables > 0:
        return "layout", "tables need structure (layout model -> HTML/Markdown tables)"
    return "text", "born-digital prose; text layer is exact and cheapest"

pages = [Page(1, 3200, 0.0, 0, 0), Page(2, 2900, 0.0, 0, 14), Page(3, 2500, 0.1, 2, 0),
         Page(4, 40, 0.95, 0, 0), Page(5, 1800, 0.3, 0, 0, charts=1), Page(6, 3100, 0.0, 0, 0)]
plan = [(p.num, *route_page(p)) for p in pages]
for row in plan:
    print(row)
routed = sum(COST_TOKENS[r] for _, r, _ in plan)
print(f"routed {routed} tokens vs all-VLM {COST_TOKENS['vlm'] * len(pages)} ({routed / (COST_TOKENS['vlm'] * len(pages)):.0%})")
assert [r for _, r, _ in plan] == ["text", "vlm", "layout", "ocr", "vlm", "text"]
```

Routing costs **62% of an all-VLM pipeline** on this mix, and it is **more accurate**: the text layer is exact where it exists. On a typical corpus with mostly prose pages the saving is larger. The same router is where **human review** attaches: low-confidence OCR or VLM pages go to a review queue (05.06).

**Pipeline output contract.** Emit one canonical representation per document:
- Markdown/JSON with explicit structure (headings, sections with MCA numbers, tables, amendment operations);
- **per-element provenance**: page, bounding box, method, and confidence.

Chunking (04.01), extraction (02.03), and the knowledge graph (10.02) all consume this contract, never raw PDFs.

---

## 4. Retrieval over Visually Rich Documents

Text-only RAG fails when meaning lives in layout (tables, charts, forms). There are two options:

1. **Parse, then embed text.** Use the §3 pipeline, then the text RAG stack (04.x). It is cheapest, and best when parsing is reliable.
2. **Embed the page image directly** with **late interaction** (ColPali / ColQwen). A VLM produces one vector per patch, about 1,030 × 128-d per page. Scoring uses ColBERT's **MaxSim**:

$$
s(q, D) = \sum_{i \in q}\ \max_{j \in D}\ \mathbf{q}_i^{\top}\mathbf{d}_j .
$$

Each query token finds its best-matching patch, so a small table cell mentioned in the query isn't diluted by the rest of a busy page. That is the failure of single-vector (mean-pooled) page embeddings:

```python
import numpy as np

def maxsim(q, D):
    """Late interaction: sum over query tokens of the best-matching page patch."""
    return float((q @ D.T).max(axis=1).sum())

rng = np.random.default_rng(1)
d, n_concepts, n_pages, patches = 128, 400, 200, 256
concepts = rng.standard_normal((n_concepts, d)); concepts /= np.linalg.norm(concepts, axis=1, keepdims=True)

def page_embed(concept_ids):
    ids = rng.choice(concept_ids, patches)                          # each patch shows one concept, plus noise
    P = concepts[ids] + 0.3 * rng.standard_normal((patches, d)) / np.sqrt(d)
    return P / np.linalg.norm(P, axis=1, keepdims=True)

page_concepts = [rng.choice(n_concepts, 30, replace=False) for _ in range(n_pages)]   # busy pages: 30 topics each
pages = [page_embed(c) for c in page_concepts]
pooled = np.stack([p.mean(0) / np.linalg.norm(p.mean(0)) for p in pages])

hits_ms = hits_pool = 0
for t in range(n_pages):
    qc = rng.choice(page_concepts[t], 3, replace=False)            # the query mentions 3 things on the target page
    q = concepts[qc] + 0.3 * rng.standard_normal((3, d)) / np.sqrt(d)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    hits_ms += int(np.argmax([maxsim(q, P) for P in pages]) == t)
    qv = q.mean(0); qv /= np.linalg.norm(qv)
    hits_pool += int(np.argmax(pooled @ qv) == t)
print(f"R@1 late-interaction MaxSim = {hits_ms / n_pages:.2f} | single-vector mean-pooled = {hits_pool / n_pages:.2f}")
kb_multi, kb_single = 1030 * 128 * 2 / 1024, 1024 * 4 / 1024
print(f"storage/page: 1030x128 bf16 = {kb_multi:.0f} KiB vs one 1024-d fp32 vector = {kb_single:.0f} KiB; "
      f"100k pages = {1e5 * kb_multi / 2**20:.1f} GiB vs {1e5 * kb_single / 2**20:.2f} GiB")
assert hits_ms / n_pages > hits_pool / n_pages
```

**Result:** R@1 is **0.98 with MaxSim vs 0.45 with mean pooling** on busy pages. The price is storage: **258 KiB per page, about 64× more**, which is 24.6 GiB per 100k pages.

Production mitigations:
- token pooling or clustering of patch vectors (3–10× smaller);
- binary or product quantization (03.02);
- **two-stage retrieval**: a cheap single-vector or BM25 first stage, then MaxSim rerank of the top 100.

Vector DBs with multi-vector support (e.g. Vespa, Qdrant multivectors) run MaxSim natively. In pgvector you store patches as rows and compute MaxSim in the rerank stage.

**Answering over pages.** Retrieve page images, then send the **top 1–3 page images** plus the question to a VLM, with the instruction to cite the page and region. Verify extracted numbers against the parsed table where one exists (§5).

---

## 5. Audio and Speech: Committee Hearings as Data

Committee hearings and floor sessions are recorded. Turning them into searchable, citable text is high-value and entirely multimodal:

```
 audio/video ─► VAD + segmentation ─► ASR (Whisper-class / hosted STT) ─► diarization ("who spoke when")
            ─► speaker attribution (roster + chair cues + voice enrolment where lawful) ─► timestamped transcript
            ─► normalisation (numbers, bill IDs "House Bill twelve" → "HB 12", MCA cites) ─► chunk by speaker turn
            ─► summaries, motion/vote extraction, links back to video timecodes
```

**Metrics:**
- **WER** $=(S + D + I)/N$ for ASR.
- **DER** (diarization error rate) for speaker turns.
- **Task metrics:** motion and vote extraction F1, and bill-reference accuracy.
- For document question answering, **ANLS** (DocVQA's metric).
- For money and counts, **exact numeric match**. String-similarity metrics reward the wrong things, as the code shows:

```python
import re

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def _norm_words(s):
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).split()

def wer(reference, hypothesis):
    """(S + D + I) / N on lowercased, punctuation-stripped words."""
    r, h = _norm_words(reference), _norm_words(hypothesis)
    return levenshtein(r, h) / max(1, len(r))

def anls(prediction, gold_answers, tau=0.5):
    """DocVQA ANLS for one question: best (1 - normalised Levenshtein) over golds, zeroed if NL >= tau."""
    best, p = 0.0, prediction.strip().lower()
    for g in gold_answers:
        g = g.strip().lower()
        nl = levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best

def numeric_exact(prediction, gold):
    """Fiscal fields: compare the parsed numbers exactly."""
    f = lambda s: [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", s)]
    return f(prediction) == f(gold)

ref = "Mister Chairman, I move that House Bill 12 be amended on page 3, line 14."
hyp = "Mr chairman I move that house bill twelve be amended on page three line fourteen"
print(f"WER without number/abbreviation normalisation = {wer(ref, hyp):.2f}")
assert abs(wer("a b c d", "a x c d e") - 0.5) < 1e-9                   # 1 substitution + 1 insertion over 4 words
print("ANLS near-miss:", round(anls("$2,000,00", ["$2,000,000"]), 3),
      "| ANLS wrong magnitude ($200,000 vs $2,000,000):", round(anls("$200,000", ["$2,000,000"]), 3),
      "| numeric_exact:", numeric_exact("$200,000", "$2,000,000"))
assert anls("$200,000", ["$2,000,000"]) == 0.8 and not numeric_exact("$200,000", "$2,000,000")
assert numeric_exact("2,000,000 dollars", "$2,000,000")
```

**Lessons:**
- A transcript that is *semantically perfect* scores **WER 0.27** because "12" vs "twelve" and "Mr" vs "Mister" count as errors. **Normalise before scoring** (Whisper's English normaliser, or your own rules for bill IDs and numbers), or you will tune the wrong thing.
- **ANLS gives 0.8 credit for a 10× fiscal error.** For money use `numeric_exact`, and treat any numeric mismatch as a hard failure (08.04 §4).

**Realtime voice agents.** Examples are a constituent phone line and a hands-free drafting assistant. The latency budget is roughly:
- VAD end-of-turn (200–500 ms);
- ASR finalisation;
- LLM TTFT (08.02);
- TTS first audio.

**Speech-to-speech models** collapse the pipeline and cut latency, but lose the text-level control points: guardrails, logging, and redaction. Keep a text transcript side-channel for audit and safety. Barge-in (interruptions) and turn-taking are UX-critical.

---

## 6. Evaluating Multimodal Systems

| Layer | Metric | Notes |
|---|---|---|
| OCR / transcription | CER/WER after normalisation; **numeric exact match** on figures | Stratify by scan quality and font size |
| Layout | Table structure (TEDS), reading-order accuracy | Multi-column bill layouts, footnotes |
| Amendment markup | Operation-level F1 (DELETE/INSERT spans exact) | The critical legislative metric |
| Doc QA | ANLS plus answer exactness for numbers, with **grounding** (page and region cited correctly) | Judge + human spot-check (06.01) |
| Visual retrieval | nDCG@10, R@k on page-level labels | Compare parse-then-embed vs ColPali-style retrieval on *your* docs |
| Audio | WER, DER, motion/vote F1, timecode accuracy | Per speaker and per microphone condition |

Public benchmarks (MMMU, DocVQA, ChartQA, OCRBench) are **model-selection hints only**. Build a 200–500-item legislative multimodal eval set with a frozen hash (06.02) that covers each page type in §3.

---

## 7. Security and Privacy for Multimodal Inputs

- **Visual prompt injection.** Instructions can be rendered *inside images*: white-on-white text, tiny fonts, text in screenshots of web pages. VLMs read them. Treat every image and PDF as **untrusted content** under 07.03's containment rules. OCR the image and run the text through the same injection classifiers, and never let image-derived text authorise actions.
- **Metadata leakage.** EXIF can carry GPS, device, and timestamps; PDFs can carry authors, revision history, and embedded files. **Re-encode from pixels** before storage or model calls.
- **PII in pixels.** Signatures, ID cards, and addresses in scanned letters. Run redaction (07.01) on OCR text *and* apply pixel redaction (black boxes over detected regions) before the image leaves your trust boundary.
- **Resource exhaustion.** Decompression bombs and gigantic TIFFs. Enforce size and pixel limits, and decode in a sandbox (07.02 §6).

```python
import io
from PIL import Image

def sanitize_image(data: bytes, max_long_edge=1568, fmt="PNG") -> bytes:
    """Re-encode from pixels only: drops EXIF/GPS/XMP/ICC metadata and trailing payloads, bounds size
    (and therefore token cost), normalises mode. Run BEFORE storage or model calls."""
    Image.MAX_IMAGE_PIXELS = 80_000_000                                   # decompression-bomb guard
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        im = im.convert("RGB")
        s = min(1.0, max_long_edge / max(im.size))
        if s < 1.0:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        clean = Image.frombytes("RGB", im.size, im.tobytes())            # pixels only, no info dict
        out = io.BytesIO(); clean.save(out, format=fmt)
        return out.getvalue()

img = Image.new("RGB", (3000, 2000), (240, 240, 240))
exif = Image.Exif(); exif[0x010F] = "PhoneMaker"; exif[0x9003] = "2026:01:15 09:12:00"; exif[0x8825] = {1: "N", 2: (46.0, 35.0, 0.0)}
buf = io.BytesIO(); img.save(buf, format="JPEG", exif=exif.tobytes())
with Image.open(io.BytesIO(buf.getvalue())) as im0:
    print("before:", im0.size, "EXIF tags:", len(im0.getexif()))
with Image.open(io.BytesIO(sanitize_image(buf.getvalue()))) as im1:
    print("after: ", im1.size, "EXIF tags:", len(im1.getexif()), "info keys:", list(im1.info))
    assert len(im1.getexif()) == 0 and max(im1.size) == 1568
```

---

## 8. Production Challenges and Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **Amendment markup lost** | Model treats stricken text as law | Detect strike/underline from PDF drawing ops; VLM on markup pages; explicit `[DELETE]/[INSERT]` representation; op-level F1 eval |
| **Hallucinated OCR** | Plausible but wrong numbers from a VLM | Prefer the text layer; cross-check numbers against layout-model tables; numeric exact-match gates; human review for low confidence |
| **Image cost blowups** | Token spend spikes with scans | Page router (§3), crop, resize, per-page high-res only, caching, batch |
| **Small text unreadable** | Footnotes and table cells missed | Higher DPI or high-res mode for those pages; crop-and-zoom second pass |
| **Multi-column reading order** | Sentences spliced across columns | Layout model reading order; validate on known layouts |
| **Visual RAG storage** | Multi-vector index too large | Token pooling, quantization, two-stage retrieval |
| **Transcript names wrong** | Misattributed speakers | Roster-constrained attribution, chair cues, human correction UI |
| **Visual prompt injection** | Model follows text inside an image | Untrusted-content containment (07.03); OCR + classifiers; no action authority from images |
| **Metadata/PII leakage** | GPS or signatures leave the boundary | Pixel re-encode; pixel redaction; data classification |

---

## 9. Hands-On Projects

### Project 1 — Amendment-Aware Bill PDF Understanding

**User stories**
- *As a bill drafter*, I want every amendment operation in an introduced or amended bill extracted as structured DELETE/INSERT operations with section and line references, so that summaries and comparisons are always correct about current vs proposed law.

**Acceptance criteria**
- The pipeline accepts bill PDFs and emits canonical JSON: sections, MCA numbers, and amendment operations with page, line, and bbox provenance.
- Strike/underline detection works from PDF drawing operations, with a VLM fallback on flagged pages. The page router (§3) is logged per page.
- On a 150-bill labelled set, **operation-level F1 ≥ 0.97**, with **0 cases** of stricken text presented as current law in downstream summaries (checked by an automated assertion over the eval set).
- Cost per bill and per-page method mix are reported. The approach costs less than an all-VLM pipeline and is at least as accurate.

**Step-by-step**
1. Parse the PDFs (pdfplumber/PyMuPDF): text runs with coordinates, plus drawing ops (lines and rects).
2. Detect strike lines (horizontal lines through a text run's x-height middle) and underlines (below the baseline). Map them to text spans.
3. Build the page router and VLM fallback prompt (structured output, 02.03), then reconcile with the text layer.
4. Label the eval set (a drafter-reviewed sample) and evaluate the operation F1.
5. Integrate as the canonical input for 04.01 chunking and 10.02 graph extraction.

### Project 2 — Committee Hearing Transcription, Attribution, and Motion Extraction

**User stories**
- *As a committee secretary*, I want draft minutes with who said what, the motions made, and the votes, each linked to the video timestamp, so I only need to review and correct.

**Acceptance criteria**
- The ASR + diarization pipeline reports normalised WER ≤ 12% and DER ≤ 15% on a 10-hour labelled set, broken down by room and microphone condition.
- Speaker attribution is constrained to the committee roster and witness sign-in sheet, with ≥ 95% turn attribution accuracy after chair-cue heuristics.
- Motion/vote extraction F1 ≥ 0.9. Every extracted item carries a timecode and transcript span, and the minutes UI links to the video.
- Public-records and disclosure requirements are reviewed (08.05 §9). A human approval step is required before publication.

**Step-by-step**
1. Build the ASR (Whisper-class or hosted) + diarization pipeline with VAD segmentation.
2. Write normalisation rules (bill IDs, numbers, MCA cites) and apply the same normaliser for WER scoring.
3. Build the roster-constrained attribution and chair-cue parser ("The chair recognises Representative …").
4. Extract motions and votes (structured outputs), then run the eval.
5. Build the review UI with timecode links and a publication workflow.

### Project 3 — Visual RAG over Fiscal Notes and Reports

**User stories**
- *As a fiscal analyst*, I want to ask questions about tables and charts across fiscal notes and agency reports and get answers citing the exact page region.

**Acceptance criteria**
- Two systems are compared on 300 labelled questions: (A) parse-then-embed (§3 + 04.x) and (B) ColPali-style page retrieval + VLM answering.
- The report gives R@5, nDCG@10, answer exactness (numeric exact for figures), grounding accuracy, cost per query, and index size.
- The chosen system (or a hybrid: B as the fallback when A's parse confidence is low) meets **numeric exact ≥ 95%** on figure questions, with a page and region citation on every answer.

**Step-by-step**
1. Build the corpus and labelled questions by type (table lookup, chart trend, text).
2. Implement A with the existing stack.
3. Implement B: multi-vector index (token pooling), MaxSim rerank of the first stage's top 100, and VLM answering with a crop.
4. Evaluate both, build the hybrid router, and document the trade-offs.
5. Ship through the 08.05 gates.

---

## 10. Foundational Papers (exact titles)

- Dosovitskiy et al., 2020 — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- Radford et al., 2021 — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Zhai et al., 2023 — *Sigmoid Loss for Language Image Pre-Training* (SigLIP)
- Alayrac et al., 2022 — *Flamingo: a Visual Language Model for Few-Shot Learning*
- Li et al., 2023 — *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*
- Liu et al., 2023 — *Visual Instruction Tuning* (LLaVA)
- Wang et al., 2024 — *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution*
- Chameleon Team, 2024 — *Chameleon: Mixed-Modal Early-Fusion Foundation Models*
- Faysse et al., 2024 — *ColPali: Efficient Document Retrieval with Vision Language Models*
- Khattab & Zaharia, 2020 — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*
- Mathew, Karatzas & Jawahar, 2020 — *DocVQA: A Dataset for VQA on Document Images*
- Kim et al., 2021 — *OCR-free Document Understanding Transformer* (Donut)
- Auer et al., 2024 — *Docling Technical Report*
- Radford et al., 2022 — *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper)
- Bagdasaryan et al., 2023 — *Abusing Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs*

## 11. Essential Tooling

| Tool | Use |
|---|---|
| **PyMuPDF, pdfplumber** | Text layer, coordinates, drawing ops (strike/underline detection) |
| **Docling, Azure AI Document Intelligence (Layout), Unstructured** | Layout, tables, reading order, OCR |
| **Tesseract / PaddleOCR** | Classic OCR (visible failure modes) |
| **ColPali/ColQwen (`colpali-engine`), Vespa/Qdrant multivector** | Visual document retrieval |
| **Whisper / faster-whisper, pyannote.audio, hosted STT** | ASR and diarization |
| **Pillow, pikepdf** | Image re-encoding, PDF metadata stripping |
| **Frontier VLM APIs; open VLMs on vLLM (08.03)** | Page understanding, chart extraction |

**Image prompt (VLM pipeline):** *"Technical diagram: a scanned bill page split into a grid of 28×28 patches (highlighted), flowing into a 'Vision Encoder (ViT)' block, then 'Projector', producing a row of image tokens that interleave with text tokens into an 'LLM Decoder'. Side annotation: 'tokens = ⌈w/28⌉·⌈h/28⌉'. Flat vector, white background, blue/orange palette."*

**Image prompt (page router):** *"Flowchart: a stack of PDF pages enters a router diamond; five labelled lanes: 'Text layer' (prose), 'VLM' (strike/underline amendment markup — show a stricken word and underlined insert), 'Layout model' (table), 'OCR' (scanned page), 'VLM' (chart). Each lane ends at a common 'Canonical JSON with provenance' box. Clean infographic style."*
