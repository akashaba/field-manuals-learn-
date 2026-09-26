# Text Splitting Code Walkthrough

## Table of Contents
1. [Class-by-Class Breakdown](#class-by-class-breakdown)
2. [Algorithm Deep Dive](#algorithm-deep-dive)
3. [Examples Walkthrough](#examples-walkthrough)
4. [Best Practices](#best-practices)

---

## Class-by-Class Breakdown

### 1. TextSplitter (Base Class)

```javascript
export class TextSplitter {
    constructor({
        chunkSize = 1000,
        chunkOverlap = 200,
        lengthFunction = t => t.length,
        keepSeparator = false
    } = {}) {
        if (chunkOverlap >= chunkSize) {
            throw new Error('chunkOverlap must be less than chunkSize');
        }
        Object.assign(this, {chunkSize, chunkOverlap, lengthFunction, keepSeparator});
    }
```

#### Constructor Parameters

**`chunkSize`** (default: 1000)
- Maximum size of each chunk
- Measured by `lengthFunction` (characters or tokens)
- Think of it as the "target size" for chunks

**`chunkOverlap`** (default: 200)
- How much content to repeat between chunks
- Creates context continuity
- Must be less than `chunkSize`

**Example:**
```javascript
Text: "ABCDEFGHIJ" (10 chars)
chunkSize = 4
chunkOverlap = 1

Result:
Chunk 1: "ABCD"
Chunk 2:    "DEFG"  (D is overlap)
Chunk 3:       "GHIJ" (G is overlap)
```

**`lengthFunction`**
- How to measure text length
- Default: character count `t => t.length`
- For tokens: `t => Math.ceil(t.length / 4)`
- Allows flexible measurement strategies

**`keepSeparator`**
- Whether to include separators in chunks
- Usually `false` (remove separators)
- Example: Keep `\n\n` vs. remove it

#### Validation

```javascript
if (chunkOverlap >= chunkSize) {
    throw new Error('chunkOverlap must be less than chunkSize');
}
```

**Why?** If overlap ≥ size, chunks would infinitely repeat content.

```
chunkSize = 5, overlap = 5
Chunk 1: "ABCDE"
Chunk 2: "ABCDE" (same content!)
Chunk 3: "ABCDE" (infinite loop!)
```

---

### 2. Core Methods

#### `splitText()`

```javascript
splitText() {
    throw new Error('splitText() must be implemented by subclass');
}
```

**Abstract Method Pattern**
- Forces subclasses to implement their own logic
- Each splitter has unique splitting strategy
- Base class provides common infrastructure

---

#### `splitDocuments()`

```javascript
async splitDocuments(documents) {
    const chunks = [];
    for (const doc of documents) {
        chunks.push(...await this.createDocuments([doc.pageContent], [doc.metadata]));
    }
    return chunks;
}
```

**Purpose:** Main entry point for splitting multiple documents

**Step-by-Step:**
```
Input:  [Document1, Document2, Document3]
           ↓
Loop:   Document1 → createDocuments() → [Chunk1, Chunk2, Chunk3]
        Document2 → createDocuments() → [Chunk4, Chunk5]
        Document3 → createDocuments() → [Chunk6, Chunk7, Chunk8]
           ↓
Output: [Chunk1, Chunk2, ..., Chunk8]
```

**Key Points:**
- Preserves metadata from original documents
- Processes each document independently
- Flattens results into single array
- `async` for potential future async operations

---

#### `createDocuments()`

```javascript
async createDocuments(texts, metadatas = []) {
    const documents = [];
    for (let i = 0; i < texts.length; i++) {
        const text = texts[i];
        const metadata = metadatas[i] || {};
        const chunks = this.splitText(text);

        for (let j = 0; j < chunks.length; j++) {
            documents.push(
                new Document(chunks[j], {
                    ...metadata,
                    chunk: j,
                    totalChunks: chunks.length
                })
            );
        }
    }
    return documents;
}
```

**Purpose:** Convert text strings into Document objects with metadata

**Step-by-Step Flow:**
```
1. Input: texts=["full text"], metadatas=[{source: "pdf", page: 5}]
   
2. Split text:
   chunks = this.splitText("full text")
   → ["chunk1", "chunk2", "chunk3"]

3. Create Documents:
   For each chunk, create Document with:
   - pageContent: the chunk text
   - metadata: original + chunk info
   
4. Metadata enrichment:
   {
       source: "pdf",        // from original
       page: 5,              // from original
       chunk: 0,             // NEW: which chunk (0-indexed)
       totalChunks: 3        // NEW: total count
   }
```

**Why add chunk metadata?**
- Track position in original document
- Enable "next/previous chunk" navigation
- Useful for reassembly or debugging
- Helps with deduplication

---

#### `mergeSplits()` - The Core Algorithm

```javascript
mergeSplits(splits, separator) {
    const chunks = [];
    let current = [];
    let length = 0;

    for (const split of splits) {
        const splitLength = this.lengthFunction(split);
        const extraLength = current.length ? separator.length : 0;

        // finalize current chunk if it exceeds size
        if (length + splitLength + extraLength > this.chunkSize) {
            if (current.length) {
                chunks.push(this.joinSplits(current, separator));
            }

            // maintain overlap
            while (length > this.chunkOverlap && current.length) {
                length -= this.lengthFunction(current.shift()) + separator.length;
            }
        }

        current.push(split);
        length += splitLength + (current.length > 1 ? separator.length : 0);
    }

    if (current.length) {
        chunks.push(this.joinSplits(current, separator));
    }

    return chunks.filter(Boolean);
}
```

**Purpose:** Core algorithm that creates overlapping chunks

**Visual Example:**
```
splits = ["A", "B", "C", "D", "E", "F"]
chunkSize = 3 (chars)
chunkOverlap = 1 (char)
separator = ""

Iteration 1: Add "A"
  current = ["A"], length = 1

Iteration 2: Add "B"
  current = ["A", "B"], length = 2

Iteration 3: Add "C"
  current = ["A", "B", "C"], length = 3

Iteration 4: Try to add "D"
  length + "D" = 3 + 1 = 4 > chunkSize (3)
  → Finalize chunk: "ABC"
  → Remove splits until length ≤ overlap (1)
  → Remove "A", "B" → Keep "C"
  → current = ["C"], length = 1
  → Add "D": current = ["C", "D"], length = 2

Iteration 5: Add "E"
  current = ["C", "D", "E"], length = 3

Iteration 6: Try to add "F"
  length + "F" = 4 > chunkSize
  → Finalize chunk: "CDE"
  → Maintain overlap: current = ["E"]
  → Add "F": current = ["E", "F"]

Final: Add remaining: "EF"

Result: ["ABC", "CDE", "EF"]
         overlap→ C
                overlap→ E
```

**Key Algorithm Steps:**

1. **Accumulation Phase:**
   ```javascript
   current.push(split);
   length += splitLength + (current.length > 1 ? separator.length : 0);
   ```
    - Add split to current chunk
    - Track cumulative length
    - Account for separators between splits

2. **Overflow Check:**
   ```javascript
   if (length + splitLength + extraLength > this.chunkSize) {
   ```
    - Would adding this split exceed the limit?
    - If yes, finalize current chunk first

3. **Finalization:**
   ```javascript
   if (current.length) {
       chunks.push(this.joinSplits(current, separator));
   }
   ```
    - Join accumulated splits
    - Add to chunks array

4. **Overlap Maintenance:**
   ```javascript
   while (length > this.chunkOverlap && current.length) {
       length -= this.lengthFunction(current.shift()) + separator.length;
   }
   ```
    - Remove splits from the beginning
    - Keep removing until length ≤ chunkOverlap
    - This creates the overlap for next chunk

**Why `filter(Boolean)` at the end?**
- `joinSplits()` might return `null` for empty chunks
- Removes any null/undefined entries
- Ensures only valid chunks returned

---

#### `joinSplits()`

```javascript
joinSplits(splits, separator) {
    const text = splits.join(separator).trim();
    return text || null;
}
```

**Purpose:** Combine text fragments back into a single string

**Examples:**
```javascript
// With separator
joinSplits(["Hello", "World"], " ")
→ "Hello World"

// With newline separator
joinSplits(["Paragraph 1", "Paragraph 2"], "\n\n")
→ "Paragraph 1\n\nParagraph 2"

// Empty result
joinSplits(["", ""], " ")
→ null (because trim() makes it empty)
```

**Why return `null` for empty?**
- Signals "no valid content"
- Filtered out by `mergeSplits()`
- Prevents empty chunks in final result

---

### 3. CharacterTextSplitter

```javascript
export class CharacterTextSplitter extends TextSplitter {
    constructor({
        separator = '\n\n',
        chunkSize = 1000,
        chunkOverlap = 200,
        lengthFunction,
        keepSeparator = false
    } = {}) {
        super({chunkSize, chunkOverlap, lengthFunction, keepSeparator});
        this.separator = separator;
    }

    splitText(text) {
        const splits = text.split(this.separator).filter(s => s.trim().length > 0);
        return this.mergeSplits(splits, this.separator);
    }
}
```

**Purpose:** Simple splitting by a single separator

**How It Works:**

```
Input text:
"Paragraph 1\n\nParagraph 2\n\nParagraph 3"

Step 1: Split by separator ('\n\n')
→ ["Paragraph 1", "Paragraph 2", "Paragraph 3"]

Step 2: Filter empty strings
→ ["Paragraph 1", "Paragraph 2", "Paragraph 3"]

Step 3: Merge with overlap
→ Uses base class mergeSplits() algorithm
```

**Use Cases:**
- Splitting by paragraphs (`\n\n`)
- Splitting by lines (`\n`)
- Splitting by sections (custom separator)

**Example:**
```javascript
const splitter = new CharacterTextSplitter({
    separator: '\n\n',
    chunkSize: 500,
    chunkOverlap: 50
});

const text = "Para 1\n\nPara 2\n\nPara 3\n\nPara 4";
const chunks = splitter.splitText(text);
// If each para is ~150 chars:
// Chunk 1: "Para 1\n\nPara 2\n\nPara 3" (450 chars)
// Chunk 2: "Para 3\n\nPara 4" (overlap from Para 3)
```

---

### 4. RecursiveCharacterTextSplitter (Most Important)

```javascript
export class RecursiveCharacterTextSplitter extends TextSplitter {
    constructor({
        separators = ['\n\n', '\n', '. ', ' ', ''],
        chunkSize = 1000,
        chunkOverlap = 200,
        lengthFunction,
        keepSeparator = false
    } = {}) {
        super({chunkSize, chunkOverlap, lengthFunction, keepSeparator});
        this.separators = separators;
    }
```

**Purpose:** Smart splitting that tries multiple separators hierarchically

**Separator Hierarchy:**
```
1. '\n\n'  → Paragraphs (largest semantic unit)
2. '\n'    → Lines
3. '. '    → Sentences
4. ' '     → Words
5. ''      → Characters (last resort)
```

**Why This Order?**
- Preserves larger semantic units when possible
- Falls back to smaller units only when necessary
- Maintains meaning and context

---

#### The `splitText()` Algorithm

```javascript
splitText(text) {
    const finalChunks = [];
    let separator = this.separators.at(-1);  // Default to last (empty string)
    let nextSeparators = [];

    // Choose appropriate separator
    for (let i = 0; i < this.separators.length; i++) {
        const sep = this.separators[i];
        if (text.includes(sep)) {
            separator = sep;
            nextSeparators = this.separators.slice(i + 1);
            break;
        }
    }

    const splits = text.split(separator).filter(Boolean);
    let temp = [];

    for (const s of splits) {
        if (this.lengthFunction(s) <= this.chunkSize) {
            temp.push(s);  // Accumulate good splits
        } else {
            // Split is too large
            if (temp.length) {
                finalChunks.push(...this.mergeSplits(temp, separator));
                temp = [];
            }

            if (nextSeparators.length === 0) {
                finalChunks.push(s);  // No more separators, keep as-is
            } else {
                // Recursively split with next separator
                const recursiveSplitter = new RecursiveCharacterTextSplitter({
                    ...this,
                    separators: nextSeparators
                });
                finalChunks.push(...recursiveSplitter.splitText(s));
            }
        }
    }

    if (temp.length) {
        finalChunks.push(...this.mergeSplits(temp, separator));
    }

    return finalChunks;
}
```

**Step-by-Step Walkthrough:**

**Example Text:**
```
"This is paragraph 1.\n\nThis is a very long paragraph that exceeds the chunk size limit and needs to be split further. It has multiple sentences.\n\nThis is paragraph 3."
```

**Execution:**

**Step 1: Choose Separator**
```javascript
for (let i = 0; i < this.separators.length; i++) {
    const sep = this.separators[i];
    if (text.includes(sep)) {
        separator = sep;
        nextSeparators = this.separators.slice(i + 1);
        break;
    }
}
```

- Check `\n\n`: ✓ Found!
- `separator = '\n\n'`
- `nextSeparators = ['\n', '. ', ' ', '']`

**Step 2: Split by Chosen Separator**
```javascript
const splits = text.split(separator).filter(Boolean);
```

Result:
```
splits = [
    "This is paragraph 1.",
    "This is a very long paragraph...",  // Too long!
    "This is paragraph 3."
]
```

**Step 3: Process Each Split**

```javascript
for (const s of splits) {
    if (this.lengthFunction(s) <= this.chunkSize) {
        temp.push(s);  // Good size, accumulate
    } else {
        // Too large, needs further splitting
    }
}
```

**Processing:**

1. **Split 1: "This is paragraph 1."** (22 chars)
    - ✓ Within limit (assuming chunkSize = 100)
    - Add to `temp`: `temp = ["This is paragraph 1."]`

2. **Split 2: "This is a very long paragraph..."** (150 chars)
    - ✗ Exceeds limit!
    - Finalize `temp`: `finalChunks.push(...mergeSplits(temp))`
    - `temp = []`
    - **Recursively split** with next separators `['\n', '. ', ' ', '']`

3. **Recursive Call:**
   ```javascript
   const recursiveSplitter = new RecursiveCharacterTextSplitter({
       ...this,
       separators: ['\n', '. ', ' ', '']  // Without '\n\n'
   });
   finalChunks.push(...recursiveSplitter.splitText(largeParagraph));
   ```

    - Try `\n`: Not found (no line breaks)
    - Try `. `: ✓ Found!
    - Split into sentences
    - Each sentence is now within limit

4. **Split 3: "This is paragraph 3."** (22 chars)
    - ✓ Within limit
    - Add to `temp`

**Step 4: Finalize Remaining**
```javascript
if (temp.length) {
    finalChunks.push(...this.mergeSplits(temp, separator));
}
```

**Visual Flow:**
```
Input:
┌─────────────────────────────────────────┐
│ Para1 \n\n HUGE_Para2 \n\n Para3        │
└─────────────────────────────────────────┘
              ↓ split by '\n\n'
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Para1   │  │ HUGE_Para│  │  Para3   │
│  (OK)    │  │   (TOO   │  │  (OK)    │
│          │  │   BIG!)  │  │          │
└──────────┘  └──────────┘  └──────────┘
                    ↓ recursive split by '. '
              ┌─────┬─────┬─────┐
              │Sent1│Sent2│Sent3│
              │(OK) │(OK) │(OK) │
              └─────┴─────┴─────┘
                    ↓
Final Chunks: [Para1, Sent1, Sent2, Sent3, Para3]
```

**Key Insights:**

1. **Greedy Separator Selection:**
    - Always use the largest separator that exists in text
    - Preserves semantic units

2. **Recursion for Large Splits:**
    - If a split is too large after using current separator
    - Try smaller separators recursively
    - Continues until all chunks fit

3. **Accumulation Strategy:**
    - Collect "good" splits in `temp`
    - Merge them together (with overlap)
    - Only recurse when absolutely necessary

---

### 5. TokenTextSplitter

```javascript
export class TokenTextSplitter extends TextSplitter {
    constructor({
        encodingName = 'cl100k_base',
        chunkSize = 1000,
        chunkOverlap = 200
    } = {}) {
        const lengthFunction = text => Math.ceil(text.length / 4);
        super({chunkSize, chunkOverlap, lengthFunction});
        this.encodingName = encodingName;
    }

    splitText(text) {
        const splitter = new RecursiveCharacterTextSplitter({
            separators: ['\n\n', '\n', '. ', ' ', ''],
            chunkSize: this.chunkSize,
            chunkOverlap: this.chunkOverlap,
            lengthFunction: this.lengthFunction
        });
        return splitter.splitText(text);
    }
}
```

**Purpose:** Split based on estimated token count instead of characters

**Token Estimation:**
```javascript
const lengthFunction = text => Math.ceil(text.length / 4);
```

**Approximation Rule:**
- 1 token ≈ 4 characters (rough estimate)
- Actual varies by language and tokenizer
- For production: use proper tokenizer (tiktoken, GPT-tokenizer)

**How It Works:**
```
Text: "Hello world" (11 chars)
Token estimate: Math.ceil(11 / 4) = 3 tokens

Actual GPT tokens: ["Hello", " world"] = 2 tokens
(Approximation is close enough for chunking)
```

**Why Approximate?**
- Proper tokenization is slow
- Good enough for determining chunk boundaries
- Exact token count happens later during embedding

**Use Cases:**
- Embedding models with token limits (e.g., 512 tokens)
- OpenAI API token limits
- Ensuring chunks fit in context windows

**Example:**
```javascript
const splitter = new TokenTextSplitter({
    encodingName: 'cl100k_base',  // GPT-4 encoding
    chunkSize: 500,                // 500 tokens
    chunkOverlap: 50               // 50 token overlap
});

// Internally uses RecursiveCharacterTextSplitter
// but with token-based length measurement
```

---

## Algorithm Deep Dive

### The Overlap Algorithm Explained

**Problem:** How do you create overlapping chunks efficiently?

**Solution:** The "Sliding Window with Memory" approach

```javascript
while (length > this.chunkOverlap && current.length) {
    length -= this.lengthFunction(current.shift()) + separator.length;
}
```

**Visual Example:**
```
Text: "A B C D E F G H I J"
chunkSize = 3 words
chunkOverlap = 1 word

Step 1: Build first chunk
current = ["A", "B", "C"]
length = 3
→ Finalize: "A B C"

Step 2: Maintain overlap
Remove from beginning until length ≤ 1:
  Remove "A": current = ["B", "C"], length = 2
  Remove "B": current = ["C"], length = 1 ✓
  
Step 3: Build next chunk starting with overlap
current = ["C"]  ← This is the overlap!
Add "D": current = ["C", "D"], length = 2
Add "E": current = ["C", "D", "E"], length = 3
→ Finalize: "C D E"

Result:
Chunk 1: "A B C"
Chunk 2:     "C D E"  ← "C" is shared
           ↑ overlap
```

**Why This Works:**
1. Keeps last N items from previous chunk
2. N determined by `chunkOverlap` setting
3. Ensures context continuity
4. Prevents information loss at boundaries

---

### Recursive Splitting Complexity

**Time Complexity Analysis:**

```
Best Case: O(n)
- Text splits cleanly with first separator
- No recursion needed

Worst Case: O(n * m)
- n = text length
- m = number of separators
- Must try all separators for each large chunk

Average Case: O(n * log m)
- Usually splits well with first 2-3 separators
```

**Space Complexity:**
```
O(n) - stores all chunks in memory
```

**Optimization Opportunities:**
1. Cache separator checks
2. Limit recursion depth
3. Stream large documents
4. Parallel processing for multiple docs

---

## Examples Walkthrough

### Example 1: Basic Character Splitting

```javascript
async function example1() {
    const url = 'https://arxiv.org/pdf/2402.19473';
    const documents = await OutputHelper.withSpinner(
        'Loading PDF...', 
        () => extractTextFromPDF(url, {splitPages: true})
    );

    OutputHelper.log.info('Using CharacterTextSplitter (500/50)');
    const splitter = new CharacterTextSplitter({
        chunkSize: 500, 
        chunkOverlap: 50
    });

    const chunks = await OutputHelper.withSpinner(
        'Splitting text...', 
        () => splitter.splitDocuments(documents)
    );

    OutputHelper.formatStats({
        Pages: documents.length,
        Chunks: chunks.length,
        AvgPerPage: (chunks.length / documents.length).toFixed(1),
        Splitter: 'CharacterTextSplitter'
    });
    
    chunks.slice(0, 3).forEach(OutputHelper.formatChunkPreview);
}
```

**What This Does:**

1. **Load PDF**: Fetches and parses PDF, splitting by pages
2. **Create Splitter**: CharacterTextSplitter with 500 char chunks, 50 char overlap
3. **Split Documents**: Each page → multiple chunks
4. **Display Stats**: Shows chunk counts and averages
5. **Preview Chunks**: Shows first 3 chunks

**Output Example:**
```
Loading PDF... ✓ Done
ℹ Using CharacterTextSplitter (500/50)
Splitting text... ✓ Done

Statistics:
  Pages: 22
  Chunks: 156
  AvgPerPage: 7.1
  Splitter: CharacterTextSplitter

[Chunk 1]
Chunk 1/7 | Page 1 | 487 chars
Retrieval-Augmented Generation for AI-Generated Content...
```

**Why 156 Chunks?**
```
22 pages × ~7 chunks/page = 154 chunks
(varies based on page content density)
```

---

### Example 2: Recursive Character Splitting

```javascript
async function example2() {
    const docs = await OutputHelper.withSpinner(
        'Loading PDF...', 
        () => extractTextFromPDF(url, {splitPages: true})
    );

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, 
        chunkOverlap: 200
    });

    const chunks = await splitter.splitDocuments(docs);
    const {avg, min, max, median} = OutputHelper.analyzeChunks(chunks);

    OutputHelper.formatStats({
        'Total Chunks': chunks.length,
        'Average Size': `${avg} chars`,
        'Min Size': `${min} chars`,
        'Max Size': `${max} chars`,
        'Median Size': `${median} chars`,
        'Chunk Size Limit': '1000',
        'Overlap': '200'
    });
}
```

**Key Difference from Example 1:**
- Uses **RecursiveCharacterTextSplitter** (smarter)
- Larger chunks (1000 vs 500)
- More overlap (200 vs 50)
- Analyzes chunk size distribution

**Why Median Matters:**
```
If chunks = [100, 200, 950, 980, 1000, 1000]
Average = 705 chars
Median = 965 chars

→ Median shows most chunks are near the limit
→ Average affected by a few small chunks
```

---

### Example 3: Token-Based Splitting

```javascript
async function example3() {
    const splitter = new TokenTextSplitter({
        chunkSize: 500,   // 500 TOKENS (not chars)
        chunkOverlap: 50
    });

    const chunks = await splitter.splitDocuments(docs);
    
    const tokens = chunks.map(c => splitter.lengthFunction(c.pageContent));
    const avg = Math.round(tokens.reduce((a, b) => a + b, 0) / tokens.length);
}
```

**Difference from Character Splitting:**
```
Character-based: chunkSize = 500 chars
Token-based:     chunkSize = 500 tokens ≈ 2000 chars

Same chunkSize number, different meaning!
```

**Why Use This?**
- OpenAI embeddings: 8191 token limit
- GPT-4 context: 128k token limit
- Need to fit within token budgets

---

### Example 4: Metadata Filtering

```javascript
async function example4() {
    const chunks = await splitter.splitDocuments(docs);
    
    // Filter by chunk position
    const firstTen = chunks.filter(c => c.metadata.chunk < 10);
    
    // Simulated retrieval
    const query = 'retrieval augmented generation';
    const relevant = chunks.filter(c => 
        c.pageContent.toLowerCase().includes(query)
    ).slice(0, 3);
}
```

**What This Shows:**
1. **Metadata is preserved** through splitting
2. **Can filter by chunk position** (useful for pagination)
3. **Simulates keyword search** (real RAG uses embeddings)

**Metadata Structure:**
```javascript
{
    source: "https://arxiv.org/pdf/2402.19473",
    pdf: {...},
    loc: {pageNumber: 5},
    chunk: 2,          // This is chunk 2
    totalChunks: 8     // Out of 8 total from this page
}
```

---

### Example 5: Strategy Comparison

```javascript
async function example5() {
    const strategies = [
        {name: 'Large (1500/150)', size: 1500, overlap: 150},
        {name: 'Medium (1000/200)', size: 1000, overlap: 200},
        {name: 'Small (500/50)', size: 500, overlap: 50}
    ];

    for (const s of strategies) {
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: s.size, 
            chunkOverlap: s.overlap
        });
        const chunks = await splitter.splitDocuments(docs);
        // Compare results...
    }
}
```

**Results Comparison:**

| Strategy | Chunks | Avg Size | Use Case |
|----------|--------|----------|----------|
| Large | 45 | 1350 | Max context, summarization |
| Medium | 67 | 920 | Balanced, general purpose |
| Small | 156 | 480 | Max precision, Q&A |

**Trade-offs:**
```
More Chunks:
  ✓ Better precision
  ✓ More granular retrieval
  ✗ Less context per chunk
  ✗ More compute (embeddings)

Fewer Chunks:
  ✓ More context
  ✓ Better for comprehension
  ✗ Less precise retrieval
  ✗ May exceed limits
```

---

## Best Practices

### 1. Choosing Chunk Size

```javascript
// Question Answering (need precision)
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50
});

// Summarization (need context)
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 400
});

// General Purpose (balanced)
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});
```

### 2. Choosing Overlap

```javascript
// Rule of thumb: 10-20% of chunk size
chunkSize: 1000 → chunkOverlap: 100-200
chunkSize: 500  → chunkOverlap: 50-100
chunkSize: 2000 → chunkOverlap: 200-400
```

### 3. Testing Different Strategies

```javascript
async function testStrategies(docs) {
    const configs = [
        {size: 500, overlap: 50},
        {size: 1000, overlap: 100},
        {size: 1000, overlap: 200},
        {size: 1500, overlap: 150}
    ];
    
    for (const config of configs) {
        const splitter = new RecursiveCharacterTextSplitter(config);
        const chunks = await splitter.splitDocuments(docs);
        
        // Measure retrieval quality
        const quality = await evaluateRetrieval(chunks, queries);
        console.log(`${config.size}/${config.overlap}: ${quality}`);
    }
}
```

### 4. Respecting Document Structure

```javascript
// For markdown documents
const splitter = new RecursiveCharacterTextSplitter({
    separators: [
        '\n## ',      // H2 headers
        '\n### ',     // H3 headers
        '\n\n',       // Paragraphs
        '\n',         // Lines
        '. '          // Sentences
    ]
});

// For code
const splitter = new RecursiveCharacterTextSplitter({
    separators: [
        '\n\nclass ',  // Class definitions
        '\n\ndef ',    // Function definitions
        '\n\n',        // Blank lines
        '\n',          // Lines
        ' '            // Words
    ]
});
```