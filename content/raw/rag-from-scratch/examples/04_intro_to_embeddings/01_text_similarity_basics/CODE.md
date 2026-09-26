# Text Similarity Basics - Code Walkthrough

This document provides a detailed, step-by-step explanation of the text similarity example code.

## Table of Contents

1. [Overview](#overview)
2. [Imports and Setup](#imports-and-setup)
3. [Sample Data](#sample-data)
4. [Core Functions](#core-functions)
5. [Examples Breakdown](#examples-breakdown)
6. [Key Concepts](#key-concepts)

---

## Overview

This example demonstrates **semantic text similarity** using embeddings. Instead of matching keywords, we convert text into numerical vectors that capture meaning. Similar texts have similar vectors, allowing us to find relevant documents even when they use different words.

**What you'll learn:**
- How to convert text into embeddings (numerical vectors)
- How to measure similarity between texts using cosine similarity
- Why embeddings are fundamental to RAG (Retrieval Augmented Generation) systems

---

## Imports and Setup

```javascript
import {fileURLToPath} from "url";
import path from "path";
import {getLlama} from "node-llama-cpp";
import {Document} from '../../02_data_loading/example.js';
import {OutputHelper} from "../../helpers/output-helper.js";
import chalk from "chalk";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
```

### What's happening here?

- **`fileURLToPath` & `path`**: Handle file paths in ES modules
- **`getLlama`**: The main interface to node-llama-cpp for loading embedding models
- **`Document`**: A class from earlier in the tutorial that wraps text with metadata
- **`OutputHelper`**: Utility for pretty console output (spinners, formatting, colors)
- **`chalk`**: Library for colored terminal output
- **`__dirname`**: Gets the current directory (needed because ES modules don't have `__dirname` by default)

---

## Sample Data

```javascript
const sampleTexts = [
    "The sky is clear and blue today",
    "I love eating pizza with extra cheese",
    "Dogs love to play fetch with their owners",
    "The capital of France is Paris",
    "Drinking water is important for staying hydrated",
    "Mount Everest is the tallest mountain in the world",
    "A warm cup of tea is perfect for a cold winter day",
    "Painting is a form of creative expression",
    "Not all the things that shine are made of gold",
    "Cleaning the house is a good way to keep it tidy"
];

const documents = sampleTexts.map((text, i) =>
    new Document(text, { source: 'sample_data' })
);
```

### Why these texts?

These 10 diverse sentences cover different topics:
- Weather (sky)
- Food (pizza)
- Animals (dogs)
- Geography (France, Mount Everest)
- Health (water)
- etc.

This variety helps demonstrate how embeddings can distinguish between different semantic categories.

### Why wrap in Document objects?

Using `Document` objects (instead of plain strings) maintains consistency with the rest of the RAG tutorial. Each document has:
- **`pageContent`**: The actual text
- **`metadata`**: Additional info (id, source, etc.)

This pattern prepares you for working with real documents later.

---

## Core Functions

### 1. Initialize Embedding Model

```javascript
async function initializeEmbeddingModel() {
    const llama = await getLlama({
        logLevel: 'error' // Only show errors, not warnings or info
    });
    const model = await llama.loadModel({
        modelPath: path.join(__dirname, "bge-small-en-v1.5.Q8_0.gguf")
    });
    return await model.createEmbeddingContext();
}
```

**Step-by-step:**

1. **`getLlama()`**: Initialize the llama.cpp runtime
    - `logLevel: 'error'` suppresses verbose logs for cleaner output

2. **`loadModel()`**: Load the embedding model from disk
    - Uses `bge-small-en-v1.5` (BAAI General Embedding)
    - `.Q8_0.gguf` = 8-bit quantized GGUF format (smaller, faster)

3. **`createEmbeddingContext()`**: Create a context for generating embeddings
    - This is what we'll use to convert text → vectors

**Why this model?**
- **Small**: ~37MB, runs on any machine
- **Quality**: Good accuracy for English text
- **Tutorial-friendly**: Fast enough for learning

---

### 2. Embed Documents

```javascript
async function embedDocuments(context, documents) {
    const embeddings = new Map();

    await Promise.all(
        documents.map(async (document) => {
            const embedding = await context.getEmbeddingFor(document.pageContent);
            embeddings.set(document, embedding);
        })
    );

    return embeddings;
}
```

**Step-by-step:**

1. **Create a Map**: We'll store `Document → Embedding` pairs
    - Using a Map (not an array) allows efficient lookups

2. **`Promise.all()`**: Process all documents in parallel
    - Much faster than processing one-by-one
    - For 10 documents, this is ~10x faster

3. **`getEmbeddingFor()`**: Convert each text into an embedding vector
    - Input: `"Mount Everest is the tallest mountain in the world"`
    - Output: Array of 384 numbers, e.g., `[0.023, -0.156, 0.891, ...]`

4. **Store in Map**: Associate each document with its embedding

**Why parallel?**
```javascript
// Sequential (slow)
for (const doc of documents) {
    await embedDocument(doc); // Wait for each
}

// Parallel (fast)
await Promise.all(
    documents.map(doc => embedDocument(doc)) // All at once
);
```

---

### 3. Find Similar Documents

```javascript
function findSimilarDocuments(queryEmbedding, documentEmbeddings, topK = 3) {
    const similarities = [];

    for (const [document, embedding] of documentEmbeddings) {
        const similarity = queryEmbedding.calculateCosineSimilarity(embedding);
        similarities.push({ document, similarity });
    }

    similarities.sort((a, b) => b.similarity - a.similarity);

    return similarities.slice(0, topK);
}
```

**Step-by-step:**

1. **Loop through all documents**: Compare query to each document

2. **Calculate cosine similarity**:
   ```javascript
   queryEmbedding.calculateCosineSimilarity(embedding)
   ```
    - Returns a number between -1 and 1
    - **1** = identical meaning
    - **0** = no similarity
    - **-1** = opposite meaning (rare)

3. **Store results**: Keep document + similarity score together

4. **Sort by similarity**: Highest scores first
   ```javascript
   .sort((a, b) => b.similarity - a.similarity)
   ```

5. **Return top K**: Only return the best matches
   ```javascript
   .slice(0, topK) // Get first 3 results
   ```

**Cosine Similarity Explained:**

Imagine two arrows in 384-dimensional space:
- If they point in the same direction → similarity = 1
- If they're perpendicular → similarity = 0
- If they point opposite → similarity = -1

```
Query:    ────────→
Document: ────────→  (similarity = 1.0, same direction)

Query:    ────────→
Document:     ↓      (similarity = 0.0, perpendicular)
```

---

## Examples Breakdown

### Example 1: Basic Text Similarity

```javascript
async function example1() {
    // 1. Load model
    const context = await OutputHelper.withSpinner(
        'Loading embedding model...',
        () => initializeEmbeddingModel()
    );

    // 2. Embed all documents
    const documentEmbeddings = await OutputHelper.withSpinner(
        'Creating embeddings...',
        () => embedDocuments(context, documents)
    );

    // 3. Define query
    const query = "What is the tallest mountain on Earth?";

    // 4. Embed query
    const queryEmbedding = await context.getEmbeddingFor(query);

    // 5. Find similar documents
    const topResults = findSimilarDocuments(queryEmbedding, documentEmbeddings, 3);

    // 6. Display results
    topResults.forEach((result, index) => {
        console.log(`${index + 1}. [Similarity: ${result.similarity.toFixed(4)}]`);
        console.log(`   "${result.document.pageContent}"`);
    });
}
```

**The Magic Moment:**

Query: `"What is the tallest mountain on Earth?"`

Top result: `"Mount Everest is the tallest mountain in the world"` (similarity: ~0.65)

**Why is this impressive?**
- Query uses: "tallest mountain on **Earth**"
- Document uses: "tallest mountain in the **world**"
- No exact keyword match, but embeddings understand they mean the same thing!

**Traditional keyword search would fail here** because "Earth" ≠ "world" as strings.

---

### Example 2: Multiple Queries

```javascript
const queries = [
    "Tell me about hydration",
    "What's a good winter drink?",
    "Information about European capitals"
];
```

**What this demonstrates:**

1. **Reusability**: Documents are embedded ONCE, then reused
   ```javascript
   // Embed documents (expensive, do once)
   const documentEmbeddings = await embedDocuments(context, documents);
   
   // Query multiple times (cheap)
   for (const query of queries) {
       const queryEmbedding = await context.getEmbeddingFor(query);
       // Find matches in pre-computed embeddings
   }
   ```

2. **Efficiency pattern for RAG**:
    - Store document embeddings in a database
    - For each user query, only embed the query (fast)
    - Search pre-computed embeddings (very fast)

**Real-world analogy:**
- Instead of reading 1000 books for each question (slow)
- You index the books once (one-time cost)
- Then quickly look up relevant pages (fast)

---

### Example 3: Understanding Embedding Vectors

```javascript
const sampleDoc = documents[0];
const embedding = await context.getEmbeddingFor(sampleDoc.pageContent);

console.log('Vector Dimensions:', embedding.vector.length); // 384
console.log('First 10 values:', embedding.vector.slice(0, 10));
// [0.0234, -0.1567, 0.8912, -0.0123, ...]
```

**What's actually in an embedding?**

```javascript
Document: "The sky is clear and blue today"

Embedding: [
    0.0234,   // Position 0
   -0.1567,   // Position 1
    0.8912,   // Position 2
   -0.0123,   // Position 3
    // ... 380 more numbers
]
```

Each of the 384 numbers represents a "feature" learned by the model:
- Position 0 might capture "positivity"
- Position 1 might capture "time-relatedness"
- Position 2 might capture "nature-relatedness"
- etc.

**Storage calculation:**
```javascript
384 dimensions × 4 bytes per float = 1,536 bytes ≈ 1.5 KB per document
```

For 1 million documents: 1.5 GB of embeddings

---

### Example 4: Similarity Score Distribution

```javascript
const allResults = findSimilarDocuments(
    queryEmbedding, 
    documentEmbeddings, 
    documents.length  // Return ALL documents, not just top 3
);

allResults.forEach((result, index) => {
    const score = result.similarity.toFixed(4);
    const bar = '█'.repeat(Math.round(result.similarity * 30));
    console.log(`${index + 1}. ${score} ${bar}`);
});
```

**Sample output:**
```
Query: "Tell me about nature and weather"

 1. 0.6234 ██████████████████
    "The sky is clear and blue today"

 2. 0.5421 ████████████████
    "A warm cup of tea is perfect for a cold winter day"

 3. 0.2134 ██████
    "Mount Everest is the tallest mountain in the world"

 4. 0.1023 ███
    "The capital of France is Paris"
```

**What this shows:**

1. **High scores (>0.5)**: Strong semantic match
2. **Medium scores (0.2-0.5)**: Some relevance
3. **Low scores (<0.2)**: Weak or no relevance

**Distribution insight:**
- Not all documents are equally relevant
- Clear separation between relevant and irrelevant
- This is what makes RAG work: we can filter out noise

---

### Example 5: Comparing Different Queries

```javascript
const queryPairs = [
    {
        type: 'Keyword vs Semantic',
        queries: [
            "mountain tallest",                        // Keyword-style
            "What is the highest peak in the world?"   // Natural language
        ]
    }
];
```

**Testing query formulations:**

| Query Type | Query | Best Match | Score |
|------------|-------|------------|-------|
| Keyword | "mountain tallest" | Mount Everest... | 0.6523 |
| Natural | "What is the highest peak?" | Mount Everest... | 0.6498 |

**Key finding:** Both styles work well!
- Keywords: Simple, direct
- Natural language: More context, still effective

**Why this matters:**
- Users can search naturally ("What's a good drink for winter?")
- No need to craft perfect keywords
- Embeddings understand intent, not just words
