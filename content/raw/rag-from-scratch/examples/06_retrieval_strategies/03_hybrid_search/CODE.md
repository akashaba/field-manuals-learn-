# Hybrid Search - Code Walkthrough

This guide explains the e-commerce hybrid search code step-by-step. By the end, you'll understand how to combine vector search (semantic understanding) with keyword search (exact matching) for product searches.

## Table of Contents

1. [What is Hybrid Search?](#what-is-hybrid-search)
2. [Setup and Imports](#setup-and-imports)
3. [Configuration](#configuration)
4. [Core Functions](#core-functions)
5. [Example 1: SKU and Brand Challenges](#example-1-sku-and-brand-challenges)
6. [Example 2: Score Normalization](#example-2-score-normalization)
7. [Example 3: Multi-Field Search](#example-3-multi-field-search)
8. [Example 4: Dynamic Weight Adjustment](#example-4-dynamic-weight-adjustment)
9. [Example 5: Fallback Strategies](#example-5-fallback-strategies)
10. [Example 6: Filter-Aware Search](#example-6-filter-aware-search)
11. [Example 7: Performance Optimization](#example-7-performance-optimization)

---

## What is Hybrid Search?

**Hybrid search** combines two different search methods:

1. **Vector Search** (Semantic)
   - Understands the *meaning* of words
   - Example: "laptop for editing" matches "MacBook Pro for video work"
   - Good for: Natural language, concepts, paraphrased queries

2. **Keyword Search** (BM25)
   - Matches *exact words*
   - Example: "MBP-M3MAX-32" only matches that exact SKU
   - Good for: Product codes, brand names, technical specs

**Why combine them?** Each has strengths and weaknesses. Hybrid search gives you the best of both worlds!

---

## Setup and Imports

```javascript
import { fileURLToPath } from "url";
import path from "path";
import { VectorDB } from "embedded-vector-db";
import { getLlama } from "node-llama-cpp";
import { Document } from "../../../src/index.js";
import { OutputHelper } from "../../../helpers/output-helper.js";
import chalk from "chalk";
```

**What each import does:**

- `fileURLToPath` and `path`: Help find files on your computer
- `VectorDB`: Database that stores product embeddings (vector representations)
- `getLlama`: Loads the AI model that creates embeddings
- `Document`: Wrapper class for product data
- `OutputHelper`: Utility for nice console output
- `chalk`: Makes console text colorful (not essential, just pretty!)

---

## Configuration

```javascript
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMBEDDING_MODEL_PATH = path.join(__dirname, "..", "..", "..", "models", "bge-small-en-v1.5.Q8_0.gguf");

const DIM = 384;
const MAX_ELEMENTS = 10000;
const NS = "product_search";
```

**What these constants mean:**

- `__dirname`: Current file's directory
- `EMBEDDING_MODEL_PATH`: Where the AI model file lives
- `DIM`: Embedding dimension (384 numbers represent each product)
- `MAX_ELEMENTS`: Maximum products the database can hold
- `NS`: Namespace (like a folder name in the database)

---

## Core Functions

### Function 1: Initialize Embedding Model

```javascript
async function initializeEmbeddingModel() {
    try {
        const llama = await getLlama({ logLevel: "error" });
        const model = await llama.loadModel({ modelPath: EMBEDDING_MODEL_PATH });
        return await model.createEmbeddingContext();
    } catch (error) {
        throw new Error(`Failed to initialize embedding model: ${error.message}`);
    }
}
```

**What it does:**
1. Loads the AI model from disk
2. Creates an "embedding context" (a tool to convert text → numbers)
3. Returns the embedding context so we can use it later

**Why we need it:** To convert product descriptions and search queries into vectors (arrays of numbers) that the computer can compare.

---

### Function 2: Create Product Catalog

```javascript
function createProductCatalog() {
    return [
        new Document("Apple MacBook Pro 16-inch with M3 Max chip...", {
            id: "PROD-001",
            title: "MacBook Pro 16-inch M3 Max",
            brand: "Apple",
            category: "laptops",
            price: 3499,
            sku: "MBP-M3MAX-32-1TB",
            attributes: "M3 Max, 32GB RAM, 1TB SSD..."
        }),
        // ... more products
    ];
}
```

**What it does:**
Creates an array of product documents. Each product has:
- **Content** (first parameter): Full description
- **Metadata** (second parameter): Structured data like SKU, price, brand

**Why structured this way:**
- Content is for semantic search (understanding meaning)
- Metadata fields are for exact matching and filtering

---

### Function 3: Add Products to Store

```javascript
async function addProductsToStore(vectorStore, embeddingContext, products) {
    try {
        // Enable full-text indexing on multiple fields
        await vectorStore.setFullTextIndexedFields(NS, 
            ['content', 'title', 'brand', 'sku', 'attributes']
        );

        for (const product of products) {
            // Convert product description to vector
            const embedding = await embeddingContext.getEmbeddingFor(product.pageContent);
            
            // Prepare metadata
            const metadata = {
                content: product.pageContent,
                ...product.metadata,
            };
            
            // Store in database
            await vectorStore.insert(
                NS,
                product.metadata.id,
                Array.from(embedding.vector),
                metadata
            );
        }
    } catch (error) {
        throw new Error(`Failed to add products to store: ${error.message}`);
    }
}
```

**Step-by-step:**

1. **`setFullTextIndexedFields`**: Tells the database which fields to index for keyword search
   - Like creating an index in a book for quick lookup

2. **Loop through products**: Process each product one by one

3. **`getEmbeddingFor`**: Convert product description to a vector (array of 384 numbers)
   - Example: "MacBook Pro 16-inch..." → [0.23, -0.45, 0.12, ...]

4. **`insert`**: Save the product with:
   - Namespace (folder)
   - Product ID
   - Vector (for semantic search)
   - Metadata (for keyword search and filtering)

---

## Example 1: SKU and Brand Challenges

This example shows why keyword search is critical for product codes.

### Test 1: Exact SKU Search

```javascript
const query1 = "MBP-M3MAX-32-1TB";
const queryEmbedding1 = await embeddingContext.getEmbeddingFor(query1);
const vectorResults1 = await vectorStore.search(NS, Array.from(queryEmbedding1.vector), 3);
const keywordResults1 = await vectorStore.fullTextSearch(NS, query1, 3);
```

**What happens:**

1. **Vector Search**:
   - Converts "MBP-M3MAX-32-1TB" to numbers
   - Problem: The model doesn't understand product codes!
   - Result: Might return random products

2. **Keyword Search (BM25)**:
   - Looks for exact text match
   - Finds products with "MBP-M3MAX-32-1TB" in SKU field
   - Result: Perfect match!

**Key Insight:**
```
SKU codes like "MBP-M3MAX-32-1TB" are "out-of-vocabulary" (OOV) for AI models.
The model treats them like random characters, but keyword search finds them perfectly.
```

### Test 2: Brand Name Search

```javascript
const query2 = "Sony headphones";
```

**What happens:**

- **Vector Search**: Finds Sony products + similar products (like Bose)
- **Keyword Search**: Prioritizes exact "Sony" matches

**Best approach:** Use BOTH (hybrid search) to get:
- Exact brand match (keyword)
- Semantic category understanding (vector)

---

## Example 2: Score Normalization

### The Problem

```javascript
const vecScores = vectorResults.map(r => r.similarity);
const keyScores = keywordResults.map(r => r.similarity);

// Vector scores might be: [0.85, 0.83, 0.80]
// Keyword scores might be: [12.5, 8.3, 5.1]
```

**Problem:** Different scales! How do you combine them fairly?

If you just add them: `0.85 + 12.5 = 13.35` (keyword dominates!)

### Solution 1: Min-Max Normalization

```javascript
const normalizeMinMax = (scores) => {
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const range = max - min || 1;
    return scores.map(s => (s - min) / range);
};
```

**What it does:**
- Scales all scores to [0, 1] range
- Formula: `(score - min) / (max - min)`
- Example: [12.5, 8.3, 5.1] → [1.0, 0.43, 0]

**Pros:** Simple, intuitive
**Cons:** Sensitive to outliers (one very high/low score affects everything)

### Solution 2: Z-Score Normalization

```javascript
const normalizeZScore = (scores) => {
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance) || 1;
    return scores.map(s => (s - mean) / stdDev);
};
```

**What it does:**
- Centers scores around 0
- Formula: `(score - mean) / standard_deviation`
- Example: [12.5, 8.3, 5.1] → [1.2, 0.1, -1.3]

**Pros:** Preserves distribution, handles outliers better
**Cons:** Can produce negative scores (but that's okay for comparison)

### Solution 3: Rank-Based Normalization

```javascript
const normalizeRank = (scores) => {
    const n = scores.length;
    return scores.map((_, idx) => (n - idx) / n);
};
```

**What it does:**
- Ignores actual scores, uses only ranking
- Formula: `(total - rank) / total`
- Example (for 5 results): [1.0, 0.8, 0.6, 0.4, 0.2]

**Pros:** Most robust, used by Reciprocal Rank Fusion (RRF)
**Cons:** Loses information about score gaps

---

## Example 3: Multi-Field Search

### Why Multiple Fields?

```javascript
await vectorStore.setFullTextIndexedFields(NS, 
    ['content', 'title', 'brand', 'sku', 'attributes']
);
```

**Each field serves a purpose:**

| Field | Purpose | Example Query |
|-------|---------|---------------|
| `content` | Semantic understanding | "laptop for video editing" |
| `title` | Product name matching | "MacBook Pro" |
| `brand` | Exact brand matching | "Apple" |
| `sku` | Product code matching | "MBP-M3MAX-32-1TB" |
| `attributes` | Technical specs | "32GB RAM" |

### How It Works

```javascript
const query = "Apple wireless keyboard";
```

**Search process:**
1. Searches ALL indexed fields simultaneously
2. "Apple" matches strongly in `brand` field
3. "wireless" matches in `title` and `attributes`
4. "keyboard" matches in `title` and `content`
5. Combines all signals for final score

**Result:** More comprehensive, accurate search!

---

## Example 4: Dynamic Weight Adjustment

### The analyzeQuery Function

```javascript
function analyzeQuery(query) {
    const upperCount = (query.match(/[A-Z]/g) || []).length;
    const digitCount = (query.match(/\d/g) || []).length;
    const hyphenCount = (query.match(/-/g) || []).length;
    const wordCount = query.split(/\s+/).length;
    const hasQuestionWord = /^(what|how|which|where|why|when|who)/i.test(query);

    // Detect query type and return optimal weights
    let queryType, weights;

    if (hyphenCount >= 2 || (upperCount > 3 && digitCount > 0)) {
        queryType = "SKU/Model Number";
        weights = { vector: 0.2, text: 0.8 };
    } else if (digitCount >= 3) {
        queryType = "Technical Specs";
        weights = { vector: 0.3, text: 0.7 };
    } else if (hasQuestionWord || wordCount >= 6) {
        queryType = "Natural Language Question";
        weights = { vector: 0.8, text: 0.2 };
    } else if (wordCount <= 2) {
        queryType = "Short Keyword";
        weights = { vector: 0.4, text: 0.6 };
    } else {
        queryType = "Mixed Query";
        weights = { vector: 0.5, text: 0.5 };
    }

    return { queryType, weights };
}
```

### Understanding the Logic

**Pattern Detection:**

1. **SKU/Model Number** (`hyphenCount >= 2`)
   - Example: "ASUS-G14-R9-4070"
   - Many hyphens + capitals + digits = product code
   - Use keyword-heavy: **0.2 vector / 0.8 text**

2. **Technical Specs** (`digitCount >= 3`)
   - Example: "32GB RAM 4K display"
   - Multiple numbers = technical requirements
   - Use keyword-heavy: **0.3 vector / 0.7 text**

3. **Natural Language Question** (starts with question word)
   - Example: "What's the best laptop for video editing?"
   - Needs semantic understanding
   - Use vector-heavy: **0.8 vector / 0.2 text**

4. **Short Keyword** (`wordCount <= 2`)
   - Example: "Sony headphones"
   - Brief, likely includes brand/category
   - Use keyword-leaning: **0.4 vector / 0.6 text**

5. **Mixed Query** (default)
   - Example: "portable charger fast charging"
   - Could benefit from both approaches
   - Use balanced: **0.5 vector / 0.5 text**

### How to Use It

```javascript
const query = "ASUS-G14-R9-4070";
const analysis = analyzeQuery(query);
// Returns: { queryType: "SKU/Model Number", weights: { vector: 0.2, text: 0.8 } }

const results = await vectorStore.hybridSearch(
    NS,
    Array.from(queryEmbedding.vector),
    query,
    {
        vectorWeight: analysis.weights.vector,  // 0.2
        textWeight: analysis.weights.text,      // 0.8
        k: 2
    }
);
```

**Why this matters:**
- Automatic optimization based on query type
- Better user experience without manual configuration
- Adapts to different search patterns

---

## Example 5: Fallback Strategies

### The Problem: Zero Results

```javascript
const query = "Microsoft Surface laptop touchscreen";
```

Problem: No Microsoft products in our catalog!

### Strategy Progression

#### Strategy 1: Pure Keyword Search (Likely Fails)

```javascript
const keywordOnly = await vectorStore.fullTextSearch(NS, query, 3);
// Result: 0 results (no exact "Microsoft" match)
```

**Why it fails:** No products contain the word "Microsoft"

#### Strategy 2: Balanced Hybrid Search

```javascript
const hybrid = await vectorStore.hybridSearch(
    NS,
    Array.from(queryEmbedding.vector),
    query,
    { k: 3 }  // Default: 0.5/0.5
);
// Result: Returns laptops with touchscreens (similar products)
```

**Why it works better:** Vector search understands "laptop" and "touchscreen" semantically

#### Strategy 3: Vector-Heavy Fallback

```javascript
const vectorFallback = await vectorStore.hybridSearch(
    NS,
    Array.from(queryEmbedding.vector),
    query,
    {
        vectorWeight: 0.9,
        textWeight: 0.1,
        k: 3
    }
);
// Result: Returns most semantically similar laptops
```

**Why use this:** When keyword matching completely fails, rely on semantic similarity

### Recommended Fallback Flow

```
1. Try balanced hybrid (0.5/0.5)
   ↓ (if < 3 results)
2. Try vector-heavy (0.8/0.2)
   ↓ (if still < 3 results)
3. Use pure vector (1.0/0.0)
   ↓
4. Show "Similar products" message to user
```

**In code:**

```javascript
let results = await hybridSearch(query, { vectorWeight: 0.5, textWeight: 0.5, k: 10 });

if (results.length < 3) {
    // Fallback to vector-heavy
    results = await hybridSearch(query, { vectorWeight: 0.8, textWeight: 0.2, k: 10 });
}

if (results.length < 3) {
    // Final fallback: pure vector
    results = await hybridSearch(query, { vectorWeight: 1.0, textWeight: 0.0, k: 10 });
    // Show message: "No exact matches. Here are similar products:"
}
```

---

## Example 6: Filter-Aware Search

### Combining Search with Business Logic

```javascript
const query = "high-end laptop professional work";
const maxPrice = 2500;
```

**Two-step process:**

#### Step 1: Hybrid Search

```javascript
const results = await vectorStore.hybridSearch(
    NS,
    Array.from(queryEmbedding.vector),
    query,
    { k: 5 }
);
```

This finds relevant products based on the query.

#### Step 2: Apply Business Filters

```javascript
// Filter by price
const priceFiltered = results.filter(doc => doc.metadata.price <= maxPrice);

// Filter by category AND price
const categoryFiltered = results.filter(doc =>
    doc.metadata.category === 'laptops' && 
    doc.metadata.price <= maxPrice
);
```

### Common E-Commerce Filters

```javascript
// Price range
const inBudget = results.filter(r => 
    r.metadata.price >= minPrice && 
    r.metadata.price <= maxPrice
);

// In stock
const available = results.filter(r => 
    r.metadata.inStock === true
);

// Category
const category = results.filter(r => 
    r.metadata.category === 'laptops'
);

// Brand
const brand = results.filter(r => 
    r.metadata.brand === 'Apple'
);

// Rating
const highRated = results.filter(r => 
    r.metadata.rating >= 4.0
);
```

### Pre-filtering vs Post-filtering

**Post-filtering (shown above):**
```javascript
// Search first, filter after
const results = await hybridSearch(query, { k: 100 });
const filtered = results.filter(r => r.metadata.price <= maxPrice);
```

**Pros:** Simple, flexible
**Cons:** Inefficient for large databases

**Pre-filtering (production approach):**
```javascript
// Filter BEFORE searching (if database supports it)
const results = await hybridSearch(query, { 
    k: 100,
    metadataFilter: { 
        price: { $lte: maxPrice },
        category: 'laptops'
    }
});
```

**Pros:** Much faster, searches smaller dataset
**Cons:** Requires database support

---

## Example 7: Performance Optimization

### Optimization 1: Query Result Caching

```javascript
const queryCache = new Map();

async function cachedHybridSearch(query, options = {}) {
    // Create cache key from query + options
    const cacheKey = `${query}:${JSON.stringify(options)}`;

    // Check cache first
    if (queryCache.has(cacheKey)) {
        return { results: queryCache.get(cacheKey), cached: true };
    }

    // Cache miss: perform search
    const queryEmbedding = await embeddingContext.getEmbeddingFor(query);
    const results = await vectorStore.hybridSearch(
        NS,
        Array.from(queryEmbedding.vector),
        query,
        options
    );

    // Store in cache
    queryCache.set(cacheKey, results);
    return { results, cached: false };
}
```

**How it works:**

1. **First search** for "wireless headphones":
   - Computes embedding (slow)
   - Searches database (slow)
   - Caches result
   - Time: ~100ms

2. **Second search** for same query:
   - Finds in cache (instant)
   - Returns cached results
   - Time: ~1ms
   - **100x speedup!**

**Important considerations:**

```javascript
// Set cache expiration (Time-To-Live)
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

queryCache.set(cacheKey, {
    results: results,
    timestamp: Date.now()
});

// Check if cache entry is still valid
if (queryCache.has(cacheKey)) {
    const cached = queryCache.get(cacheKey);
    if (Date.now() - cached.timestamp < CACHE_TTL) {
        return cached.results;
    } else {
        queryCache.delete(cacheKey); // Expired
    }
}
```

### Optimization 2: Two-Stage Retrieval

**Problem:** Computing embeddings for many products is slow.

**Solution:** Search in two stages:

```javascript
// Stage 1: Fast keyword search to get candidates
const candidates = await vectorStore.fullTextSearch(NS, query, 100);
// Returns 100 candidates in ~10ms

// Stage 2: Rerank top candidates with vector search
const candidateIds = candidates.map(c => c.metadata.id);
const reranked = await vectorStore.search(NS, queryEmbedding, 10, {
    filter: { id: { $in: candidateIds } }
});
// Only computes similarity for 100 products, not all 10,000!
```

**Why it's faster:**
- Stage 1: Keyword search is very fast
- Stage 2: Only compute expensive vectors for top candidates
- **10-100x faster** for large databases

### Optimization 3: Index Partitioning

```javascript
// Create separate indices for categories
const laptopStore = new VectorDB({ dim: 384, namespace: "laptops" });
const audioStore = new VectorDB({ dim: 384, namespace: "audio" });
const accessoryStore = new VectorDB({ dim: 384, namespace: "accessories" });

// Search only relevant partition
async function partitionedSearch(query, category) {
    let store;
    switch(category) {
        case 'laptops': store = laptopStore; break;
        case 'audio': store = audioStore; break;
        case 'accessories': store = accessoryStore; break;
    }
    
    return await store.hybridSearch(/* ... */);
}
```

**Benefits:**
- Smaller search space = faster
- Can optimize each partition differently
- Easier to scale horizontally

### Optimization 4: Pre-computed Embeddings

**Bad (slow):**
```javascript
// Compute embedding every time
for (const product of products) {
    const embedding = await getEmbedding(product.description);
    await store.insert(id, embedding, metadata);
}
```

**Good (fast):**
```javascript
// Compute embeddings once, store them
const productsWithEmbeddings = products.map(async (p) => ({
    ...p,
    embedding: await getEmbedding(p.description)
}));

// Save embeddings to database or file
await saveEmbeddings(productsWithEmbeddings);

// Later: Load pre-computed embeddings
const cachedProducts = await loadEmbeddings();
for (const product of cachedProducts) {
    await store.insert(id, product.embedding, metadata);
}
```

**Why:** Computing embeddings is the slowest part. Do it once, reuse many times!

---

## Complete Flow Diagram

Here's how all pieces fit together:

```
User Query: "Apple wireless keyboard"
           ↓
    [Query Analysis]
    Detects: Short Keyword
    Weights: 0.4 vector / 0.6 text
           ↓
    ┌──────────────────┐
    │  Hybrid Search   │
    └──────────────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
[Vector Search] [Keyword Search]
    ↓             ↓
Semantic match  Exact match
"keyboard"      "Apple"
    ↓             ↓
Score: 0.85     Score: 12.5
    ↓             ↓
[Normalize Scores]
    ↓             ↓
Norm: 0.92      Norm: 0.88
    ↓             ↓
    └──────┬──────┘
           ↓
[Combine with Weights]
Combined = (0.92 × 0.4) + (0.88 × 0.6) = 0.896
           ↓
[Apply Business Filters]
- Price <= $200
- In stock = true
- Category = accessories
           ↓
[Return Top Results]
1. Apple Magic Keyboard Touch ID (0.896)
2. Apple Magic Keyboard (0.874)
3. ...
```

---

## Best Practices Summary

### 1. Choose Weights Based on Query Type

| Query Type | Vector Weight | Text Weight |
|------------|---------------|-------------|
| SKU/Product Code | 0.2 | 0.8 |
| Technical Specs | 0.3 | 0.7 |
| Natural Language | 0.8 | 0.2 |
| Short Keywords | 0.4 | 0.6 |
| Mixed/Default | 0.5 | 0.5 |

### 2. Index Multiple Fields

```javascript
await vectorStore.setFullTextIndexedFields(NS, 
    ['content', 'title', 'brand', 'sku', 'attributes']
);
```

### 3. Implement Fallback Strategy

```javascript
// Try balanced → vector-heavy → pure vector
if (results.length < threshold) {
    // Shift toward vector search
}
```

### 4. Cache Popular Queries

```javascript
const cache = new Map();
// Cache results for 5-10 minutes
```

### 5. Use Pre-filtering When Possible

```javascript
// Filter BEFORE search, not after
const results = await search(query, { 
    metadataFilter: { price: { $lte: maxPrice } }
});
```

### 6. Monitor and Optimize

```javascript
// Log slow queries
if (searchTime > 100) {
    console.log(`Slow query: ${query} (${searchTime}ms)`);
}
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Normalize Scores

**Problem:**
```javascript
// Keyword scores: [15.2, 12.8, 9.3]
// Vector scores: [0.85, 0.82, 0.79]
combined = vectorScore + keywordScore; // Keywords dominate!
```

**Solution:**
```javascript
const normVector = normalizeMinMax(vectorScores);
const normKeyword = normalizeMinMax(keywordScores);
combined = (normVector * 0.5) + (normKeyword * 0.5);
```

### Pitfall 2: Not Handling Zero Results

**Problem:**
```javascript
const results = await search(query);
return results[0]; // Error if no results!
```

**Solution:**
```javascript
let results = await search(query, { vectorWeight: 0.5, textWeight: 0.5 });

if (results.length === 0) {
    // Fallback to vector-only
    results = await search(query, { vectorWeight: 1.0, textWeight: 0.0 });
}

if (results.length === 0) {
    return { message: "No products found", suggestions: getPopularProducts() };
}

return results[0];
```

### Pitfall 3: Over-caching

**Problem:**
```javascript
// Cache never expires!
queryCache.set(query, results);
```

**Solution:**
```javascript
// Set TTL and max cache size
const MAX_CACHE_SIZE = 1000;
const CACHE_TTL = 5 * 60 * 1000;

if (queryCache.size >= MAX_CACHE_SIZE) {
    const firstKey = queryCache.keys().next().value;
    queryCache.delete(firstKey); // Remove oldest
}

queryCache.set(query, {
    results,
    timestamp: Date.now()
});
```

### Pitfall 4: Not Using Query Analysis

**Problem:**
```javascript
// Always use same weights for all queries
const results = await hybridSearch(query, { vectorWeight: 0.5, textWeight: 0.5 });
```

**Solution:**
```javascript
// Adapt weights to query type
const analysis = analyzeQuery(query);
const results = await hybridSearch(query, {
    vectorWeight: analysis.weights.vector,
    textWeight: analysis.weights.text
});
```

---

## Conclusion

You now understand:

✅ **What hybrid search is** and why it's powerful  
✅ **How to set up** a vector database with multi-field indexing  
✅ **Score normalization** techniques for fair combination  
✅ **Dynamic weight adjustment** based on query patterns  
✅ **Fallback strategies** for zero-result scenarios  
✅ **Performance optimization** with caching and two-stage retrieval

### Next Steps

1. **Experiment**: Try different weight combinations
2. **Monitor**: Track which queries are slow
3. **Optimize**: Implement caching for popular queries
4. **Iterate**: Adjust weights based on user feedback
5. **Scale**: Add index partitioning for large catalogs

### Additional Resources

- **Vector Search**: Learn about HNSW and FAISS
- **BM25 Algorithm**: Understand the math behind keyword search
- **Query Classification**: Use ML to auto-detect query types
- **Reranking**: Add a third stage for fine-tuned ranking

Happy searching! 🚀