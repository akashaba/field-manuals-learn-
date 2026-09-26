# Hybrid Search

## What is Hybrid Search?

### The Simple Answer

Hybrid search is like having **two search engines working together**:

1. **One that understands meaning** (semantic search)
2. **One that finds exact words** (keyword search)

By combining both, you get better results than using either one alone.

### A Real-World Analogy

Imagine you're looking for a restaurant:

**Friend #1** (Semantic searcher):
- Understands: "I want Italian food" = pizza, pasta, risotto
- Good at: Finding things by concept and meaning
- Weakness: Might miss if you say "I want Alfredo's Pizza" (specific name)

**Friend #2** (Keyword searcher):
- Searches for: Exact words like "Alfredo's Pizza"
- Good at: Finding specific names and terms
- Weakness: If you say "Italian food," won't connect it to pizza places

**Both friends together** (Hybrid search):
- Friend #1 finds Italian restaurants (concept)
- Friend #2 finds "Alfredo's Pizza" (exact name)
- You get the best of both! 

---

## The Real Problem Nobody Talks About

Most tutorials tell you "combine vector search with keyword search" and show you a 0.5/0.5 weight split. That's it. But when you build this with real product data, you immediately hit these problems:

### Problem #1: SKU Codes Break Vector Search

**The scenario:**
```
Query: "MBP-M3MAX-32-1TB"
Vector search returns: Generic laptop descriptions ✗
Keyword search returns: Exact product match ✓

Why? Vector embeddings don't understand alphanumeric codes!
```

**Real example:**
```
Product in your catalog:
  Title: "Apple MacBook Pro 16-inch M3 Max"
  SKU: "MBP-M3MAX-32-1TB"
  Description: "Powerful laptop with advanced M3 Max chip..."

Vector Search Result (similarity: 0.3):
  → "Dell Inspiron laptop with Intel processor..." ✗ WRONG
  
Keyword Search Result (BM25 score: 45.2):
  → "Apple MacBook Pro 16-inch M3 Max" ✓ CORRECT
```

This is why a fixed 0.5/0.5 split fails catastrophically for product searches.

### Problem #2: Score Ranges Don't Match

**The mismatch:**
```
Vector similarity scores: 0.3 to 0.4 (small range)
BM25 keyword scores: 15 to 50 (large range)

If you just add them together, BM25 dominates everything!
```

**Concrete example:**
```
Document A:
  Vector: 0.35
  BM25: 18.5
  Naive sum: 18.85 ← BM25 score dominates

Document B:
  Vector: 0.42  ← Actually a better semantic match!
  BM25: 15.0
  Naive sum: 15.42 ← Ranks LOWER despite better meaning

Result: Your vector search becomes completely useless!
```

**You MUST normalize scores before combining. This is not optional.**

### Problem #3: Different Queries Need Different Weights

**Why 0.5/0.5 fails:**

```
Query 1: "MBP-M3MAX-32-1TB"
What it needs: Keyword-heavy (0.2 vector / 0.8 keyword)
Why: It's a product code, semantic search is useless here

Query 2: "What's the best laptop for video editing?"
What it needs: Vector-heavy (0.8 vector / 0.2 keyword)
Why: Natural language question needs understanding

Query 3: "Sony headphones"
What it needs: Balanced (0.5 / 0.5)
Why: Brand name + product category, both matter

One weight setting cannot possibly work for all three!
```

---

## How to Build Hybrid Search That Actually Works

### Step 1: Score Normalization (CRITICAL)

You have three options. Here's when to use each:

#### Option 1: Min-Max Normalization

**Formula:**
```
normalized_score = (score - min_score) / (max_score - min_score)
```

**Example with BM25 scores:**
```
Raw scores: [15, 25, 50]
Min: 15, Max: 50, Range: 35

Normalized:
  15 → (15-15)/35 = 0.0
  25 → (25-15)/35 = 0.286
  50 → (50-15)/35 = 1.0
```

**Pros:** Simple, always gives 0-1 range  
**Cons:** Sensitive to outliers (one extreme score skews everything)  
**Use when:** Quick prototyping, well-behaved data

#### Option 2: Z-Score Normalization

**Formula:**
```
normalized_score = (score - mean) / standard_deviation
```

**Example:**
```
Raw scores: [15, 25, 50]
Mean: 30, Std Dev: 14.79

Normalized:
  15 → (15-30)/14.79 = -1.01
  25 → (25-30)/14.79 = -0.34
  50 → (50-30)/14.79 = 1.35
```

**Pros:** Preserves distribution shape  
**Cons:** Can give negative values, needs additional scaling  
**Use when:** Statistical analysis matters

#### Option 3: Rank-Based Normalization (RECOMMENDED)

**Formula:**
```
normalized_score = rank_position / total_results
```

**Example:**
```
Sorted results:
  1st place (score 50) → 1/3 = 0.33
  2nd place (score 25) → 2/3 = 0.67
  3rd place (score 15) → 3/3 = 1.0
```

**Pros:** Most robust, handles any score scale, used by RRF  
**Cons:** Loses exact score magnitude  
**Use when:** Production systems (this is what actually works)

**Decision guide:**
```
Score scales vastly different? → Rank-Based (safest choice)
Need exact precision? → Min-Max
Statistical analysis? → Z-Score
Not sure? → Rank-Based (production default)
```

---

### Step 2: Dynamic Weight Selection

**Stop hardcoding weights. Auto-detect query patterns instead.**

You have two approaches: **rule-based** (fast, predictable) or **LLM-based** (smart, flexible).

#### Approach A: Rule-Based Pattern Detection (Fast)

**Best for:** Production systems, low latency requirements, predictable patterns

```javascript
function analyzeQuery(query) {
    const upperCount = (query.match(/[A-Z]/g) || []).length;
    const digitCount = (query.match(/\d/g) || []).length;
    const hyphenCount = (query.match(/-/g) || []).length;
    
    // SKU/Product code pattern
    // Examples: "MBP-M3MAX-32-1TB", "PROD-12345-XL"
    if (hyphenCount >= 2 || (upperCount > 3 && digitCount > 0)) {
        return { vector: 0.2, text: 0.8 }; // keyword-heavy
    }
    
    // Natural language question
    // Examples: "What laptop is best?", "How do I...?"
    if (/^(what|how|which|why|when|where)/i.test(query)) {
        return { vector: 0.8, text: 0.2 }; // vector-heavy
    }
    
    // Brand + product (2 words, one capitalized)
    // Examples: "Sony headphones", "Apple laptop"
    const words = query.split(' ');
    if (words.length === 2 && upperCount > 0) {
        return { vector: 0.5, text: 0.5 }; // balanced
    }
    
    // Default to balanced
    return { vector: 0.5, text: 0.5 };
}
```

**Pros:**
- Fast (< 1ms)
- Predictable
- No API costs
- Easy to debug

**Cons:**
- Limited to patterns you define
- Can't handle nuanced queries
- Requires manual tuning

#### Approach B: LLM-Based Query Classification (Smart)

**Best for:** Complex queries, when you need reasoning, high-quality search experiences

```javascript
async function analyzeQueryWithLLM(query) {
    const prompt = `Analyze this search query and classify it to determine optimal search weights.

Query: "${query}"

Classify into ONE of these categories:

1. PRODUCT_CODE: Contains SKU, product code, or alphanumeric identifier
   Examples: "MBP-M3MAX-32-1TB", "SKU-12345", "PROD-001-XL"
   Weights: vector=0.2, keyword=0.8

2. NATURAL_QUESTION: Conversational question seeking recommendations
   Examples: "What's the best laptop?", "How do I choose headphones?"
   Weights: vector=0.8, keyword=0.2

3. BRAND_CATEGORY: Brand name + product category
   Examples: "Sony headphones", "Apple laptop", "Dell monitor"
   Weights: vector=0.5, keyword=0.5

4. TECHNICAL_TERM: Technical jargon, acronyms, specific terminology
   Examples: "PostgreSQL ACID", "CNN architecture", "GraphQL schema"
   Weights: vector=0.3, keyword=0.7

5. FEATURE_SEARCH: Describes features or attributes
   Examples: "wireless noise canceling", "16GB RAM laptop", "4K monitor"
   Weights: vector=0.6, keyword=0.4

Respond ONLY with JSON:
{
  "category": "CATEGORY_NAME",
  "reasoning": "brief explanation",
  "weights": { "vector": 0.X, "keyword": 0.X }
}`;

    const response = await llm.complete(prompt);
    const result = JSON.parse(response);
    
    return {
        vector: result.weights.vector,
        text: result.weights.keyword,
        reasoning: result.reasoning
    };
}
```

**Example LLM responses:**

```javascript
Query: "MBP-M3MAX-32-1TB"
Response: {
    "category": "PRODUCT_CODE",
    "reasoning": "Contains alphanumeric code with hyphens typical of SKU",
    "weights": { "vector": 0.2, "keyword": 0.8 }
  }

Query: "laptop for machine learning under $2000"
Response: {
    "category": "FEATURE_SEARCH",
    "reasoning": "Describes use case (ML) and constraint (price)",
    "weights": { "vector": 0.6, "keyword": 0.4 }
  }

Query: "how do Sony headphones compare to Bose"
Response: {
    "category": "NATURAL_QUESTION",
    "reasoning": "Comparative question needing conceptual understanding",
    "weights": { "vector": 0.8, "keyword": 0.2 }
  }
```

**Pros:**
- Handles complex/nuanced queries
- Can reason about edge cases
- Adapts to new patterns without code changes
- Can explain its decisions

**Cons:**
- Slower (100-500ms)
- Costs per query
- Needs fallback if LLM fails
- Less predictable

#### Approach C: Hybrid (Best of Both)

Use rule-based with LLM fallback for complex queries:

```javascript
async function analyzeQueryHybrid(query) {
    // Fast path: Simple patterns
    const ruleBasedResult = analyzeQueryRuleBased(query);
    
    // If confident, return immediately
    if (ruleBasedResult.confidence > 0.8) {
        return ruleBasedResult;
    }
    
    // Slow path: LLM for complex queries
    try {
        const llmResult = await analyzeQueryWithLLM(query);
        return llmResult;
    } catch (error) {
        // Fallback to rule-based
        console.log("LLM failed, using rules");
        return ruleBasedResult;
    }
}

function analyzeQueryRuleBased(query) {
    const upperCount = (query.match(/[A-Z]/g) || []).length;
    const digitCount = (query.match(/\d/g) || []).length;
    const hyphenCount = (query.match(/-/g) || []).length;
    
    // High confidence patterns
    if (hyphenCount >= 2 || (upperCount > 3 && digitCount > 0)) {
        return { 
            vector: 0.2, 
            text: 0.8, 
            confidence: 0.95,
            method: 'rule' 
        };
    }
    
    if (/^(what|how|which|why)/i.test(query)) {
        return { 
            vector: 0.8, 
            text: 0.2, 
            confidence: 0.85,
            method: 'rule'
        };
    }
    
    // Low confidence - let LLM decide
    return { 
        vector: 0.5, 
        text: 0.5, 
        confidence: 0.5,
        method: 'rule'
    };
}
```

#### When to Use Which Approach?

```
┌───────────────────────────────────────────────────────────┐
│              QUERY CLASSIFICATION DECISION                │
├───────────────────────────────────────────────────────────┤
│                                                           │
│ Simple, predictable queries (SKUs, brands)?               │
│   → Rule-Based (Approach A)                               │
│                                                           │
│ Complex, nuanced queries (comparisons, multi-intent)?     │
│   → LLM-Based (Approach B)                                │
│                                                           │
│ Mixed traffic, want best of both?                         │
│   → Hybrid (Approach C)                                   │
│                                                           │
│ Ultra-low latency required (< 10ms)?                      │
│   → Rule-Based only                                       │
│                                                           │
│ Cost is primary concern?                                  │
│   → Rule-Based (free)                                     │
│                                                           │
│ Quality is primary concern?                               │
│   → LLM-Based (best results)                              │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

#### Real-World Performance Comparison

Test queries and results:

```
Query: "MBP-M3MAX-32-1TB"
├─ Rule-Based: ✓ 0.2/0.8 (correct) - 0.5ms
└─ LLM-Based:  ✓ 0.2/0.8 (correct) - 180ms

Query: "Sony headphones"  
├─ Rule-Based: ✓ 0.5/0.5 (correct) - 0.3ms
└─ LLM-Based:  ✓ 0.5/0.5 (correct) - 150ms

Query: "laptop for machine learning under $2000 with good battery"
├─ Rule-Based: ~ 0.5/0.5 (generic fallback) - 0.4ms
└─ LLM-Based:  ✓ 0.6/0.4 (optimal feature search) - 220ms

Query: "compare Sony WH-1000XM5 to Bose QC45 for travel"
├─ Rule-Based: ~ 0.5/0.5 (generic fallback) - 0.3ms  
└─ LLM-Based:  ✓ 0.7/0.3 (comparison question) - 200ms
```

**Key insight:** Rule-based handles ~70% of queries perfectly. LLM improves the remaining 30% of complex queries.

#### Production Recommendation

Start with rule-based, add LLM later:

```javascript
// Phase 1: Launch with rules (week 1)
const weights = analyzeQueryRuleBased(query);

// Phase 2: Add LLM for complex queries (week 4)
const weights = await analyzeQueryHybrid(query);

// Phase 3: Optimize with caching (week 8)
const weights = await analyzeQueryHybridCached(query);
```

**Why this order?**
1. Rules get you 80% of the way with zero cost
2. Monitor which queries fail
3. Add LLM only for those patterns
4. Cache LLM results for repeated queries

#### Query Pattern Examples

**Pattern 1: Product Codes → Keyword-Heavy (0.2/0.8)**
```
"MBP-M3MAX-32-1TB"
"SKU-99887-V2"
"PROD-001-XL-BLK"

Detection: Multiple hyphens + uppercase + digits
Why keyword-heavy: Vector embeddings fail on codes
```

**Pattern 2: Questions → Vector-Heavy (0.8/0.2)**
```
"What's the best laptop for video editing?"
"How do wireless headphones compare?"
"Which monitor is good for gaming?"

Detection: Starts with question words
Why vector-heavy: Need conceptual understanding
```

**Pattern 3: Brand + Category → Balanced (0.5/0.5)**
```
"Sony headphones"
"Apple laptop"
"Dell monitor"

Detection: Two words, one capitalized
Why balanced: Both brand name and concept matter
```

---

### Step 3: Multi-Field Indexing

For product search, single-field indexing fails. You MUST index multiple fields.

#### The Problem

```
Product data:
  Title: "MacBook Pro 16-inch M3 Max"
  Brand: "Apple"
  SKU: "MBP-M3MAX-32-1TB"
  Description: "Powerful laptop with advanced chip..."
  Attributes: "M3 Max, 32GB RAM, 1TB SSD"

Query: "Apple wireless keyboard"

With single-field (description only):
  ✗ Misses "Apple" in brand field
  ✗ Misses exact title match
  ✗ Only searches description text
  Result: Poor results
```

#### The Solution

```javascript
// Index ALL relevant fields
await vectorStore.setFullTextIndexedFields(namespace, [
    'content',    // Full product description (semantic)
    'title',      // Product name (keyword + semantic)
    'brand',      // Apple, Dell, Sony (exact match)
    'sku',        // Product codes (exact match)
    'attributes'  // Technical specs (mixed)
]);
```

#### How It Works

```
Query: "Apple wireless keyboard"

Brand field matches:
  "Apple" → Exact brand match (high keyword score)

Title field matches:
  "Magic Keyboard" → Contains "keyboard" (medium score)

Description matches:
  "Wireless connectivity..." → Semantic match (medium score)

Combined result: Product surfaces because multiple fields contribute!
```

#### Product Catalog Structure

```javascript
// Structure your products like this:
new Document(
    "Apple MacBook Pro 16-inch with M3 Max chip, featuring...", 
    {
        id: "PROD-001",
        title: "MacBook Pro 16-inch M3 Max",
        brand: "Apple",
        category: "laptops",
        price: 3499,
        sku: "MBP-M3MAX-32-1TB",
        attributes: "M3 Max, 32GB RAM, 1TB SSD, 16-inch Liquid Retina XDR"
    }
)
```

**Why each field matters:**
- `brand`: Exact brand searches ("Apple")
- `sku`: Product code searches ("MBP-M3MAX-32-1TB")
- `title`: Product name searches ("MacBook Pro")
- `attributes`: Feature searches ("32GB RAM")
- `content`: Natural language searches ("best laptop for coding")

---

### Step 4: Fallback Strategies

Never show empty results when similar products exist.

#### The Problem

```
User searches: "Microsoft laptop"
Your catalog: Only Apple, Dell, HP laptops

Keyword search: 0 results (no "Microsoft" products)
Vector search: Finds similar laptops

Without fallback: User sees "No results found" (terrible UX!)
```

#### The Solution

```javascript
async function searchWithFallback(query) {
    // Attempt 1: Balanced search
    let results = await hybridSearch(query, { 
        vector: 0.5, 
        text: 0.5,
        limit: 10
    });
    
    // Attempt 2: If few results, go vector-heavy
    if (results.length < 3) {
        console.log("Few results, expanding search...");
        results = await hybridSearch(query, { 
            vector: 0.8, 
            text: 0.2,
            limit: 10
        });
    }
    
    // Attempt 3: Pure semantic if still nothing
    if (results.length === 0) {
        console.log("No exact matches, finding similar...");
        results = await hybridSearch(query, { 
            vector: 1.0, 
            text: 0.0,
            limit: 10
        });
    }
    
    return results;
}
```

#### User Experience

```
Query: "Microsoft laptop"

Step 1 (Balanced 0.5/0.5):
  → 0 results (no Microsoft in catalog)

Step 2 (Vector-heavy 0.8/0.2):
  → 3 results found!
     - Dell Inspiron 15 (similar specs)
     - HP Pavilion (similar price)
     - Apple MacBook (similar category)

Display message:
  "No Microsoft laptops found. Here are similar options:"
```

---

## Two Combination Methods: Weighted vs RRF

### Method 1: Weighted Combination (Score-Based)

**How it works:** Uses actual scores with adjustable weights

**Formula:**
```
final_score = (vector_score × vector_weight) + (keyword_score × keyword_weight)
```

**Example:**
```
Document A:
  Vector score: 0.8 (after normalization)
  Keyword score: 0.3 (after normalization)
  Weights: 0.7 vector / 0.3 keyword
  
Final: (0.8 × 0.7) + (0.3 × 0.3) = 0.56 + 0.09 = 0.65
```

**Pros:**
- Simple and intuitive
- Easy to tune weights
- Full control over semantic/keyword balance

**Cons:**
- Requires good normalization
- Sensitive to score distributions

**When to use:** Default choice, especially when you want control

### Method 2: Reciprocal Rank Fusion (RRF)

**How it works:** Uses position/rank instead of scores

**Formula:**
```
RRF_score = 1 / (k + rank)
where k is a constant (typically 60)
```

**Example:**
```
Semantic results:          Keyword results:
1. Doc A (rank 1)         1. Doc C (rank 1)
2. Doc B (rank 2)         2. Doc A (rank 2)
3. Doc C (rank 3)         3. Doc D (rank 3)

RRF Scores (k=60):
Doc A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 ← Highest!
Doc C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
Doc B: 1/(60+2) + 0        = 0.0161
Doc D: 0        + 1/(60+3) = 0.0159
```

**Pros:**
- No normalization needed
- Very stable across queries
- Less sensitive to score scale issues

**Cons:**
- Can't easily adjust semantic/keyword balance
- Slightly less intuitive

**When to use:** Score scales vary wildly, want "set and forget"

### Which Should You Use?

```
Use Weighted when:
  ✓ You want fine control over weights
  ✓ Scores are well-normalized
  ✓ You understand your query patterns

Use RRF when:
  ✓ Score ranges are very different
  ✓ You want maximum stability
  ✓ Normalization is problematic
```

---

## Real-World Examples

### Example 1: E-commerce SKU Search

**Query:** `"MBP-M3MAX-32-1TB"`

**Auto-detected weights:** Keyword-heavy (0.2/0.8)

**What happens:**
```
Vector search (20% weight):
  Similarity: 0.4 → Finds MacBook Pro category
  Normalized: 0.4
  
Keyword search (80% weight):
  BM25: 45.2 → Exact SKU match!
  Normalized: 0.95

Combined: (0.4 × 0.2) + (0.95 × 0.8) = 0.08 + 0.76 = 0.84 ✓
```

**Without dynamic weights (0.5/0.5):**
```
Combined: (0.4 × 0.5) + (0.95 × 0.5) = 0.20 + 0.475 = 0.675
Generic products might rank higher! ✗
```

### Example 2: Natural Language Question

**Query:** `"What's the best laptop for video editing?"`

**Auto-detected weights:** Vector-heavy (0.8/0.2)

**What happens:**
```
Vector search (80% weight):
  Understands: video editing = high performance, graphics, RAM
  Finds: MacBook Pro M3 Max, Dell XPS 15, etc.
  Normalized: 0.88

Keyword search (20% weight):
  Matches: "laptop", "video", "editing"
  Normalized: 0.6

Combined: (0.88 × 0.8) + (0.6 × 0.2) = 0.704 + 0.12 = 0.824 ✓
```

### Example 3: Brand + Category

**Query:** `"Sony headphones"`

**Auto-detected weights:** Balanced (0.5/0.5)

**What happens:**
```
Vector search (50% weight):
  Finds: Audio products, similar Sony items
  Normalized: 0.75

Keyword search (50% weight):
  Exact matches: "Sony" in brand + "headphones" in title
  Normalized: 0.90

Combined: (0.75 × 0.5) + (0.90 × 0.5) = 0.375 + 0.45 = 0.825 ✓
```

### Example 4: Fallback in Action

**Query:** `"Microsoft laptop"`  
**Problem:** No Microsoft products in catalog

**Fallback sequence:**
```
Attempt 1 (Balanced 0.5/0.5):
  Keyword: 0 matches (no "Microsoft")
  Vector: 0.3 similarity (finds laptops generally)
  Result: 0 strong matches

Attempt 2 (Vector-heavy 0.8/0.2):
  Vector: 0.8 similarity (finds similar laptops)
  Results: Dell Inspiron, HP Pavilion, etc.
  Display: "No Microsoft laptops. Similar options:"
```

---

## Combining with Filters

Hybrid search + metadata filters = production-ready search

### Implementation

```javascript
async function searchProducts(query, filters = {}) {
    // Step 1: Auto-detect query type
    const weights = analyzeQuery(query);
    
    // Step 2: Perform hybrid search
    let results = await hybridSearch(query, weights);
    
    // Step 3: Apply filters
    if (filters.category) {
        results = results.filter(r => 
            r.metadata.category === filters.category
        );
    }
    
    if (filters.priceRange) {
        const [min, max] = filters.priceRange;
        results = results.filter(r => 
            r.metadata.price >= min && r.metadata.price <= max
        );
    }
    
    if (filters.brand) {
        results = results.filter(r => 
            r.metadata.brand === filters.brand
        );
    }
    
    return results;
}
```

### Example Usage

```javascript
// Natural language + price filter
const results = await searchProducts(
    "best laptop for gaming",
    {
        category: "laptops",
        priceRange: [1000, 2000]
    }
);

// Hybrid search: Finds relevant gaming laptops
// Filters: Only those priced $1000-$2000
```

---

## Understanding BM25

BM25 is the keyword search algorithm. Here's what actually matters:

### The Three Questions BM25 Asks

**1. Does the document contain the search terms?**
```
Query: "machine learning"
Doc A: Contains both terms ✓
Doc B: Contains "machine" only
Doc C: Contains neither ✗
```

**2. How rare are the terms?**
```
"the" appears in 90% of docs → Low importance
"GraphQL" appears in 5% of docs → High importance

Rare terms get higher scores!
```

**3. How long is the document?**
```
Doc A: 100 words, "machine learning" appears 5 times (5% density)
Doc B: 500 words, "machine learning" appears 5 times (1% density)

Doc A scores higher (better density)!
```

### The Two BM25 Parameters

**k1 (Term Frequency Saturation):**
```
Low k1 (1.2): "machine machine machine" ≈ "machine"
High k1 (2.0): More repetition = much higher score
Default: 1.5 (balanced)
```

**b (Length Normalization):**
```
b = 0.0: No penalty for long documents
b = 1.0: Strong penalty for long documents
Default: 0.75 (mostly penalize length)
```

**Pro tip:** The defaults work well 99% of the time. Only tune if results seem off.

---

## Common Patterns and Tips

### Pattern 1: Start Simple, Then Optimize

```
Week 1: Use balanced (0.5/0.5) for everything
Week 2: Track which queries work/fail
Week 3: Identify patterns (SKUs, questions, brands)
Week 4: Implement auto-detection
```

### Pattern 2: Auto-Detection Rules

```
SKU codes (hyphens + caps + digits) → 0.2/0.8
Questions (what/how/why) → 0.8/0.2
Brand names (2 words, capitalized) → 0.5/0.5
Technical terms (multiple caps) → 0.3/0.7
Everything else → 0.5/0.5
```

### Pattern 3: Let Users Tune

```
Search bar with mode selector:
○ Exact match (keyword-heavy)
○ Similar results (semantic-heavy)
○ Balanced (default)
```

### Pattern 4: Monitor and Iterate

```
Log every search:
  - Query text
  - Detected weights
  - Result count
  - Click-through rate

Analyze weekly:
  - Which patterns work?
  - Which fail?
  - Adjust detection logic
```

---

## When Hybrid Search Really Shines

### Scenario 1: E-commerce

**Why it's perfect:**
- SKU codes (keyword search)
- Product descriptions (semantic search)
- Brand names (exact matches)
- Feature searches (conceptual)

**Example:**
```
"wireless headphones under $100"
→ Multi-field search across title, attributes, price
→ Semantic understanding of "wireless"
→ Keyword match on "$100"
```

### Scenario 2: Help Centers

**Why it's perfect:**
- Users ask questions many ways
- Need conceptual understanding
- But also need exact error codes

**Example:**
```
"how do I reset my password"
→ Semantic: Understands intent
→ Keyword: Ensures "password" appears
→ Finds all password recovery articles
```

### Scenario 3: Technical Documentation

**Why it's perfect:**
- Specific terms matter (PostgreSQL, ACID)
- But concepts matter too (transactions)

**Example:**
```
"PostgreSQL ACID transactions"
→ Keyword: Exact "PostgreSQL" and "ACID"
→ Semantic: Related transaction concepts
→ Perfect technical matches
```

### Scenario 4: Code Search

**Why it's perfect:**
- Function names (exact keywords)
- Descriptions (semantic)
- Comments (conceptual)

**Example:**
```
"function to parse JSON"
→ Keyword: "parse", "JSON"
→ Semantic: Similar to "decode", "deserialize"
→ Finds all relevant code
```

---

### The Golden Rules

1. **Always normalize scores** (rank-based is safest)
2. **Auto-detect query patterns** (don't hardcode weights)
3. **Index multiple fields** (especially for products)
4. **Implement fallbacks** (never show empty results)
5. **Start balanced, then optimize** (measure everything)

---

## Quick Reference

![Hybrid Search Decision Tree](../../../images/hybrid-search-decision-tree.png)

### Normalization Methods:
```
┌─────────────────┬──────────────┬─────────────┬──────────────────┐
│ Method          │ Best For     │ Pros        │ Cons             │
├─────────────────┼──────────────┼─────────────┼──────────────────┤
│ Rank-Based      │ Production   │ Most robust │ Loses magnitude  │
│ (RECOMMENDED)   │ systems      │ No outliers │                  │
├─────────────────┼──────────────┼─────────────┼──────────────────┤
│ Min-Max         │ Prototyping  │ Simple      │ Outlier          │
│                 │              │ Fast        │ sensitive        │
├─────────────────┼──────────────┼─────────────┼──────────────────┤
│ Z-Score         │ Statistical  │ Preserves   │ Can be negative  │
│                 │ analysis     │ distribution│                  │
└─────────────────┴──────────────┴─────────────┴──────────────────┘
```

### Combination Methods:
```
┌─────────────┬───────────────┬──────────────────┬────────────────┐
│ Method      │ Control Level │ Requires         │ Best For       │
├─────────────┼───────────────┼──────────────────┼────────────────┤
│ Weighted    │ Full control  │ Normalization    │ Fine-tuning    │
│             │ over balance  │                  │                │
├─────────────┼───────────────┼──────────────────┼────────────────┤
│ RRF         │ Less tunable  │ Nothing          │ Stability,     │
│             │               │                  │ set-and-forget │
└─────────────┴───────────────┴──────────────────┴────────────────┘
```