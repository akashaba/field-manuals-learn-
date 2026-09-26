# Multi-Query Retrieval: The Big Picture

This guide explains the *why* and *how* of multi-query retrieval using intuitive concepts and real-world analogies. Perfect for understanding the ideas before diving into the code.

---

## The Core Idea

Imagine you're searching for information in a library. You could:

1. **Single Query Approach**: Walk up to the librarian and say "I need books about web development"
    - The librarian gives you 3-5 books
    - They might all focus on one aspect (like just JavaScript)

2. **Multi-Query Approach**: Ask several specific questions:
    - "What books cover web architecture?"
    - "What about scaling applications?"
    - "And testing practices?"
    - Now you get 10-15 books covering all aspects!

**Multi-query retrieval is asking multiple focused questions instead of one broad question.**

---

## Why Does This Matter?

### The Single-Query Problem

When you search with just one query, you face three main problems:

**1. Tunnel Vision**
```
Query: "How do I cook a healthy meal?"
Result: Might only return recipes, missing nutrition guides or meal planning
```

**2. Vocabulary Gap**
```
Your words: "make apps faster"
Document words: "optimize performance", "improve efficiency"
Result: Mismatch means you miss relevant content
```

**3. Complexity Overload**
```
Query: "Compare React vs Vue for building scalable e-commerce sites with good SEO"
Result: Search gets confused by too many concepts at once
```

### The Multi-Query Solution

**Instead of one complex question, ask several simple ones:**

```
Single (avoid): "Compare React vs Vue for building scalable e-commerce sites with good SEO"

Multiple (prefer):
   1. "What is React?"
   2. "What is Vue?"
   3. "React vs Vue comparison"
   4. "Building scalable web applications"
   5. "E-commerce platform architecture"
   6. "SEO best practices for web apps"
```

Each focused question retrieves different relevant documents. Together, they provide comprehensive coverage.

---

## Core Concepts

### 1. Query Decomposition

**The Idea:** Break complex questions into simpler parts.

**Real-world analogy:**

Imagine planning a trip:
- Bad: "How do I plan a trip to Europe?"
- Good:
    - "What are the best cities to visit in Europe?"
    - "How to book affordable flights?"
    - "What travel documents do I need?"
    - "Where should I stay in each city?"

Each question is easier to answer, and together they cover everything you need.

**In RAG:**
```
Complex: "How do I build a secure, scalable web app with good performance?"

Decomposed:
1. "Web application architecture patterns"
2. "Security best practices for web apps"
3. "Application scalability techniques"
4. "Web performance optimization"
```

**Result:** Instead of 3 vaguely relevant documents, you get 2-3 highly relevant documents per sub-query = 8-12 total relevant documents!

---

### 2. Query Expansion

**The Idea:** Say the same thing in different ways.

**Real-world analogy:**

If you're looking for your car keys, you might ask:
- "Where are my keys?"
- "Have you seen my car keys?"
- "Did I leave my keys here?"
- "Anyone know where my keys are?"

Different people might respond to different phrasings.

**In RAG:**
```
Original: "How to optimize database queries?"

Expanded:
1. "How to optimize database queries?"
2. "Database performance improvement techniques"
3. "Speed up slow database operations"
4. "Query optimization best practices"
```

**Why it works:**
- Different documents use different terminology
- "Optimize" vs "improve" vs "speed up" vs "enhance"
- Each phrasing catches different documents
- Combined = comprehensive results

**Visual representation:**
```
Query 1: "optimize database" → Finds: Doc A, Doc C, Doc E
Query 2: "improve performance" → Finds: Doc B, Doc C, Doc F
Query 3: "speed up queries" → Finds: Doc A, Doc D, Doc G

Combined unique results: Doc A, B, C, D, E, F, G
```

---

### 3. Parallel Execution

**The Idea:** Do multiple things at once instead of one at a time.

**Real-world analogy:**

Imagine you're a chef preparing a meal:

**Sequential (slow):**
```
1. Boil water → wait 10 minutes
2. Chop vegetables → wait 5 minutes
3. Cook meat → wait 15 minutes
Total: 30 minutes
```

**Parallel (fast):**
```
1. Start boiling water AND chop vegetables AND cook meat
   All happening at the same time!
Total: 15 minutes (time of the longest task)
```

**In RAG:**
```javascript
// Sequential: 400ms total
Query 1 → wait 100ms
Query 2 → wait 100ms
Query 3 → wait 100ms
Query 4 → wait 100ms

// Parallel: 100ms total
All queries start together → wait 100ms (for slowest to finish)
```

**Speedup:** 3-4x faster!

---

### 4. Result Fusion

**The Idea:** Intelligently combine multiple result lists into one ranked list.

**Real-world analogy:**

You're choosing a restaurant and ask three friends for recommendations:

**Friend 1's list:**
1. Italian Place (loves it)
2. Sushi Bar (likes it)
3. Burger Joint (okay)

**Friend 2's list:**
1. Sushi Bar (loves it)
2. Thai Restaurant (likes it)
3. Italian Place (okay)

**Friend 3's list:**
1. Italian Place (loves it)
2. Burger Joint (likes it)
3. Mexican Spot (okay)

**How to combine them?**

**Method 1: Score Averaging** (problematic)
- Problem: Different friends use different scoring scales
- Friend 1 is generous (scores 8-10)
- Friend 2 is harsh (scores 4-6)
- Averaging is unfair!

**Method 2: Rank-Based (RRF)** (recommended)
- Italian Place: Appears in 3 lists (1st, 3rd, 1st) → Consistently good!
- Sushi Bar: Appears in 2 lists (2nd, 1st) → Also great
- Burger Joint: Appears in 2 lists (3rd, 2nd) → Okay

**Final ranking:** Italian Place > Sushi Bar > Burger Joint

**In RAG:**

This is exactly what Reciprocal Rank Fusion (RRF) does:
- Looks at rankings, not scores
- Documents appearing in multiple result lists rank higher
- Robust to score scale differences

---

### 5. Reciprocal Rank Fusion (RRF) - Deep Dive

**The Formula:** `Score = 1 / (k + rank + 1)`

**Why this formula works:**

Think of it as a "consistency bonus":

```
Document appears once at rank 1:
Score = 1/(60+0+1) = 0.016

Document appears three times (ranks 0, 1, 2):
Score = 1/61 + 1/62 + 1/63 = 0.049

The more times it appears, the higher the total score!
```

**Visual example:**

```
Query 1 results:          Query 2 results:          Query 3 results:
1. doc_A                  1. doc_B                  1. doc_A
2. doc_B                  2. doc_A                  2. doc_C
3. doc_C                  3. doc_D                  3. doc_B

RRF calculation:
doc_A: 1/61 + 1/62 + 1/61 = 0.049 (appears 3 times) - Best
doc_B: 1/62 + 1/61 + 1/63 = 0.048 (appears 3 times)
doc_C: 1/63 + 1/63 = 0.032 (appears 2 times)
doc_D: 1/63 = 0.016 (appears 1 time)
```

**Key insight:** doc_A consistently ranks highly, so it wins!

---

### 6. Perspective-Based Retrieval

**The Idea:** Look at the same topic from different angles.

**Real-world analogy:**

You're considering buying a new car. You ask:
- **Salesperson:** "Why should I buy this car?" → Hears benefits
- **Mechanic:** "What problems does this car have?" → Hears issues
- **Current owner:** "What's it like day-to-day?" → Hears reality
- **Auto journalist:** "How does it compare to competitors?" → Hears context

Each perspective gives you different valuable information!

**In RAG:**

```
Topic: "microservices architecture"

Different perspectives:
1. Benefits: "What are the advantages of microservices?"
2. Challenges: "What are common microservices problems?"
3. Implementation: "How to implement microservices?"
4. Comparison: "Microservices vs monolithic architecture?"
```

**Result:** Balanced, comprehensive understanding instead of one-sided information.

---

### 7. Weighted Query Fusion

**The Idea:** Not all queries are equally important.

**Real-world analogy:**

You're planning a party and gathering input:
- Your preferences: 100% important (it's YOUR party!)
- Best friend's suggestions: 80% important (they know you well)
- Acquaintance's ideas: 30% important (nice to consider)

You weight each person's input appropriately.

**In RAG:**

```
Main query: "React state management"                 (weight: 1.0)
Related query: "React hooks for state"               (weight: 0.8)
Context query: "JavaScript state patterns"           (weight: 0.5)
Background query: "Frontend architecture basics"     (weight: 0.3)
```

**Scoring:**

```
Document found in:
- Main query (score 0.95)      → contribution: 0.95 × 1.0 = 0.95
- Related query (score 0.88)   → contribution: 0.88 × 0.8 = 0.70
Total weighted score: 1.65
```

This ensures the main query dominates, but related queries add context.

---

### 8. Adaptive Strategy Selection

**The Idea:** Different questions need different approaches.

**Real-world analogy:**

Different problems need different tools:
- Hammering a nail → Hammer
- Tightening a screw → Screwdriver
- Cutting wood → Saw

You wouldn't use a hammer for everything!

**In RAG:**

```
Simple query: "What is React?"
→ Strategy: Single query (fast, simple)

Moderate query: "How to optimize React performance?"
→ Strategy: Query expansion (better coverage)

Complex query: "What's the difference between REST and GraphQL?"
→ Strategy: Perspective-based (balanced comparison)

Very complex: "How do I build scalable apps with microservices, CI/CD, and testing?"
→ Strategy: Multi-part decomposition (comprehensive)
```

**Automatic detection:**

```javascript
if (wordCount < 5) → Simple → Single query
else if (contains "difference" or "vs") → Comparison → Perspective-based
else if (contains "and" + long) → Multi-part → Decomposition
else → Moderate → Query expansion
```

---

## Putting It All Together: A Real Example

Let's walk through a complete multi-query retrieval pipeline:

### User Query
```
"How do I build a scalable web application with good testing practices?"
```

### Step 1: Analyze Query
```
- 11 words (moderate complexity)
- Contains "and" (multiple topics)
- Broad scope
→ Decision: Use query decomposition
```

### Step 2: Decompose
```
Sub-queries:
1. "Web application architecture patterns"
2. "Building scalable applications"
3. "Software testing best practices"
4. "Automated testing implementation"
```

### Step 3: Execute Queries (in parallel)
```
Start all 4 at once:
├─ Query 1 → finds 3 docs about architecture
├─ Query 2 → finds 3 docs about scalability
├─ Query 3 → finds 3 docs about testing
└─ Query 4 → finds 3 docs about automation

Total: 12 documents retrieved
Time: 120ms (vs 400ms sequential)
```

### Step 4: Deduplicate
```
12 documents → some overlap

Deduplication (keep highest score per doc):
12 documents → 8 unique documents
```

### Step 5: Apply RRF Fusion
```
Calculate RRF scores:

doc_010 (microservices): appears in queries 1, 2 → score: 0.032
doc_016 (unit testing): appears in queries 3, 4 → score: 0.032
doc_022 (kubernetes): appears in query 2 → score: 0.016
doc_017 (integration): appears in queries 3, 4 → score: 0.031
...
```

### Step 6: Return Ranked Results
```
1. [0.032] Microservices architecture (covers scalability)
2. [0.032] Unit testing practices (covers testing)
3. [0.031] Integration testing (covers testing)
4. [0.028] Docker containers (covers architecture)
5. [0.025] CI/CD pipelines (covers automation)
```

### Result Quality
```
- Comprehensive coverage (architecture + scalability + testing)
- Diverse perspectives (multiple aspects of each topic)
- High relevance (focused sub-queries)
- Fast execution (parallel retrieval)
```

---

## When to Use Multi-Query Retrieval

### Good Use Cases

**1. Complex Questions**
```
"How do I migrate from monolithic to microservices while maintaining uptime?"
→ Multiple concepts, needs decomposition
```

**2. Comparison Questions**
```
"Should I use SQL or NoSQL for my project?"
→ Needs balanced perspectives
```

**3. Learning/Research**
```
"Tell me everything about Kubernetes"
→ Needs comprehensive coverage
```

**4. Decision-Making**
```
"What are the pros and cons of serverless architecture?"
→ Needs multiple viewpoints
```

### Poor Use Cases

**1. Simple Factual Questions**
```
"What is the capital of France?"
→ Single query is fine, decomposition wastes time
```

**2. Very Specific Queries**
```
"What's the error code for HTTP unauthorized?"
→ Direct match needed, expansion doesn't help
```

**3. Time-Critical Applications**
```
Real-time chatbots needing <100ms response
→ Single query is faster
```

**4. Limited Computation**
```
Running on mobile device with limited resources
→ Multiple queries drain battery/bandwidth
```

---

## Trade-offs to Consider

### Quality vs Speed

```
Single Query:
- Speed: 5/5 (50ms)
- Quality: 3/5 (may miss aspects)
- Complexity: 1/5 (simple)

Multi-Query:
- Speed: 3/5 (150-300ms)
- Quality: 5/5 (comprehensive)
- Complexity: 4/5 (more complex)
```

**Recommendation:** Use adaptive strategy to balance automatically!

### Breadth vs Precision

```
Single Query:
- Precision: 4/5 (focused)
- Recall: 3/5 (may miss documents)
- Diversity: 2/5 (limited perspectives)

Multi-Query:
- Precision: 3/5 (some irrelevant)
- Recall: 5/5 (catches more)
- Diversity: 5/5 (multiple perspectives)
```

**Recommendation:** Use for complex queries where recall matters!

---

## Key Takeaways

### The Big Ideas

1. **One question isn't always enough** - Complex queries need decomposition
2. **Different words, same meaning** - Query expansion handles vocabulary gaps
3. **Do things simultaneously** - Parallel execution speeds things up
4. **Rankings matter more than scores** - RRF combines results robustly
5. **Different questions need different strategies** - Adapt approach to complexity
6. **Multiple perspectives = complete picture** - See the whole topic, not just one angle

### The Mental Model

Think of multi-query retrieval like:

**Investigating a story as a journalist:**
- Don't just ask one person one question
- Ask multiple people (perspectives)
- Ask the same thing different ways (expansion)
- Ask follow-up questions (decomposition)
- Talk to everyone at once (parallel)
- Weigh sources by credibility (weighted fusion)
- Look for consistent information (RRF)

**Result:** Comprehensive, accurate, balanced story!

### When to Use It

```
Use multi-query when:
- Query is complex (multiple concepts)
- Need comprehensive coverage
- Comparison or decision-making
- Learning/research scenarios
- Single query gives poor results

Skip multi-query when:
- Query is simple (factual lookup)
- Need ultra-fast response (<100ms)
- Limited computational resources
- Single query works well enough
```

---

## Practical Implementation Tips

### Start Simple
```
1. Begin with single query
2. If results are poor, add query expansion (2-3 variations)
3. If still poor, try decomposition
4. Use RRF to combine results
5. Measure and iterate!
```

### Optimize Incrementally
```
Phase 1: Get it working (sequential execution)
Phase 2: Make it fast (parallel execution)
Phase 3: Make it smart (adaptive strategy)
Phase 4: Make it efficient (caching, limits)
```

### Monitor and Adapt
```
Track metrics:
├─ Average query time
├─ Number of results returned
├─ User satisfaction (click-through rate)
└─ Most common query patterns

Adjust based on data:
├─ If too slow → reduce sub-queries
├─ If poor results → try different decomposition
└─ If working well → keep it!
```

---

## Common Questions

**Q: Isn't this just overcomplicating search?**
A: For simple queries, yes! Use single query. For complex queries, multi-query retrieval dramatically improves results. Use adaptively.

**Q: How many sub-queries should I use?**
A: Sweet spot is 3-5 sub-queries. More than 7 usually doesn't help and slows things down.

**Q: What if my sub-queries overlap?**
A: That's okay! Deduplication handles it. Overlapping queries actually help—they reinforce important documents.

**Q: Is parallel execution always faster?**
A: Almost always, yes. The only exception is if you have very limited resources or your backend doesn't support concurrent requests.

**Q: Should I always use RRF for fusion?**
A: RRF is a great default. Use weighted fusion when you want to prioritize certain queries over others.

**Q: How do I know if it's working?**
A: Compare single-query vs multi-query results:
- Are multi-query results more comprehensive?
- Do users find what they need faster?
- Is the latency acceptable?

If yes to all three, it's working!

---

## Next Steps

Now that you understand the concepts:

1. **Read the CODE.md** - See how to implement these ideas
2. **Try the examples** - Run the code and see it in action
3. **Experiment** - Try different decomposition strategies
4. **Measure** - Track what works for your use case
5. **Iterate** - Refine based on results

Remember: Start simple, measure results, and add complexity only when needed!

---

**Happy querying!**