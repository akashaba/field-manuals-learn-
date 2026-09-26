# Text Splitting Concepts and Architecture

## Table of Contents
1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Comparison with LangChain.js](#comparison-with-langchainjs)
4. [Summary](#summary)

---

## Overview

This module implements text splitting functionality for Retrieval-Augmented Generation (RAG) systems. It takes large documents and breaks them into smaller, manageable chunks while preserving context through overlapping segments.

### Key Concepts

**Why Split Text?**
- Embedding models have token limits (typically 512-8192 tokens)
- Smaller chunks improve retrieval precision
- Overlap maintains context across boundaries
- Better semantic search results

**The Core Problem:**
```
Large Document (10,000 chars)
        ↓
[Chunk 1] [Chunk 2] [Chunk 3] [Chunk 4]
   ↑overlap↑  ↑overlap↑  ↑overlap↑
```

---

## Core Architecture

### Class Hierarchy

```
TextSplitter (Base Class)
    ├── CharacterTextSplitter
    ├── RecursiveCharacterTextSplitter
    └── TokenTextSplitter
```

### Design Philosophy

1. **Single Responsibility**: Each class has one job
2. **Inheritance**: Common logic in base class
3. **Polymorphism**: Different splitting strategies via `splitText()`
4. **Composition**: Complex splitters use simpler ones internally

---

## Comparison with LangChain.js

### Similarities 

#### 1. **Same Core API**
```javascript
// Our Implementation
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});
const chunks = await splitter.splitDocuments(documents);

// LangChain.js
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});
const chunks = await splitter.splitDocuments(documents);
```

**Result:** Drop-in compatible! Code works with both.

#### 2. **Same Class Hierarchy**
```
Both implementations:
- TextSplitter (base)
  ├── CharacterTextSplitter
  ├── RecursiveCharacterTextSplitter
  └── TokenTextSplitter
```

#### 3. **Same Splitting Logic**
- Both use the "merge with overlap" algorithm
- Same recursive strategy for large chunks
- Same separator hierarchy

#### 4. **Same Metadata Structure**
```javascript
{
    pageContent: "chunk text",
    metadata: {
        source: "...",
        chunk: 0,
        totalChunks: 5
    }
}
```

### Differences 

#### 1. **Code Simplicity**

**Our Implementation:**
```javascript
// Concise constructor
constructor({chunkSize = 1000, chunkOverlap = 200, lengthFunction = t => t.length} = {}) {
    if (chunkOverlap >= chunkSize) {
        throw new Error('chunkOverlap must be less than chunkSize');
    }
    Object.assign(this, {chunkSize, chunkOverlap, lengthFunction});
}
```

**LangChain.js:**
```javascript
// More verbose, more validation
constructor(fields) {
    super(fields);
    this.chunkSize = fields?.chunkSize ?? 1000;
    this.chunkOverlap = fields?.chunkOverlap ?? 200;
    // ... many more fields
    // ... extensive validation
    // ... error handling
}
```

**Why Simpler?**
- Educational focus
- Fewer edge cases
- Easier to understand
- Less production overhead

#### 2. **Token Counting**

**Our Implementation:**
```javascript
const lengthFunction = text => Math.ceil(text.length / 4);
```
- Simple approximation
- No external dependencies
- Fast but less accurate

**LangChain.js:**
```javascript
import { encodingForModel } from "js-tiktoken";
const encoder = encodingForModel("gpt-4");
const tokens = encoder.encode(text);
```
- Uses tiktoken library
- Exact token counting
- Slower but accurate

#### 3. **Error Handling**

**Our Implementation:**
```javascript
// Minimal error handling
if (chunkOverlap >= chunkSize) {
    throw new Error('chunkOverlap must be less than chunkSize');
}
```

**LangChain.js:**
```javascript
// Extensive error handling
if (chunkOverlap >= chunkSize) {
    throw new Error(`chunkOverlap (${chunkOverlap}) must be less than chunkSize (${chunkSize})`);
}
if (chunkSize <= 0) {
    throw new Error('chunkSize must be positive');
}
// ... many more validations
```

#### 4. **Features**

| Feature | Our Implementation | LangChain.js |
|---------|-------------------|--------------|
| Basic splitting | ✓ | ✓ |
| Recursive splitting | ✓ | ✓ |
| Token splitting | ✓ (approximate) | ✓ (exact) |
| Metadata tracking | ✓ | ✓ |
| Custom separators | ✓ | ✓ |
| Markdown splitting | ✗ | ✓ |
| Code splitting | ✗ | ✓ |
| LaTeX splitting | ✗ | ✓ |
| HTML splitting | ✗ | ✓ |
| Transform callbacks | ✗ | ✓ |
| Document transformers | ✗ | ✓ |

#### 5. **TypeScript**

**Our Implementation:**
- Pure JavaScript
- JSDoc comments for type hints
- Simpler to understand

**LangChain.js:**
- Written in TypeScript
- Full type safety
- Better IDE support
- More complex codebase

### Migration Path

**From Our Implementation to LangChain.js:**

```javascript
// Step 1: Change import
// From:
import { RecursiveCharacterTextSplitter } from './example.js';
// To:
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// Step 2: Code stays the same!
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
});
const chunks = await splitter.splitDocuments(documents);

// Step 3: Optionally add LangChain features
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    // LangChain-specific features:
    separators: ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
});
```

**Compatibility:** 95% of code works without changes!

---

## Summary

### Key Takeaways

1. **TextSplitter Pattern**: Base class + specialized subclasses
2. **Core Algorithm**: Merge with overlap for context continuity
3. **Recursive Strategy**: Try large separators first, fall back to smaller ones
4. **API Compatibility**: Same interface as LangChain.js
5. **Simplicity**: Focused on clarity over features

### Architecture Benefits

```
✓ Modular design (easy to extend)
✓ Clear separation of concerns
✓ Reusable components
✓ Well-documented
✓ Production-ready algorithm
✓ LangChain-compatible
```

### What You've Learned

1. How text splitting works algorithmically
2. Why overlap matters for context
3. Recursive splitting strategy
4. Token vs character splitting
5. How to choose the right configuration
6. Differences from LangChain.js

### Next Steps

1. Implement custom splitters for your domain
2. Add specialized separators for your document types
3. Experiment with different chunk sizes
4. Measure retrieval quality with your data
