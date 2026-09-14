# Embeddings — Foundations Module

Hands-on foundations for embeddings: what they are, how to generate them,
how to compare them, and how to use them to build a minimal semantic
search engine. This is meant to be the first building block under
`01-foundations/` in a larger Agentic AI repo — later modules (retrieval,
tool use, planning, memory) build on the concepts here.

## What's in here

```
embedding/
├── README.md
├── requirements.txt
├── embedding_demo.py   # embedding providers (offline + local model + OpenAI)
├── similarity.py        # cosine / dot / euclidean similarity + top-k ranking
├── search.py             # minimal in-memory vector search engine
└── tests/
    └── test_similarity.py
```

## Concepts covered

1. **What an embedding is** — a fixed-length vector representation of text,
   where geometric closeness (in some metric) approximates semantic
   closeness.
2. **Providers vs. mechanics** — `embedding_demo.py` separates "how do I
   get a vector for this text" (swappable providers) from "what do I do
   with vectors" (`similarity.py`), which is the same separation you'll
   see in real RAG/agent stacks.
3. **Similarity metrics** — cosine similarity, dot product, and Euclidean
   distance, and why cosine is the default for text embeddings (it
   ignores magnitude, which for text often just reflects length).
4. **Search as ranking** — `search.py` shows that "semantic search" is
   just "embed everything once, embed the query, rank by similarity" —
   an O(n) brute-force baseline before reaching for FAISS/HNSW/a vector DB.

## Setup

```bash
cd foundations/embedding
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You do **not** need an API key or even `sentence-transformers` installed
to run the demos or tests — everything works offline out of the box via
a deterministic hash-based fallback embedding. Install
`sentence-transformers` and/or set `OPENAI_API_KEY` to upgrade to real
semantic embeddings without changing any code (see "Providers" below).

## Usage

### 1. Run the embedding demo

```bash
python embedding_demo.py
```

This picks the best available provider (OpenAI if `OPENAI_API_KEY` is
set, else `sentence-transformers` if installed, else the offline hash
fallback), embeds four example sentences, and prints their pairwise
cosine similarity.

### 2. Run the search demo

```bash
python search.py
```

Builds a tiny 5-document in-memory index and runs a query against it,
printing the top-3 ranked results with scores.

### 3. Use it in your own code

```python
from embedding_demo import get_default_provider
from similarity import cosine_similarity
from search import VectorSearchEngine

# Similarity between two pieces of text
provider = get_default_provider()
a = provider.embed_one("Claude is an AI assistant.")
b = provider.embed_one("Claude helps answer questions.")
print(cosine_similarity(a, b))

# Minimal semantic search
engine = VectorSearchEngine()
engine.add_documents([
    {"id": "doc1", "text": "Agentic AI plans and calls tools."},
    {"id": "doc2", "text": "Embeddings turn text into vectors."},
])
for result in engine.search("what is an embedding?", k=2):
    print(result)

# Persist and reload an index
engine.save("my_index.json")
reloaded = VectorSearchEngine()
reloaded.load("my_index.json")
```

## Providers

| Provider | Install | Needs API key? | Quality |
|---|---|---|---|
| `SimpleHashEmbeddingProvider` | none (built-in) | No | Lexical overlap only — not real semantics. Good for tests/offline demos. |
| `SentenceTransformerEmbeddingProvider` | `pip install sentence-transformers` | No (local model) | Real semantic embeddings, runs on CPU. |
| `OpenAIEmbeddingProvider` | `pip install openai` | Yes (`OPENAI_API_KEY`) | Real semantic embeddings, hosted API. |

`get_default_provider()` in `embedding_demo.py` auto-selects in that
priority order, so the same code works whether you're offline, running
locally, or wired up to OpenAI.

## Running the tests

```bash
pytest tests/ -v
```

All tests run fully offline using `SimpleHashEmbeddingProvider` — no
model download or network access required, so this suite is safe to run
in CI.

## Design notes / things worth noticing

- **Batching matters.** `VectorSearchEngine.add_documents` embeds all
  texts in a single call to the provider rather than looping per-document
  — this is the difference between one API call and N API calls in a
  real system.
- **Cosine vs. Euclidean vs. dot product** aren't interchangeable —
  `similarity.py` documents when each is appropriate, and
  `euclidean_to_similarity()` exists specifically so you can rank
  euclidean results on the same "higher is better" footing as cosine.
- **Brute-force search is O(n).** Fine here; the next foundations module
  should introduce an approximate nearest-neighbor index (FAISS/HNSW) and
  benchmark it against this baseline.
- **Swapping providers doesn't make old embeddings comparable to new
  ones.** If you change providers, re-embed your corpus — `search.py`'s
  `load()` docstring calls this out explicitly.

## Next steps in `01-foundations/`

- `chunking/` — strategies for splitting documents before embedding
- `vector-stores/` — swapping the brute-force index for FAISS/Chroma/pgvector
- `rag-basics/` — combining retrieval with an LLM call
