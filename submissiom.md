HW3 — Two retrievers and a model: what RAG actually sends
Name:Abay Adema Student ID: S23070175 Group: CSS4007-ENG-8 Repository:hw3-ademaabai-glitch
# Part A — relevance by hand

**Documents and query**

| | Text |
|---|---|
| D1 | Large language models can generate text and answer questions. |
| D2 | Retrieval-Augmented Generation retrieves relevant documents before generating an answer. |
| D3 | Embeddings represent text as numerical vectors. |
| Query | How does RAG retrieve information? |

Parameters: k1 = 1.5, b = 0.75, N = 3, idf(t) = ln((N − n(t) + 0.5) / (n(t) + 0.5)).
Tokenisation: lowercase, split at spaces and hyphens, drop punctuation, no stop words removed.

---

## A1 (a) — BM25 as written

**Tokens of each document and its length**

| Doc | Tokens | \|d\| |
|---|---|---|
| D1 | large, language, models, can, generate, text, and, answer, questions | 9 |
| D2 | retrieval, augmented, generation, retrieves, relevant, documents, before, generating, an, answer | 10 |
| D3 | embeddings, represent, text, as, numerical, vectors | 6 |

avgdl = (9 + 10 + 6) / 3 = 25 / 3 ≈ **8.333**

**Query terms:** how, does, rag, retrieve, information

**TF, DF and IDF of each query term**

| Query term | f(t, D1) | f(t, D2) | f(t, D3) | n(t) (DF) | idf(t) |
|---|---|---|---|---|---|
| how | 0 | 0 | 0 | 0 | ln(3.5 / 0.5) = ln 7 = 1.946 |
| does | 0 | 0 | 0 | 0 | 1.946 |
| rag | 0 | 0 | 0 | 0 | 1.946 |
| retrieve | 0 | 0 | 0 | 0 | 1.946 |
| information | 0 | 0 | 0 | 0 | 1.946 |

D2 contains `retrieval` and `retrieves`, but not the exact token `retrieve`. It also does not contain `rag` (only the words `retrieval-augmented generation`).

**BM25 of each document.** Every f(t, d) is 0, so every numerator idf · f · (k1 + 1) is 0:

- D1: 1.946·0 + 1.946·0 + 1.946·0 + 1.946·0 + 1.946·0 = 0
- D2: 1.946·0 + 1.946·0 + 1.946·0 + 1.946·0 + 1.946·0 = 0
- D3: 1.946·0 + 1.946·0 + 1.946·0 + 1.946·0 + 1.946·0 = 0

| Document | BM25 Score | Rank |
|---|---|---|
| D1 | 0 | 1 (three-way tie) |
| D2 | 0 | 1 (three-way tie) |
| D3 | 0 | 1 (three-way tie) |

This is not a mistake. None of the five query tokens equals any token in any document (`retrieve` ≠ `retrieves` and ≠ `retrieval`), so BM25 finds nothing. The order of the three documents is arbitrary.

---

## A1 (b) — BM25 with the stems from README §3

Stems used: retrieve / retrieves / retrieval → `retriev`; generate / generating / generation → `gener`; information → `inform`; does → `doe`.

**Stemmed tokens of each document and its length**

| Doc | Tokens | \|d\| |
|---|---|---|
| D1 | large, language, models, can, **gener**, text, and, answer, questions | 9 |
| D2 | **retriev**, augmented, **gener**, **retriev**, relevant, documents, before, **gener**, an, answer | 10 |
| D3 | embeddings, represent, text, as, numerical, vectors | 6 |

avgdl = 25 / 3 ≈ **8.333** (lengths do not change)

**Stemmed query terms:** how, doe, rag, retriev, inform

| Query term | f(t, D1) | f(t, D2) | f(t, D3) | n(t) (DF) | idf(t) |
|---|---|---|---|---|---|
| how | 0 | 0 | 0 | 0 | ln 7 = 1.946 |
| doe | 0 | 0 | 0 | 0 | 1.946 |
| rag | 0 | 0 | 0 | 0 | 1.946 |
| retriev | 0 | 2 | 0 | 1 | ln(2.5 / 1.5) = ln 1.667 = **0.511** |
| inform | 0 | 0 | 0 | 0 | 1.946 |

**BM25 of D2** (only `retriev` contributes, f = 2):

- \|d\| / avgdl = 10 / 8.333 = 1.2
- 1 − b + b · \|d\|/avgdl = 0.25 + 0.75 · 1.2 = 1.15
- denominator: f + k1 · 1.15 = 2 + 1.5 · 1.15 = 3.725
- numerator: idf · f · (k1 + 1) = 0.511 · 2 · 2.5 = 2.554
- score(D2) = 2.554 / 3.725 ≈ **0.686**

The other four terms have f = 0 in D2, so they add 0.

**BM25 of D1 and D3:** `retriev` does not occur in them, all f = 0, so score = 0.

| Document | BM25 Score | Rank |
|---|---|---|
| D2 | 0.686 | 1 |
| D1 | 0 | 2 (tie) |
| D3 | 0 | 2 (tie) |

**Which document does BM25 retrieve first, in (a) and in (b)? Why?**

- In **(a)** there is no real first document: all three score 0, because no query token matches any document token exactly. `retrieve` in the query and `retrieves` / `retrieval` in D2 are different strings for BM25.
- In **(b)** the first document is **D2**. Stemming maps `retrieve`, `retrieves` and `retrieval` to the same token `retriev`. It occurs twice in D2 and nowhere else, so D2 is the only document with a non-zero score. The other query words (how, doe, rag, inform) match nothing, so the whole result rests on that one stem.

---

## A2 — cosine similarity

Query = [0.8, 0.2, 0.5]
D1 = [0.7, 0.1, 0.4], D2 = [0.9, 0.3, 0.6], D3 = [0.1, 0.8, 0.2]

‖Query‖ = √(0.8² + 0.2² + 0.5²) = √(0.64 + 0.04 + 0.25) = √0.93 = **0.9644**

| Pair | Dot product (worked) | ‖Query‖ | ‖D‖ | Cosine |
|---|---|---|---|---|
| Query · D1 | 0.8·0.7 + 0.2·0.1 + 0.5·0.4 = 0.56 + 0.02 + 0.20 = **0.78** | 0.9644 | √(0.49 + 0.01 + 0.16) = √0.66 = 0.8124 | 0.78 / (0.9644 · 0.8124) = 0.78 / 0.7835 = **0.9956** |
| Query · D2 | 0.8·0.9 + 0.2·0.3 + 0.5·0.6 = 0.72 + 0.06 + 0.30 = **1.08** | 0.9644 | √(0.81 + 0.09 + 0.36) = √1.26 = 1.1225 | 1.08 / (0.9644 · 1.1225) = 1.08 / 1.0825 = **0.9977** |
| Query · D3 | 0.8·0.1 + 0.2·0.8 + 0.5·0.2 = 0.08 + 0.16 + 0.10 = **0.34** | 0.9644 | √(0.01 + 0.64 + 0.04) = √0.69 = 0.8307 | 0.34 / (0.9644 · 0.8307) = 0.34 / 0.8011 = **0.4244** |

| Document | Cosine Similarity | Rank |
|---|---|---|
| D2 | 0.9977 | 1 |
| D1 | 0.9956 | 2 |
| D3 | 0.4244 | 3 |

**Which document is most similar to the query?**
D2 (cosine ≈ 0.998).

**Is the ranking the same as BM25's?**

- Compared with A1 (a): no. BM25 gave three zeros (a tie), while cosine gives a clear order D2 > D1 > D3.
- Compared with A1 (b): only partly. The first place is the same (D2). After that they differ: BM25 puts D1 and D3 level at 0, while the embedding vectors show that D1 is almost as close to the query (0.996) as D2, and D3 is far away (0.424).

**Why can BM25 and embedding retrieval disagree?**

- The query says "RAG", but D2 never contains that word; it says "Retrieval-Augmented Generation". BM25 cannot connect the two (so (a) gives 0). An embedding model can place the abbreviation and its full name close together, which is why D2 comes first by cosine.
- `retrieve` (query) versus `retrieves` / `retrieval` (D2) are different strings for BM25 until stemming merges them. For an embedding they are the same meaning.
- D1 ("language models can generate text and answer questions") shares no word with the query, so BM25 gives it 0. But it is close in topic (language models answering questions), so cosine ranks it second, far above D3 (vectors for text).
- BM25 rewards only words that match exactly; embeddings reward closeness of meaning. So BM25 is strong when the exact string is present, and embeddings are strong when the wording differs but the meaning is the same.

Note: the vectors in A2 are given in the task, not produced by a real model, so this is an illustration rather than a measurement.

---
```

Expected output: (a) all scores 0; (b) D2 = 0.6857, D1 = D3 = 0; cosines 0.9956 / 0.9977 / 0.4244.
