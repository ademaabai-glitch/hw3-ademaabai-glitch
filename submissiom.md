HW3 — Two retrievers and a model: what RAG actually sends                        
Name:Abay Adema Student 
ID: S23070175
Group: CSS4007-ENG-8 
Repository:hw3-ademaabai-glitch
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

---
##PART B: the same two retrievers, from libraries##

## B1. BM25 (rank_bm25)

Library: `rank_bm25.BM25Okapi`, k1 = 1.5, b = 0.75.
Tokenisation: lowercase, `\w+` (works for Russian and Kazakh letters too), no stemming, no stop-word removal. A hyphen splits a word (`GR-305` becomes `gr` and `305`).

## B2. EmbeddingGemma (sentence-transformers)

Model `google/embeddinggemma-300m`, float32. I embedded the chunks once with `encode_document` and each query with `encode_query`. I used cosine similarity (`model.similarity`). The 74 vectors are stored in a plain NumPy array.

## Predictions for my own queries (written before running)

- **Q-09** "What is chyk and kese?" (relevant: KZ-25). I expect **BM25** to win. The words "chyk" and "kese" are rare and appear only in KZ-25.
- **Q-10** "Какая гора самая высокая в Казахстане?" (relevant: KZ-36). I expect **embeddings** to win. The question is in Russian, but the answer chunk is in English, so BM25 has no shared words.

## Results for all ten queries

| Query | Relevant | BM25 top 3 | BM25 hit@3 | Embedding top 3 | Embedding hit@3 |
|---|---|---|---|---|---|
| Q-01 | GO-08, GO-12, GO-23 | GO-08, GO-12, GO-06 | True | GO-08, GO-12, GO-23 | True |
| Q-02 | GO-01, GO-02, GO-03 | KZ-17, GO-20, KZ-25 | False | GO-01, GO-10, GO-20 | True |
| Q-03 | GO-09, GO-14 | GO-14, KZ-01, KZ-02 | True | GO-14, GO-09, GO-11 | True |
| Q-04 | GO-02 | GO-05, GO-17, GO-19 | False | GO-02, GO-05, GO-17 | True |
| Q-05 | KZ-05 | KZ-05, KZ-06, KZ-11 | True | KZ-05, KZ-06, KZ-03 | True |
| Q-06 | KZ-37, KZ-38 | KZ-37, KZ-38, KZ-03 | True | KZ-37, KZ-38, KZ-10 | True |
| Q-07 | GO-15 | GO-21, GO-16, GO-01 | False | GO-15, GO-18, GO-01 | True |
| Q-08 | GO-17 | GO-17, GO-05, GO-04 | True | GO-17, GO-18, GO-15 | True |
| Q-09 | KZ-25 | KZ-25, KZ-13, KZ-17 | True | KZ-25, KZ-12, KZ-39 | True |
| Q-10 | KZ-36 | GO-15, GO-23, KZ-01 | False | KZ-36, KZ-06, KZ-51 | True |

**hit@3 totals:** BM25 = 6 / 10, EmbeddingGemma = 10 / 10.

## B3. Comparison

**Both agree: Q-05** ("When was Astana renamed Nur-Sultan, and why?"). Both put KZ-05 first. It was easy because the question and the chunk share special words ("Astana", "renamed", "Nur-Sultan") that are almost only in KZ-05. So both methods find the same chunk.

**BM25 did better: none in my runs.** BM25 never had a hit where embeddings missed (BM25 6/10, embeddings 10/10), so I cannot name a real BM25 win. The closest is **Q-09** ("What is chyk and kese?"). I expected BM25 to win, but both put KZ-25 first, so I was only half right. BM25 matched the exact words "chyk" and "kese", which are only in KZ-25. After that match, its other results (KZ-13, KZ-17) were not useful.

**Embeddings did better: Q-07** ("Can the grant be sent to my mother's card?"). The right chunk is GO-15, written in Russian: "перевод на счёт родителей ... невозможен". BM25 missed it because the words do not match: "mother" is not "родителей" and "card" is not "счёт". Embeddings understand the meaning across languages, so GO-15 was first. The same happened in Q-10 (Russian question, English chunk) and Q-02 ("poor" and "financial help" are not in the handbook's words "low-income" and "need-based").

**Extra note, Q-04** ("What is the minimum GPA for the grant?"). BM25 put GO-05 first. It is the archived 2025 rule, and it uses the same words as the question. BM25 missed the right chunk GO-02. Embeddings found GO-02, but GO-05 was still second, so the old GPA of 3.0 can still reach the model.

---

# PART C: connect retrieval to the model

Pipeline: question → retrieve (BM25 or EmbeddingGemma) → top 3 chunks → prompt → `gpt-5.6-luna` → answer. System message: "Answer the question using only the provided context. If the answer cannot be found in the context, say that you do not know. Cite the context blocks you used, like [1] or [2][3]."

## C1. Look at what is sent

Question: C-01, "How much does a student in income band 2 receive, and when is it paid?" Top 3 from EmbeddingGemma: GO-04 (0.5845), GO-03 (0.4842), GO-15 (0.4506).

Full prompt sent to the model:

```text
--- SYSTEM ---
Answer the question using only the provided context.
If the answer cannot be found in the context, say that you do not know.
Cite the context blocks you used, like [1] or [2][3].

--- USER ---
Context:
[1] The grant amount depends only on the income band, not on the GPA. A student in band 1 receives 250,000 tenge for the academic year. A student in band 2 receives 150,000 tenge for the academic year. The amount is paid in two equal instalments, the first in October and the second in February. There is no partial grant: a student either receives the full amount for their band or nothing.

[2] The income band is set from the monthly household income per person, declared on form GR-210. Band 1: below 85,000 tenge per person per month. Band 2: from 85,000 up to 150,000 tenge per person per month. Band 3: above 150,000 tenge per person per month. Households in band 3 are not eligible for the need-based grant, although their students may still apply for merit scholarships, which are run separately.

[3] Грант выплачивается двумя равными частями — в октябре и в феврале — на банковский счёт, указанный студентом в форме GR-104. Счёт должен быть открыт на имя самого студента: перевод на счёт родителей или других лиц невозможен. Если реквизиты изменились, студент обязан обновить их в портале до 1 октября или до 1 февраля соответственно.

Question: How much does a student in income band 2 receive, and when is it paid?
```

| Part of the input | Tokens |
|---|---|
| System instructions | 44 |
| Question | 18 |
| Retrieved context | 288 |
| Labels and new lines in the user message | 4 |
| Total input (tiktoken, o200k_base) | 354 |
| usage.prompt_tokens (API) | 364 |

Chunks: 3. Tokens per chunk: 90, 104, 85. Context size: 288 tokens and 1,145 characters.

**Why the two totals are different.** tiktoken counts only the text I give it (354 tokens). The API also adds a few extra tokens for the chat format (role labels and separators between the messages), so `usage.prompt_tokens` is 364. The difference is 10 tokens. I saw the same gap in HW1.

**What does retrieval add to the model's input?** Three pieces of text (GO-04, GO-03, GO-15), put in the prompt as numbered blocks [1], [2], [3]. That is 288 tokens and 1,145 characters, about 80% of the whole input (288 of 364 tokens). Nothing else. The model does not open any document and does not see the other 71 chunks. It only reads the text in the prompt.

## C2. Three systems, six questions

| Question | Reference | LLM only | Agrees? | BM25 RAG | Agrees? | Embedding RAG | Agrees? |
|---|---|---|---|---|---|---|---|
| C-01 | 150,000 tenge, two equal payments (October, February) | No answer. Asks which scheme and country | No | "150,000 tenge for the academic year, paid in two equal instalments: October and February. [1]" | Yes | "150,000 tenge for the academic year, paid in two equal instalments of 75,000 tenge: October and February. [1]" | Yes |
| C-02 | 14 calendar days from the decision letter, form GR-305 | No answer. Says it depends on the country and asks for more details | No | "14 calendar days ... form GR-305. [3]" | Yes | "14 calendar days ... Form GR-305. [1][2]" | Yes |
| C-03 | GPA at least 2.67 (3.0 in GO-05 is the old 2025 rule) | No answer. Asks which grant, mentions the US Pell Grant | No | "2.67 on a 4.0 scale. [2]" | Yes | "2.67 on a 4.0 scale. [1]" | Yes |
| C-04 | Decree in October 2017 for 2025; in January 2021 moved to 2031 | "October 2017 ... original target 2025 ... now 2031" | Yes | "October 2017 ... initially 2025, later moved to 2031 in January 2021. [1][2]" | Yes | "October 2017 ... now due by 2031. [1][2]" | Yes |
| C-05 | 1963, Almaty Institute of National Economy; new name in 2016 | "1963 as the Alma-Ata Institute of National Economy" | Yes | "1963 as the Almaty Institute of National Economy. [1]" | Yes | "1963, originally called the Almaty Institute of National Economy. [1]" | Yes |
| C-06 | "I do not know" (the corpus does not say) | Assumes the US, talks about Pell Grant and loans, never says "I do not know" | No | "I don't know the amount ... master's students are covered by a separate scheme; ask your department. [2][3]" | Yes | "You cannot apply for this need-based grant ... Master's students are covered by a separate scheme ... [1][3]" | No |

Notes on my marking:
- C-01, Embedding RAG: "75,000 tenge" is not written in the context, but it is simply 150,000 divided by two equal payments, so it is correct and I mark it as agreeing.
- C-05, LLM only: "Alma-Ata" is the older name of Almaty, and the year and the institute are right, so I mark it as agreeing.
- C-06, Embedding RAG: the answer comes from GO-20, but it says "you cannot apply" as a sure fact and does not say "I do not know". The reference says the correct answer is that the corpus does not say, so I mark it as not agreeing.

**Agreement counts:** LLM only = 2 / 6, BM25 RAG = 6 / 6, Embedding RAG = 5 / 6.

LLM only was right only on C-04 and C-05, which are Wikipedia facts the model already knew. It failed on all four handbook questions (C-01, C-02, C-03, C-06), because the handbook is new text that the model has never seen.

## C3. Three questions taken apart

### C-03: "What is the minimum GPA needed to qualify for the need-based grant?"

- **BM25 retrieved:** GO-16 (15.7985), GO-02 (14.0587), GO-06 (13.8516).
- **EmbeddingGemma retrieved:** GO-02 (0.6286), GO-20 (0.5691), GO-05 (0.5451).
- **Sent to the model:** BM25 sent GO-16 (limit for combining grants), GO-02 ("cumulative GPA is at least 2.67 on the 4.0 scale") and GO-06 (form GR-104). Embeddings sent GO-02, GO-20 (who cannot apply) and GO-05 (the archived rule: "the minimum GPA for the grant was 3.0 ... do not apply to any application made for the 2026 academic year").
- **How retrieval changed the answer:** it made it correct and grounded. Without context the model could not answer. With context, both systems said "2.67 on a 4.0 scale" and cited the right block.
- **What was missed:** nothing in the answer. The fact was in the context both times, and the model ignored the archived GO-05 because that chunk says it does not apply to 2026. But the risk is real: BM25 ranked GO-02 only second, and embeddings put the old rule in the top 3.

### C-06: "I am a master's student. How much need-based grant can I get?"

- **BM25 retrieved:** GO-16 (16.9958), GO-01 (14.1725), GO-20 (13.7202).
- **EmbeddingGemma retrieved:** GO-20 (0.6664), GO-16 (0.6161), GO-01 (0.5539).
- **Sent to the model:** the same three chunks in both systems, in a different order. The key one is GO-20: "Master's and doctoral students are supported by a separate scheme, which this handbook does not describe."
- **How retrieval changed the answer:** without context, the model guessed the US and talked about Pell Grants and loans. With BM25 context, the answer was correct: "I don't know the amount ... ask your department." With embedding context, the answer was too sure: "You cannot apply."
- **What was missed:** this is a **generation problem, not a retrieval problem**. GO-20 was in the context in both systems (rank 3 for BM25, rank 1 for embeddings). The handbook says the master's scheme is "not described", but the embedding-based answer states a sure conclusion and does not say it does not know. The same chunks gave a correct answer in the BM25 system, so retrieval was not the difference.

### C-02: "My grant application was refused. How long do I have to appeal, and which form do I use?"

- **BM25 retrieved:** GO-05 (17.2414), GO-08 (14.5287), GO-12 (13.5735).
- **EmbeddingGemma retrieved:** GO-08 (0.6029), GO-12 (0.5591), GO-11 (0.5343).
- **Sent to the model:** BM25 sent GO-05 (the archived rule, not useful), GO-08 (form GR-305 is for appeals) and GO-12 ("appeal within 14 calendar days ... using form GR-305"). Embeddings sent GO-08, GO-12 and GO-11 (missing documents, not useful).
- **How retrieval changed the answer:** it made the answer specific and grounded. Without context, the model gave no number and no form. With context, both systems answered "14 calendar days, form GR-305" and cited the right blocks.
- **What was missed:** nothing in the answer. Retrieval was not perfect: BM25 put the useless archived GO-05 first, embeddings added the useless GO-11, and neither found the Russian GO-23. This did not matter, because GO-08 and GO-12 had the full facts and the model ignored the extra chunks. Useless chunks cost tokens but did not cause a wrong answer here.

---

# FINAL QUESTIONS

**1. What is the main difference between BM25 and embedding retrieval?**
BM25 looks for the same words in the question and the text. Embeddings look for the same meaning, even when the words are different.

**2. When can BM25 perform better than embeddings?**
When the question has a rare exact word, like a form code or a name. In my test, Q-09 had the rare words "chyk" and "kese", and BM25 found them easily.

**3. When can embeddings perform better than BM25?**
When the question and the text use different words for the same idea, or different languages. In Q-07, the question was about "mother's card", but the answer was in Russian ("счёт родителей"). Only embeddings found it.

**4. Why is cosine similarity used with embeddings?**
It checks if two vectors point in the same direction, not how long they are. So a long text does not win just because its vector is big.

**5. What information is actually sent to an LLM in a RAG system?**
Only text: the instruction, a few numbered chunks that were found, and the question. In C1 this was 364 tokens, and 288 of them were the retrieved chunks.

**6. Does the LLM search the original documents itself?**
No. My code does the search and puts the chunks in the prompt. The model only reads the prompt.

**7. What happens if retrieval returns an irrelevant chunk?**
The chunk goes into the prompt, and the model can use it by mistake. In my tests the model mostly ignored bad chunks, but they still cost tokens. An old rule like the archived GO-05 (GPA 3.0) can confuse a weaker model.

**8. Why can a bigger Top-K help or hurt? (K = 1, 3, 8 on C-03)**
With K = 1 the input was 177 tokens, with K = 3 it was 326, and with K = 8 it was 757. The answer was the same each time: "2.67 [1]". So a bigger K made the input about 4 times longer with no gain, because the right chunk was already first. A bigger K can help when the right chunk is not first, but it also adds noise and cost.

**9. Is a vector database required to build a RAG system?**
No. My notebook kept the 74 vectors in a simple NumPy array in memory. A vector database is only needed for very many chunks or for saving the index.

**10. Which retrieval approach worked better for this corpus?**
Embeddings worked better for retrieval. The hit@3 was 10/10 for embeddings and 6/10 for BM25. In C2, BM25 RAG agreed with the reference in 6/6 questions and Embedding RAG in 5/6 (the only miss was C-06, and that was a generation problem, not a retrieval problem). The corpus has English, Russian and Kazakh text, and the questions use different words than the text, so embeddings fit better for search. BM25 is still good for exact words. Both RAG systems clearly beat LLM only (2/6), because the model has never seen the handbook.

## Conclusion

Next I would build a hybrid search that uses both BM25 and embeddings, so I get exact matches and meaning matches. I would also remove old chunks like GO-05 from the search and add a rule: if no chunk is good enough, the system says "I do not know". I would also make the instruction stronger for cases like C-06, so the model says "I do not know" when the text only says something is "not described". This fixes the problems I saw: old rules in the context, useless chunks, and sure answers when the fact is missing.
