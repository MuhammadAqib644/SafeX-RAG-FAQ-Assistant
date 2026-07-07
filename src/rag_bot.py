"""
WEEK 2-3 — RAG (Retrieval-Augmented Generation) FAQ Bot
---------------------------------------------------------
How this differs from the TF-IDF version:

TF-IDF matches on shared WORDS. RAG matches on shared MEANING.

Pipeline:
1. INDEXING (done once, offline):
   - Take every FAQ entry, turn it into an embedding vector using a real
     neural embedding model (all-MiniLM-L6-v2 from sentence-transformers).
   - These vectors capture semantic meaning, not just keywords — so
     "cyber security stuff" and "cybersecurity" end up close together
     in vector space even though they don't share tokens.
   - Store all vectors in a vector index (numpy array here for simplicity;
     swap for FAISS/Chroma/Pinecone at real scale).

2. RETRIEVAL (per user query):
   - Embed the user's question with the same model.
   - Compute cosine similarity against every FAQ vector.
   - Take the top-k most similar FAQ chunks.

3. GENERATION (per user query):
   - Feed the user's question + the retrieved FAQ chunks into an LLM
     with a prompt that says "only answer using this context".
   - The LLM writes a natural, conversational answer instead of just
     returning a canned FAQ answer verbatim.
   - This also lets it politely say "I don't know" when nothing relevant
     was retrieved, and handle multi-part or follow-up questions.

This file supports TWO generation modes:
  - "llm"       : calls a real LLM API (Anthropic Claude) to generate the answer.
                  Requires ANTHROPIC_API_KEY environment variable.
  - "extractive": no API key needed — returns the best-matching FAQ answer(s)
                  directly. Useful for offline testing/demo before you wire
                  up an API key.
"""

import json
import os
import numpy as np


class RagFAQBot:
    """
    embedding_backend:
      - "sentence-transformers" : REAL neural embeddings (all-MiniLM-L6-v2).
                                   This is what you should use in production.
                                   Requires internet access to huggingface.co
                                   to download the model on first run.
      - "tfidf-svd"             : Lightweight offline fallback (TF-IDF + SVD,
                                   i.e. classic LSA). Works with zero external
                                   downloads. Useful in locked-down networks
                                   (corporate proxies, CI runners, sandboxes)
                                   where huggingface.co isn't reachable. Lower
                                   semantic quality than real embeddings, but
                                   demonstrates the identical RAG pipeline.
    """

    def __init__(
        self,
        faq_path: str,
        top_k: int = 3,
        generation_mode: str = "extractive",
        embedding_backend: str = "sentence-transformers",
    ):
        self.faq_path = faq_path
        self.top_k = top_k
        self.generation_mode = generation_mode  # "llm" or "extractive"
        self.embedding_backend = embedding_backend

        self.faqs = self._load_faqs()
        self.corpus_texts = [f"{f['question']} {f['answer']}" for f in self.faqs]

        if embedding_backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            print("Loading embedding model (all-MiniLM-L6-v2)...")
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embed_fn = lambda texts: self._st_model.encode(
                texts, normalize_embeddings=True
            )
        elif embedding_backend == "tfidf-svd":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize

            print("Building offline TF-IDF+SVD embedding space (LSA)...")
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf_matrix = self._vectorizer.fit_transform(self.corpus_texts)
            n_components = min(50, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
            self._svd = TruncatedSVD(n_components=n_components, random_state=42)
            self._svd.fit(tfidf_matrix)

            def embed_fn(texts):
                vecs = self._svd.transform(self._vectorizer.transform(texts))
                return normalize(vecs)

            self._embed_fn = embed_fn
        else:
            raise ValueError(f"Unknown embedding_backend: {embedding_backend}")

        print("Embedding FAQ corpus (this happens once, then is reused)...")
        self.corpus_embeddings = np.array(self._embed_fn(self.corpus_texts))

    def _load_faqs(self):
        with open(self.faq_path, "r") as f:
            return json.load(f)

    def _retrieve(self, query: str):
        query_embedding = np.array(self._embed_fn([query]))[0]
        # normalized vectors -> dot product == cosine similarity
        scores = self.corpus_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][: self.top_k]
        return [
            {"faq": self.faqs[i], "score": float(scores[i])} for i in top_indices
        ]

    def _generate_extractive(self, query: str, retrieved: list) -> str:
        """No LLM call — just surface the best-matched answer(s)."""
        best = retrieved[0]
        if best["score"] < 0.30:
            return (
                "I couldn't find anything relevant to that in our FAQ. "
                "Try rephrasing, or reach out via safexsolutions.com/contact."
            )
        return best["faq"]["answer"]

    def _generate_llm(self, query: str, retrieved: list) -> str:
        """Calls Anthropic's Claude API to generate a natural answer
        grounded strictly in the retrieved FAQ context."""
        try:
            import anthropic
        except ImportError:
            return (
                "[LLM mode requires the 'anthropic' package: "
                "pip install anthropic] Falling back to extractive answer:\n"
                + self._generate_extractive(query, retrieved)
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return (
                "[LLM mode requires ANTHROPIC_API_KEY env var to be set] "
                "Falling back to extractive answer:\n"
                + self._generate_extractive(query, retrieved)
            )

        context = "\n\n".join(
            f"Q: {r['faq']['question']}\nA: {r['faq']['answer']}" for r in retrieved
        )

        client = anthropic.Anthropic(api_key=api_key)
        system_prompt = (
            "You are the FAQ support assistant for SafeX Solutions "
            "(safexsolutions.com), a global tech, cybersecurity, cloud, and "
            "digital marketing company. Answer the user's question using ONLY "
            "the context below. Be concise and friendly. If the context doesn't "
            "contain the answer, say you don't have that information and suggest "
            "contacting SafeX Solutions directly through their Contact page.\n\n"
            f"CONTEXT:\n{context}"
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        return response.content[0].text

    def ask(self, user_query: str) -> dict:
        retrieved = self._retrieve(user_query)

        if self.generation_mode == "llm":
            answer = self._generate_llm(user_query, retrieved)
        else:
            answer = self._generate_extractive(user_query, retrieved)

        return {
            "answer": answer,
            "retrieved_questions": [r["faq"]["question"] for r in retrieved],
            "top_score": retrieved[0]["score"],
            "method": f"rag-{self.generation_mode}",
        }


if __name__ == "__main__":
    faq_path = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.json")

    # NOTE: embedding_backend="tfidf-svd" is used here only because this
    # sandbox environment's network blocks huggingface.co. On your own
    # machine or a cloud VM with normal internet access, switch this to
    # embedding_backend="sentence-transformers" for real semantic embeddings.
    # Change generation_mode to "llm" once ANTHROPIC_API_KEY is set in your env.
    bot = RagFAQBot(
        faq_path,
        generation_mode="extractive",
        embedding_backend="tfidf-svd",
    )

    test_queries = [
        "Where are you guys based?",
        "cyber security stuff",
        "do you make websites",
        "what's your hourly rate",
        "do you have a canteen on site",
    ]

    print("=" * 60)
    print("WEEK 2-3 — RAG FAQ BOT — quick test run")
    print("=" * 60)
    for q in test_queries:
        result = bot.ask(q)
        print(f"\nUser: {q}")
        print(f"Top retrieved match: {result['retrieved_questions'][0]}")
        print(f"Top score: {result['top_score']:.3f}")
        print(f"Answer: {result['answer']}")
