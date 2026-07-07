"""
WEEK 1 — TF-IDF FAQ Bot
------------------------
How it works:
1. Load all FAQ questions.
2. Convert every question into a TF-IDF vector (a vector of word-importance scores).
3. When a user asks something, convert their question into a TF-IDF vector the same way.
4. Compute cosine similarity between the user's vector and every FAQ question vector.
5. Return the answer for the FAQ question with the highest similarity score
   (if it clears a minimum confidence threshold).

This has NO understanding of meaning — it only matches based on shared/similar words.
That's exactly why we upgrade to RAG in Week 2-3.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfFAQBot:
    def __init__(self, faq_path: str, similarity_threshold: float = 0.25):
        self.faq_path = faq_path
        self.similarity_threshold = similarity_threshold
        self.faqs = self._load_faqs()
        self.questions = [f["question"] for f in self.faqs]

        # ngram_range=(1,2) lets it match single words AND two-word phrases
        # (e.g. "web development"), which noticeably improves match quality.
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def _load_faqs(self):
        with open(self.faq_path, "r") as f:
            return json.load(f)

    def ask(self, user_query: str) -> dict:
        query_vec = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, self.question_vectors)[0]

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.similarity_threshold:
            return {
                "answer": (
                    "I'm not confident I have an answer to that in our FAQ. "
                    "Could you rephrase, or contact us directly through the "
                    "Contact page at safexsolutions.com/contact?"
                ),
                "matched_question": None,
                "confidence": float(best_score),
                "method": "tfidf",
            }

        matched_faq = self.faqs[best_idx]
        return {
            "answer": matched_faq["answer"],
            "matched_question": matched_faq["question"],
            "confidence": float(best_score),
            "method": "tfidf",
        }


if __name__ == "__main__":
    faq_path = os.path.join(os.path.dirname(__file__), "..", "data", "faqs.json")
    bot = TfidfFAQBot(faq_path)

    test_queries = [
        "Where are you guys based?",
        "do you make websites",
        "cyber security stuff",
        "what's your hourly rate",
        "do you have a canteen on site",  # should trigger low-confidence fallback
    ]

    print("=" * 60)
    print("WEEK 1 — TF-IDF FAQ BOT — quick test run")
    print("=" * 60)
    for q in test_queries:
        result = bot.ask(q)
        print(f"\nUser: {q}")
        print(f"Matched FAQ: {result['matched_question']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Answer: {result['answer']}")
