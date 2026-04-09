"""
tests.py — Unit tests for pdf_summarizer
Run with: python tests.py
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from summarizer import (
    tokenize_sentences,
    tokenize_words,
    compute_tfidf_scores,
    summarize,
    position_boost,
    STOPWORDS,
)

SAMPLE_TEXT = """
Artificial intelligence (AI) refers to the simulation of human intelligence processes by machines.
Machine learning is a subset of AI that enables systems to learn from data automatically.
Deep learning uses neural networks with many layers to analyze complex patterns in data.
Natural language processing allows computers to understand and generate human language.
Computer vision enables machines to interpret visual information from images and videos.
Reinforcement learning trains agents to make decisions by rewarding desired behaviors.
Ethics in AI focuses on ensuring fairness, transparency, and accountability in algorithms.
Quantum computing may dramatically accelerate AI research in the coming decades.
Healthcare AI applications include medical image analysis, drug discovery, and diagnostics.
Self-driving vehicles rely on a combination of computer vision and reinforcement learning.
"""


class TestTokenization(unittest.TestCase):

    def test_tokenize_sentences_basic(self):
        text = "Hello world. This is a test sentence. Another long sentence here."
        sents = tokenize_sentences(text)
        self.assertGreaterEqual(len(sents), 1)

    def test_tokenize_sentences_min_length(self):
        text = "OK. This is a meaningful sentence. Sure."
        sents = tokenize_sentences(text)
        for s in sents:
            self.assertGreaterEqual(len(s.split()), 4)

    def test_tokenize_words_removes_stopwords(self):
        words = tokenize_words("The quick brown fox jumps over the lazy dog")
        self.assertNotIn("the", words)
        self.assertNotIn("over", words)
        self.assertIn("quick", words)
        self.assertIn("brown", words)

    def test_tokenize_words_lowercase(self):
        words = tokenize_words("Artificial Intelligence Research")
        self.assertTrue(all(w == w.lower() for w in words))

    def test_tokenize_words_min_length(self):
        words = tokenize_words("AI is a big deal")
        for w in words:
            self.assertGreaterEqual(len(w), 3)


class TestScoring(unittest.TestCase):

    def test_tfidf_scores_keys_match_sentences(self):
        sentences = tokenize_sentences(SAMPLE_TEXT)
        scores = compute_tfidf_scores(sentences)
        self.assertEqual(set(scores.keys()), set(range(len(sentences))))

    def test_tfidf_scores_non_negative(self):
        sentences = tokenize_sentences(SAMPLE_TEXT)
        scores = compute_tfidf_scores(sentences)
        for v in scores.values():
            self.assertGreaterEqual(v, 0.0)

    def test_position_boost_start(self):
        self.assertGreater(position_boost(0, 20), 1.0)

    def test_position_boost_end(self):
        self.assertGreater(position_boost(19, 20), 1.0)

    def test_position_boost_middle(self):
        self.assertEqual(position_boost(10, 20), 1.0)


class TestSummarize(unittest.TestCase):

    def test_summary_returns_correct_keys(self):
        result = summarize(SAMPLE_TEXT, num_sentences=3)
        for key in ("summary", "word_count", "sentence_count",
                    "compression", "top_keywords"):
            self.assertIn(key, result)

    def test_summary_not_empty(self):
        result = summarize(SAMPLE_TEXT, num_sentences=3)
        self.assertTrue(result["summary"].strip())

    def test_summary_sentence_count(self):
        result = summarize(SAMPLE_TEXT, num_sentences=3)
        # Should produce at most 3 sentences
        sentences = tokenize_sentences(result["summary"])
        self.assertLessEqual(len(sentences), 3 + 1)  # +1 tolerance for edge cases

    def test_summary_compression_range(self):
        result = summarize(SAMPLE_TEXT, num_sentences=3)
        self.assertGreater(result["compression"], 0)
        self.assertLess(result["compression"], 100)

    def test_top_keywords_is_list(self):
        result = summarize(SAMPLE_TEXT, num_sentences=3)
        self.assertIsInstance(result["top_keywords"], list)
        self.assertGreater(len(result["top_keywords"]), 0)

    def test_empty_text(self):
        result = summarize("", num_sentences=3)
        self.assertEqual(result["word_count"], 0)

    def test_single_sentence(self):
        text = "This is a single sentence that has enough words to qualify."
        result = summarize(text, num_sentences=5)
        self.assertIn("single", result["summary"].lower())

    def test_clamping_num_sentences(self):
        # Asking for more sentences than available shouldn't crash
        result = summarize(SAMPLE_TEXT, num_sentences=1000)
        self.assertTrue(result["summary"].strip())

    def test_word_count_positive(self):
        result = summarize(SAMPLE_TEXT)
        self.assertGreater(result["word_count"], 0)


class TestStopwords(unittest.TestCase):

    def test_common_words_in_stopwords(self):
        for word in ("the", "is", "are", "and", "of", "in", "to"):
            self.assertIn(word, STOPWORDS)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__("__main__"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
