"""
PDF Text Extractor & NLP Summarizer
Extracts text from PDFs and generates summaries using TF-IDF scoring.
"""

import re
import math
import argparse
import sys
from collections import Counter
from pathlib import Path
from langdetect import detect, DetectorFactory, LangDetectException
from stopwordsiso import stopwords as iso_stopwords, has_lang

DetectorFactory.seed = 0

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ── Stopwords (built-in, no NLTK download needed) ────────────────────────────

STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","get","got","had","hadn't","has","hasn't",
    "have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's","i","i'd",
    "i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
    "itself","let's","me","more","most","mustn't","my","myself","no","nor",
    "not","of","off","on","once","only","or","other","ought","our","ours",
    "ourselves","out","over","own","same","shan't","she","she'd","she'll",
    "she's","should","shouldn't","so","some","such","than","that","that's",
    "the","their","theirs","them","themselves","then","there","there's",
    "these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we",
    "we'd","we'll","we're","we've","were","weren't","what","what's","when",
    "when's","where","where's","which","while","who","who's","whom","why",
    "why's","will","with","won't","would","wouldn't","you","you'd","you'll",
    "you're","you've","your","yours","yourself","yourselves","also","many",
    "use","used","using","may","one","two","well","new","like","just","can",
    "make","made","much","even","however","thus","hence","therefore","shall",
    "whether","within","without","yet","already","often","could","might",
    "such","across","along","although","among","another","around","back",
    "been","both","came","come","currently","either","enable","especially",
    "example","first","following","given","includes","including","known",
    "last","later","less","made","means","must","need","needs","next","now",
    "number","often","other","part","parts","per","rather","related","said",
    "second","several","since","still","taken","take","things","third",
    "though","three","time","times","today","toward","towards","type","types",
    "various","way","ways","where","which","whole","widely","will","used",
}


# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_with_pdfplumber(path: str) -> tuple[str, int]:
    """Extract text using pdfplumber (better layout handling)."""
    if pdfplumber is None:
        raise ImportError("pdfplumber not installed")
    pages_text = []
    with pdfplumber.open(path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n\n".join(pages_text), num_pages


def extract_with_pypdf(path: str) -> tuple[str, int]:
    """Extract text using pypdf (fallback)."""
    if PdfReader is None:
        raise ImportError("pypdf not installed")
    reader = PdfReader(path)
    num_pages = len(reader.pages)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text), num_pages


def extract_text(pdf_path: str) -> tuple[str, int]:
    """
    Extract text from a PDF, trying pdfplumber first, then pypdf.
    Returns (text, page_count).
    """
    path = str(Path(pdf_path).resolve())

    for extractor, name in [(extract_with_pdfplumber, "pdfplumber"),
                             (extract_with_pypdf, "pypdf")]:
        try:
            text, pages = extractor(path)
            if text.strip():
                print(f"  ✓ Extracted text using {name} ({pages} pages, "
                      f"{len(text):,} characters)")
                return text, pages
        except ImportError:
            continue
        except Exception as e:
            print(f"  ⚠ {name} failed: {e}", file=sys.stderr)

    raise RuntimeError("No text could be extracted. Is this a scanned PDF?")


# ── NLP Utilities ─────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language from text sample. Falls back to English."""
    try:
        return detect(text[:3000])
    except LangDetectException:
        return "en"


def get_stopwords(lang_code: str) -> set:
    """Return stopwords for detected language, falling back to English."""
    lang_code = lang_code.lower()
    if has_lang(lang_code):
        return iso_stopwords(lang_code)
    return iso_stopwords("en")


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no NLTK needed)."""
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.',
                  r'\1<DOT>', text)
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.replace('<DOT>', '.').strip() for s in raw]
    return [s for s in sentences if len(s.split()) >= 4]


def tokenize_words(text: str, stop_words: set = None) -> list[str]:
    if stop_words is None:
        stop_words = STOPWORDS
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in stop_words]


def compute_tfidf_scores(sentences: list[str], stop_words: set = None) -> dict[int, float]:
    if stop_words is None:
        stop_words = STOPWORDS
    N = len(sentences)
    tokenized = [tokenize_words(s, stop_words) for s in sentences]

    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    scores: dict[int, float] = {}
    for idx, tokens in enumerate(tokenized):
        if not tokens:
            scores[idx] = 0.0
            continue
        tf = Counter(tokens)
        score = 0.0
        for word, count in tf.items():
            idf = math.log((N + 1) / (df[word] + 1)) + 1
            score += (count / len(tokens)) * idf
        scores[idx] = score

    return scores


def position_boost(idx: int, total: int) -> float:
    """Boost sentences near the beginning and end of the document."""
    rel = idx / max(total - 1, 1)
    if rel <= 0.15:
        return 1.25
    if rel >= 0.85:
        return 1.10
    return 1.0


# ── Q&A ───────────────────────────────────────────────────────────────────────

def answer_question(text: str, question: str, top_n: int = 4) -> list[str]:
    """
    Find the most relevant sentences in the PDF for a given question.
    Uses cosine similarity between question vector and sentence vectors.
    """
    lang = detect_language(text)
    stop_words = get_stopwords(lang)

    sentences = tokenize_sentences(text)
    if not sentences:
        return ["No readable text found."]

    question_tokens = tokenize_words(question, stop_words)
    if not question_tokens:
        return ["Question too vague — try using more specific words."]

    # Build vocabulary from sentences + question together for IDF
    all_docs = sentences + [question]
    N = len(all_docs)
    tokenized = [tokenize_words(s, stop_words) for s in all_docs]

    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    def get_vector(tokens):
        tf = Counter(tokens)
        vec = {}
        for word, count in tf.items():
            idf = math.log((N + 1) / (df[word] + 1)) + 1
            vec[word] = (count / len(tokens)) * idf
        return vec

    def cosine_similarity(vec_a, vec_b):
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(vec_a[w] * vec_b[w] for w in common)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    question_vec = get_vector(question_tokens)

    scored = []
    for i, tokens in enumerate(tokenized[:-1]):  # exclude question itself
        if not tokens:
            continue
        sent_vec = get_vector(tokens)
        score = cosine_similarity(question_vec, sent_vec)
        scored.append((score, sentences[i]))

    top = sorted(scored, reverse=True)[:top_n]
    top_sentences = [s for score, s in top if score > 0]

    if not top_sentences:
        return ["No relevant answer found in the document."]

    return top_sentences


# ── Summarization ─────────────────────────────────────────────────────────────

def summarize(text: str,
              num_sentences: int = 5,
              min_word_ratio: float = 0.05) -> dict:

    lang = detect_language(text)
    stop_words = get_stopwords(lang)

    sentences = tokenize_sentences(text)
    total_sentences = len(sentences)

    if total_sentences == 0:
        return {"summary": "No readable text found.", "word_count": 0,
                "sentence_count": 0, "compression": 0.0, "top_keywords": [],
                "language": lang}

    num_sentences = min(num_sentences, total_sentences)

    tfidf_scores = compute_tfidf_scores(sentences, stop_words)
    final_scores = {
        idx: score * position_boost(idx, total_sentences)
        for idx, score in tfidf_scores.items()
    }

    top_indices = sorted(
        sorted(final_scores, key=final_scores.get, reverse=True)[:num_sentences]
    )
    summary_sentences = [sentences[i] for i in top_indices]

    cleaned = []
    for s in summary_sentences:
        s = s.strip()
        if s and s[-1] not in ".!?":
            s += "."
        cleaned.append(s)

    summary = " ".join(cleaned)

    all_words = tokenize_words(text, stop_words)
    word_freq = Counter(all_words)
    total_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    summary_words = len(re.findall(r'\b[a-zA-Z]+\b', summary))
    compression = round(summary_words / total_words * 100, 1) if total_words else 0

    return {
        "summary": summary,
        "word_count": total_words,
        "sentence_count": total_sentences,
        "compression": compression,
        "top_keywords": [w for w, _ in word_freq.most_common(10)],
        "language": lang,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "═" * 60)
    print("   📄 PDF Summarizer — NLP-Powered Text Extraction")
    print("═" * 60)


def format_output(pdf_path: str, pages: int, result: dict) -> str:
    bar = "─" * 60
    lines = [
        bar,
        f"📂 File     : {Path(pdf_path).name}",
        f"📄 Pages    : {pages}",
        f"📝 Words    : {result['word_count']:,}",
        f"🔢 Sentences: {result['sentence_count']}",
        f"📉 Summary  : {result['compression']}% of original length",
        bar,
        "",
        "✨ SUMMARY",
        "─" * 60,
        result["summary"],
        "",
        "─" * 60,
        "🔑 TOP KEYWORDS",
        "  " + " · ".join(result["top_keywords"]),
        bar,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and generate an NLP summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summarizer.py report.pdf
  python summarizer.py report.pdf --sentences 7
  python summarizer.py report.pdf --sentences 3 --output summary.txt
        """,
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument(
        "--sentences", "-s", type=int, default=5,
        help="Number of sentences in the summary (default: 5)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Save summary to this file (optional)"
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"❌ Error: File not found — {args.pdf}", file=sys.stderr)
        sys.exit(1)

    print_banner()
    print(f"\n⏳ Processing: {args.pdf}")

    try:
        text, pages = extract_text(args.pdf)
    except RuntimeError as e:
        print(f"\n❌ Extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  ✓ Running TF-IDF summarization ({args.sentences} sentences)…")
    result = summarize(text, num_sentences=args.sentences)

    output = format_output(args.pdf, pages, result)
    print("\n" + output)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\n💾 Summary saved to: {args.output}")


if __name__ == "__main__":
    main()