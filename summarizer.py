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

def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no NLTK needed)."""
    # Handle common abbreviations so they don't break sentence splitting
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.',
                  r'\1<DOT>', text)
    # Split on sentence-ending punctuation
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Restore abbreviation dots
    sentences = [s.replace('<DOT>', '.').strip() for s in raw]
    # Remove very short fragments
    return [s for s in sentences if len(s.split()) >= 4]


def tokenize_words(text: str) -> list[str]:
    """Extract meaningful words, lower-cased, stopwords removed."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


def compute_tfidf_scores(sentences: list[str]) -> dict[int, float]:
    """
    Score each sentence using TF-IDF:
    - TF  = frequency of meaningful words in the sentence
    - IDF = log(total_sentences / sentences_containing_word)
    """
    N = len(sentences)
    tokenized = [tokenize_words(s) for s in sentences]

    # Document frequency (how many sentences contain each word)
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
            idf = math.log((N + 1) / (df[word] + 1)) + 1  # smoothed
            score += (count / len(tokens)) * idf
        scores[idx] = score

    return scores


def position_boost(idx: int, total: int) -> float:
    """Boost sentences near the beginning and end of the document."""
    rel = idx / max(total - 1, 1)
    if rel <= 0.15:   # first ~15%
        return 1.25
    if rel >= 0.85:   # last ~15%
        return 1.10
    return 1.0


# ── Summarization ─────────────────────────────────────────────────────────────

def summarize(text: str,
              num_sentences: int = 5,
              min_word_ratio: float = 0.05) -> dict:
    """
    Generate an extractive summary using TF-IDF scoring.

    Returns a dict with:
      - summary        : the summary text
      - word_count     : total words in original
      - sentence_count : total sentences found
      - compression    : compression ratio (summary / original words)
      - top_keywords   : top 10 keywords
    """
    sentences = tokenize_sentences(text)
    total_sentences = len(sentences)

    if total_sentences == 0:
        return {"summary": "No readable text found.", "word_count": 0,
                "sentence_count": 0, "compression": 0.0, "top_keywords": []}

    # Clamp num_sentences
    num_sentences = min(num_sentences, total_sentences)

    # Score sentences
    tfidf_scores = compute_tfidf_scores(sentences)
    final_scores = {
        idx: score * position_boost(idx, total_sentences)
        for idx, score in tfidf_scores.items()
    }

    # Pick top-N, preserving original document order
    top_indices = sorted(
        sorted(final_scores, key=final_scores.get, reverse=True)[:num_sentences]
    )
    summary_sentences = [sentences[i] for i in top_indices]

    # Post-process: ensure each sentence ends with punctuation
    cleaned = []
    for s in summary_sentences:
        s = s.strip()
        if s and s[-1] not in ".!?":
            s += "."
        cleaned.append(s)

    summary = " ".join(cleaned)

    # Stats
    all_words = tokenize_words(text)
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

    # 1. Extract
    try:
        text, pages = extract_text(args.pdf)
    except RuntimeError as e:
        print(f"\n❌ Extraction failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Summarize
    print(f"  ✓ Running TF-IDF summarization ({args.sentences} sentences)…")
    result = summarize(text, num_sentences=args.sentences)

    # 3. Display
    output = format_output(args.pdf, pages, result)
    print("\n" + output)

    # 4. Optionally save
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\n💾 Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
