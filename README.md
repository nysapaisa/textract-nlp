# 📄 Textract-NLP — NLP-Powered PDF Summarizer
🌐 Live App → https://pdf-summarizer-xdi7.onrender.com

Extracts text from PDF files and generates concise, diverse summaries using **TF-IDF** sentence scoring combined with **MMR (Maximal Marginal Relevance)** a classic, robust NLP technique which requires no internet access and no large model downloads.

---

## Features

| Capability | Detail |
|---|---|
| **PDF extraction** | pdfplumber (primary) → pypdf (fallback) |
| **Summarization** | Extractive TF-IDF with position boosting |
| **Redundancy filter** | MMR algorithm removes near-duplicate sentences |
| **Q&A mode** | Ask questions about your PDF, get relevant answers |
| **Multi-language** | Auto-detects language, applies correct stopwords |
| **Keyword extraction** | Top-10 meaningful terms |
| **Compression ratio** | Shows how much shorter the summary is |
| **Batch mode** | Summarize every PDF in a folder at once |
| **Save output** | Optionally write summary to a `.txt` file |
| **Zero downloads** | No NLTK/spaCy corpora required |
| **Web UI** | Clean Gradio interface for non-technical users |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Summarize a single PDF
```bash
python summarizer.py report.pdf
```

### 3. Control summary length
```bash
python summarizer.py report.pdf --sentences 7
```

### 4. Save summary to file
```bash
python summarizer.py report.pdf --output summary.txt
```

### 5. Batch-summarize a folder
```bash
python batch_summarize.py ./pdfs/ --sentences 5 --output-dir ./summaries/
```

### 6. Launch the web UI
```bash
python app.py
```

### 7. Run tests
```bash
python tests.py
```

---

## How It Works

PDF File
│
▼
┌──────────────────────────┐
│  Text Extraction Layer   │  pdfplumber → pypdf (fallback)
└──────────────────────────┘
│  raw text
▼
┌──────────────────────────┐
│  Language Detection      │  langdetect → locale stopwords
└──────────────────────────┘
│  detected language
▼
┌──────────────────────────┐
│  Sentence Tokenization   │  Regex-based, handles abbreviations
└──────────────────────────┘
│  sentence list
▼
┌──────────────────────────┐
│   TF-IDF Sentence Scorer │  Measures importance of each sentence
│   + Position Boosting    │  Boosts intro/conclusion sentences
└──────────────────────────┘
│  ranked sentences
▼
┌──────────────────────────┐
│  MMR Redundancy Filter   │  Removes near-duplicate sentences
└──────────────────────────┘
│
▼
Summary + Keywords + Stats + Language

### TF-IDF scoring

For each sentence:
- **TF** (term frequency): how often each meaningful word appears in that sentence
- **IDF** (inverse document frequency): how rare that word is across all sentences
- **Score** = sum of `(TF × IDF)` for every word in the sentence
- **Position boost**: ×1.25 for sentences in the first 15%, ×1.10 for the last 15%

### MMR Redundancy Filter

After scoring, MMR selects sentences that are both **relevant** and **diverse**:
- First pick: highest TF-IDF scoring sentence
- Each subsequent pick: highest scoring sentence that is sufficiently different from already selected sentences
- Prevents repetitive summaries on documents with repeated phrasing

### Q&A Mode

Uses cosine similarity between a TF-IDF vector of the question and each sentence in the document to find the most relevant answers — no LLM required.

### Multi-language Support

Automatically detects the document language using `langdetect` and loads the appropriate stopword list via `stopwordsiso`. Supports 40+ languages including French, Spanish, German, Arabic, Hindi, and more.

---

## Project Structure
textract-nlp/
├── summarizer.py       # Core NLP logic + CLI entry point
├── app.py              # Gradio web UI
├── batch_summarize.py  # Batch-processing utility
├── tests.py            # Unit test suite
├── requirements.txt    # Python dependencies
└── README.md           # This file

---

## Requirements
pdfplumber
pypdf
langdetect
stopwordsiso
gradio

---

## Limitations

- **Scanned PDFs** (image-only) produce no extractable text — use an OCR tool (e.g. `tesseract`) to pre-process them first.
- **Extractive** summarization: the summary is composed of actual sentences from the document, not newly generated text.
- Very short documents (< 5 sentences) may yield thin summaries.
- CJK languages (Chinese, Japanese, Korean) use character-level tokenization which may affect summary quality.
