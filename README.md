# 📄 PDF Summarizer — NLP-Powered Text Extraction
🌐 Live App → https://pdf-summarizer-xdi7.onrender.com

Extracts text from PDF files and generates concise summaries using **TF-IDF**
sentence scoring — a classic, robust NLP technique that requires no internet
access and no large model downloads.

---

## Features

| Capability | Detail |
|---|---|
| **PDF extraction** | pdfplumber (primary) → pypdf (fallback) |
| **Summarization** | Extractive TF-IDF with position boosting |
| **Keyword extraction** | Top-10 meaningful terms |
| **Compression ratio** | Shows how much shorter the summary is |
| **Batch mode** | Summarize every PDF in a folder at once |
| **Save output** | Optionally write summary to a `.txt` file |
| **Zero downloads** | No NLTK/spaCy corpora required |

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

### 6. Run tests
```bash
python tests.py
```

---

## How It Works

```
PDF File
   │
   ▼
┌──────────────────────────┐
│  Text Extraction Layer   │  pdfplumber → pypdf (fallback)
└──────────────────────────┘
   │  raw text
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
│  Top-N Selection         │  Preserves original reading order
└──────────────────────────┘
   │
   ▼
Summary + Keywords + Stats
```

### TF-IDF scoring

For each sentence:
- **TF** (term frequency): how often each meaningful word appears in that sentence
- **IDF** (inverse document frequency): how rare that word is across all sentences
- **Score** = sum of `(TF × IDF)` for every word in the sentence
- **Position boost**: ×1.25 for sentences in the first 15%, ×1.10 for the last 15%

Sentences are then ranked by score and the top-N are returned in their
original order to form a coherent summary.

---

## Project Structure

```
pdf_summarizer/
├── summarizer.py       # Core logic + CLI entry point
├── batch_summarize.py  # Batch-processing utility
├── tests.py            # Unit test suite
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Limitations

- **Scanned PDFs** (image-only) produce no extractable text — use an OCR tool
  (e.g. `tesseract`) to pre-process them first.
- **Extractive** summarization: the summary is composed of actual sentences
  from the document, not newly generated text.
- Very short documents (< 5 sentences) may yield thin summaries.
