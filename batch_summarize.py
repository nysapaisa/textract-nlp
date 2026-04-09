"""
batch_summarize.py — Summarize all PDFs in a folder at once.

Usage:
    python batch_summarize.py /path/to/pdf/folder
    python batch_summarize.py /path/to/pdf/folder --sentences 4 --output-dir summaries/
"""

import argparse
import sys
from pathlib import Path

from summarizer import extract_text, summarize, format_output


def batch_summarize(folder: str, sentences: int, output_dir: str | None):
    folder_path = Path(folder)
    pdfs = sorted(folder_path.glob("*.pdf"))

    if not pdfs:
        print(f"No PDF files found in: {folder}")
        return

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Found {len(pdfs)} PDF(s) in '{folder_path.name}'\n")

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            text, pages = extract_text(str(pdf))
            result = summarize(text, num_sentences=sentences)
            output = format_output(str(pdf), pages, result)
            print(output)

            if out_path:
                save_path = out_path / (pdf.stem + "_summary.txt")
                save_path.write_text(output, encoding="utf-8")
                print(f"  💾 Saved → {save_path}")

        except Exception as e:
            print(f"  ❌ Failed: {e}", file=sys.stderr)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Batch-summarize all PDFs in a directory."
    )
    parser.add_argument("folder", help="Directory containing PDF files")
    parser.add_argument("--sentences", "-s", type=int, default=5)
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save summary text files")
    args = parser.parse_args()

    if not Path(args.folder).is_dir():
        print(f"❌ Not a directory: {args.folder}", file=sys.stderr)
        sys.exit(1)

    batch_summarize(args.folder, args.sentences, args.output_dir)


if __name__ == "__main__":
    main()
