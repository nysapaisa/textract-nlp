import gradio as gr
from summarizer import extract_text, summarize

def run_summary(pdf_file, num_sentences):
    if pdf_file is None:
        return "⚠️ Please upload a PDF file."
    try:
        text, pages = extract_text(pdf_file.name)
        result = summarize(text, num_sentences=int(num_sentences))
        keywords = " · ".join(result["top_keywords"])
        output = f"""📄 Pages: {pages}  |  📝 Words: {result['word_count']}  |  📉 Compression: {result['compression']}%

✨ SUMMARY
{'─' * 60}
{result['summary']}

🔑 TOP KEYWORDS
{'─' * 60}
{keywords}
"""
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


demo = gr.Interface(
    fn=run_summary,
    inputs=[
        gr.File(label="📂 Upload your PDF", file_types=[".pdf"]),
        gr.Slider(minimum=3, maximum=10, value=5, step=1,
                  label="Number of sentences in summary")
    ],
    outputs=gr.Textbox(label="📋 Summary Output", lines=20),
    title="📄 PDF Summarizer",
    description="Upload any PDF and get a clean, concise summary powered by TF-IDF NLP.",
    examples=[],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
