import gradio as gr
from summarizer import extract_text, summarize

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    padding: 2rem !important;
}

/* Title */
h1 {
    text-align: center !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.5rem !important;
}

/* Description */
.description {
    text-align: center !important;
    color: #a0aec0 !important;
    font-size: 1rem !important;
    margin-bottom: 2rem !important;
}

/* Panels */
.panel, .gr-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px) !important;
}

/* Input/Output boxes */
textarea, input[type="text"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.95rem !important;
}

/* Submit button */
button.primary {
    background: linear-gradient(90deg, #7c3aed, #3b82f6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
}

/* Clear button */
button.secondary {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover {
    background: rgba(255,255,255,0.15) !important;
}

/* Slider */
input[type="range"] {
    accent-color: #7c3aed !important;
}

/* Labels */
label, .gr-label {
    color: #a78bfa !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* File upload */
.gr-file {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
}

/* Output textbox */
.gr-textbox {
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    color: #e2e8f0 !important;
}

/* Footer */
footer {
    display: none !important;
}
"""

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
    outputs=gr.Textbox(label="✨ Summary Output", lines=20),
    title="📄 PDF Summarizer",
    description="Upload any PDF and get a clean, concise summary powered by TF-IDF NLP.",
    theme=gr.themes.Soft(),
    css=custom_css
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
