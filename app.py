import gradio as gr
from summarizer import extract_text, summarize, answer_question, generate_wordcloud

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

h1 {
    text-align: center !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.5rem !important;
}

.description {
    text-align: center !important;
    color: #a0aec0 !important;
    font-size: 1rem !important;
    margin-bottom: 2rem !important;
}

.panel, .gr-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px) !important;
}

textarea, input[type="text"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.95rem !important;
}

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

input[type="range"] {
    accent-color: #7c3aed !important;
}

label, .gr-label {
    color: #a78bfa !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

.gr-file {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(167, 139, 250, 0.4) !important;
    border-radius: 12px !important;
}

.gr-textbox {
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    color: #e2e8f0 !important;
}

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
        output = f"""🌐 Language: {result['language'].upper()}  |  📄 Pages: {pages}  |  📝 Words: {result['word_count']}  |  📉 Compression: {result['compression']}%

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


def run_qa(pdf_file, question):
    if pdf_file is None:
        return "⚠️ Please upload a PDF file."
    if not question.strip():
        return "⚠️ Please type a question."
    try:
        text, pages = extract_text(pdf_file.name)
        answers = answer_question(text, question)
        output = f"🔍 Question: {question}\n\n"
        output += "─" * 60 + "\n\n"
        for i, sentence in enumerate(answers, 1):
            output += f"{i}. {sentence}\n\n"
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_wordcloud(pdf_file):
    if pdf_file is None:
        return None
    try:
        text, pages = extract_text(pdf_file.name)
        img = generate_wordcloud(text)
        return img
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


with gr.Blocks(title="PDF Summarizer") as demo:
    gr.Markdown("# 📄 PDF Summarizer")
    gr.Markdown("Upload any PDF to summarise it, ask questions, or visualise its keywords.")

    with gr.Tabs():
        with gr.Tab("✨ Summarise"):
            with gr.Row():
                with gr.Column():
                    pdf_input_sum = gr.File(label="📂 Upload your PDF", file_types=[".pdf"])
                    slider = gr.Slider(minimum=3, maximum=10, value=5, step=1,
                                       label="Number of sentences in summary")
                    sum_btn = gr.Button("Summarise", variant="primary")
                with gr.Column():
                    sum_output = gr.Textbox(label="✨ Summary Output", lines=20)
            sum_btn.click(fn=run_summary, inputs=[pdf_input_sum, slider], outputs=sum_output)

        with gr.Tab("❓ Ask a Question"):
            with gr.Row():
                with gr.Column():
                    pdf_input_qa = gr.File(label="📂 Upload your PDF", file_types=[".pdf"])
                    question_input = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What are the main conclusions?"
                    )
                    qa_btn = gr.Button("Ask", variant="primary")
                with gr.Column():
                    qa_output = gr.Textbox(label="📖 Answer", lines=20)
            qa_btn.click(fn=run_qa, inputs=[pdf_input_qa, question_input], outputs=qa_output)

        with gr.Tab("☁️ Word Cloud"):
            with gr.Row():
                with gr.Column():
                    pdf_input_wc = gr.File(label="📂 Upload your PDF", file_types=[".pdf"])
                    wc_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    wc_output = gr.Image(label="🔑 Word Cloud", type="pil")
            wc_btn.click(fn=run_wordcloud, inputs=[pdf_input_wc], outputs=wc_output)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=custom_css
    )