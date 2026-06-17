import os
from pathlib import Path

import gradio as gr
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name(".matplotlib")))

from nemo.collections.asr.models import EncDecSpeakerLabelModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSS = """
    .gradio-container {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .idv-title {
        max-width: 860px;
        margin: 0 auto 8px;
        text-align: center;
    }
    .idv-title h1 {
        margin-bottom: 8px;
        letter-spacing: 0;
    }
    .idv-title p {
        color: #4b5563;
        margin: 0;
    }
    .engine-note {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        padding: 8px 12px;
        border: 1px solid #c7d7fe;
        border-radius: 999px;
        color: #1d4ed8;
        background: #eff6ff;
        font-size: 14px;
        font-weight: 650;
    }
    .idv-step {
        max-width: 860px;
        margin: 14px auto;
        padding: 18px;
        border: 1px solid #dde3ed;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 8px 22px rgba(18, 28, 45, 0.06);
    }
    .idv-step h3 {
        margin-top: 0;
    }
    .idv-step .block,
    .idv-step .form,
    .idv-step .wrap {
        border-color: transparent !important;
        box-shadow: none !important;
    }
    .idv-audio {
        border: 1px solid #e5eaf3 !important;
        border-radius: 8px !important;
        background: #f8fafc !important;
    }
    .idv-upload {
        margin-top: 12px;
    }
    .idv-upload .label-wrap {
        font-weight: 650;
    }
    .idv-actions {
        max-width: 860px;
        margin: 14px auto;
    }
    .idv-result {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 20px;
        background: #ffffff;
        max-width: 520px;
        margin: 0 auto;
        text-align: center;
        box-shadow: 0 10px 26px rgba(18, 28, 45, 0.08);
    }
    .idv-result h2 {
        margin: 0 0 8px;
        font-size: 24px;
        line-height: 1.2;
    }
    .idv-result .score {
        margin: 0;
        font-size: 42px;
        line-height: 1.1;
        font-weight: 700;
    }
    .idv-result .meta {
        margin: 10px 0 0;
        color: #4b5563;
        font-size: 14px;
    }
    .result-icon {
        display: inline-grid;
        place-items: center;
        width: 44px;
        height: 44px;
        margin-bottom: 10px;
        border-radius: 999px;
        font-size: 24px;
        font-weight: 800;
    }
    .idv-pass {
        border-left: 6px solid #16803c;
    }
    .idv-pass h2,
    .idv-pass .score {
        color: #16803c;
    }
    .idv-pass .result-icon {
        color: #16803c;
        background: #e8f6ee;
    }
    .idv-fail {
        border-left: 6px solid #b42318;
    }
    .idv-fail h2,
    .idv-fail .score {
        color: #b42318;
    }
    .idv-fail .result-icon {
        color: #b42318;
        background: #fdebea;
    }
    .idv-error {
        border-left: 6px solid #b54708;
    }
    .idv-error h2 {
        color: #b54708;
    }
    .idv-error .result-icon {
        color: #b54708;
        background: #fff3e0;
    }
"""
READY_OUTPUT = """
    <div class="idv-result">
        <span class="result-icon">i</span>
        <h2>Ready</h2>
        <p class="meta">Run verification to calculate a match score.</p>
    </div>
"""
OUTPUT_MISSING = """
    <div class="idv-result idv-error">
        <span class="result-icon">!</span>
        <h2>Missing audio</h2>
        <p class="meta">Enrollment and verification samples are both required.</p>
    </div>
"""

THRESHOLD = 0.80

model_name = "nvidia/speakerverification_en_titanet_large"
model = EncDecSpeakerLabelModel.from_pretrained(model_name).to(device)


def render_result(passed, score_percent):
    status = "PASS" if passed else "FAIL"
    result_class = "idv-pass" if passed else "idv-fail"
    icon = "✓" if passed else "!"
    return f"""
        <div class="idv-result {result_class}">
            <span class="result-icon">{icon}</span>
            <h2>{status}</h2>
            <p class="score">{score_percent:.1f}%</p>
            <p class="meta">Voice match score. Threshold: {THRESHOLD * 100:.0f}%.</p>
        </div>
    """


def selected_audio(record_path, upload_path):
    return upload_path or record_path


def compare_sample_files(enrollment_path, verification_path):
    if not (enrollment_path and verification_path):
        return OUTPUT_MISSING

    embs1 = model.get_embedding(enrollment_path).squeeze()
    embs2 = model.get_embedding(verification_path).squeeze()
    
    #Length Normalize
    X = embs1 / torch.linalg.norm(embs1)
    Y = embs2 / torch.linalg.norm(embs2)
    
    # Score
    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2
    score_percent = float(similarity_score.detach().cpu()) * 100

    return render_result(score_percent >= THRESHOLD * 100, score_percent)


def compare_samples(
    enrollment_record_path,
    enrollment_upload_path,
    verification_record_path,
    verification_upload_path,
):
    enrollment_path = selected_audio(enrollment_record_path, enrollment_upload_path)
    verification_path = selected_audio(verification_record_path, verification_upload_path)
    return compare_sample_files(enrollment_path, verification_path)


examples = [
    ["data/id10270_5r0dWxy17C8-00001.wav", "data/id10270_5r0dWxy17C8-00002.wav"],
    ["data/id10271_1gtz-CUIygI-00001.wav", "data/id10271_1gtz-CUIygI-00002.wav"],
    ["data/id10270_5r0dWxy17C8-00001.wav", "data/id10271_1gtz-CUIygI-00001.wav"],
    ["data/id10270_5r0dWxy17C8-00002.wav", "data/id10271_1gtz-CUIygI-00002.wav"],
]

# Keep examples only when the data files exist locally. This makes the repo
# runnable even if you did not clone/download the original Space data folder.
examples = examples if all(Path(f).exists() for pair in examples for f in pair) else None

with gr.Blocks(title="IDV Voice Verification", css=CSS) as demo:
    gr.Markdown(
        """
        # IDV Voice Verification
        Enroll a reference voice sample, verify a second sample, then review the match decision.

        <span class="engine-note">IDV engine: NVIDIA TitaNet speaker verification</span>
        """,
        elem_classes=["idv-title"],
    )

    with gr.Column(elem_classes=["idv-step"]):
        gr.Markdown("### 1. Enroll")
        enrollment_record_audio = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record enrollment sample",
            elem_classes=["idv-audio"],
        )
        with gr.Accordion("Upload enrollment sample instead", open=False, elem_classes=["idv-upload"]):
            enrollment_upload_audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Drop or choose enrollment file",
                elem_classes=["idv-audio"],
            )

    with gr.Column(elem_classes=["idv-step"]):
        gr.Markdown("### 2. Verify")
        verification_record_audio = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record verification sample",
            elem_classes=["idv-audio"],
        )
        with gr.Accordion("Upload verification sample instead", open=False, elem_classes=["idv-upload"]):
            verification_upload_audio = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Drop or choose verification file",
                elem_classes=["idv-audio"],
            )

    with gr.Group(elem_classes=["idv-actions"]):
        verify_button = gr.Button("Verify identity", variant="primary")

    with gr.Column(elem_classes=["idv-step"]):
        gr.Markdown("### 3. Result")
        result = gr.HTML(value=READY_OUTPUT, label="")

    verify_button.click(
        fn=compare_samples,
        inputs=[
            enrollment_record_audio,
            enrollment_upload_audio,
            verification_record_audio,
            verification_upload_audio,
        ],
        outputs=result,
    )

    if examples:
        with gr.Accordion("Sample test cases", open=False):
            gr.Examples(
                examples=examples,
                inputs=[enrollment_record_audio, verification_record_audio],
                outputs=result,
                fn=compare_sample_files,
                cache_examples=False,
            )

demo.queue(max_size=5, default_concurrency_limit=4)

if __name__ == "__main__":
    # Defaults work for local Windows browser + WSL2 backend.
    # Override when needed, e.g. GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python app.py
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)
