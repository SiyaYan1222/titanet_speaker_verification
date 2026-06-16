import os
from pathlib import Path

import gradio as gr
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STYLE = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" integrity="sha256-YvdLHPgkqJ8DVUxjjnGVlMMJtNimJ6dYkowFFvp4kKs=" crossorigin="anonymous">
"""
OUTPUT_OK = (
    STYLE
    + """
    <div class="container">
        <div class="row"><h1 style="text-align: center">The provided samples are</h1></div>
        <div class="row"><h1 class="text-success" style="text-align: center">Same Speakers!!!</h1></div>
        <div class="row"><h1 class="display-1 text-success" style="text-align: center">similarity score: {:.1f}%</h1></div>
        <div class="row"><tiny style="text-align: center">(Similarity score must be atleast 80% to be considered as same speaker)</small><div class="row">

    </div>
"""
)
OUTPUT_FAIL = (
    STYLE
    + """
    <div class="container">
        <div class="row"><h1 style="text-align: center">The provided samples are from </h1></div>
        <div class="row"><h1 class="text-danger" style="text-align: center">Different Speakers!!!</h1></div>       
        <div class="row"><h1 class="display-1 text-danger" style="text-align: center">similarity score: {:.1f}%</h1></div>
        <div class="row"><tiny style="text-align: center">(Similarity score must be atleast 80% to be considered as same speaker)</small><div class="row">
    </div>
"""
)

THRESHOLD = 0.80

model_name = "nvidia/speakerverification_en_titanet_large"
model = EncDecSpeakerLabelModel.from_pretrained(model_name).to(device)


def compare_samples(path1, path2):
    if not (path1 and path2):
        return '<b style="color:red">ERROR: Please record audio for *both* speakers!</b>'

    embs1 = model.get_embedding(path1).squeeze()
    embs2 = model.get_embedding(path2).squeeze()
    
    #Length Normalize
    X = embs1 / torch.linalg.norm(embs1)
    Y = embs2 / torch.linalg.norm(embs2)
    
    # Score
    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2
    
    # Decision
    if similarity_score >= THRESHOLD:
        return OUTPUT_OK.format(similarity_score * 100)
    else:
        return OUTPUT_FAIL.format(similarity_score * 100)


inputs = [
    gr.Audio(sources=["microphone"], type="filepath", label="Speaker #1"),
    gr.Audio(sources=["microphone"], type="filepath", label="Speaker #2"),
]

upload_inputs = [
    gr.Audio(sources=["upload"], type="filepath",  label="Speaker #1"),
    gr.Audio(sources=["upload"], type="filepath", label="Speaker #2"),
]

description = (
    "This demonstration will analyze two recordings of speech and ascertain whether they have been spoken by the same individual.\n"
    "You can attempt this exercise using your own voice."
)
article = (
    "<p style='text-align: center'>"
    "<a href='https://huggingface.co/nvidia/speakerverification_en_titanet_large' target='_blank'>🎙️ Learn more about TitaNet model</a> | "
    "<a href='https://arxiv.org/pdf/2110.04410.pdf' target='_blank'>📚 TitaNet paper</a> | "
    "<a href='https://github.com/NVIDIA/NeMo' target='_blank'>🧑‍💻 Repository</a>"
    "</p>"
)
examples = [
    ["data/id10270_5r0dWxy17C8-00001.wav", "data/id10270_5r0dWxy17C8-00002.wav"],
    ["data/id10271_1gtz-CUIygI-00001.wav", "data/id10271_1gtz-CUIygI-00002.wav"],
    ["data/id10270_5r0dWxy17C8-00001.wav", "data/id10271_1gtz-CUIygI-00001.wav"],
    ["data/id10270_5r0dWxy17C8-00002.wav", "data/id10271_1gtz-CUIygI-00002.wav"],
]

# Keep examples only when the data files exist locally. This makes the repo
# runnable even if you did not clone/download the original Space data folder.
examples = examples if all(Path(f).exists() for pair in examples for f in pair) else None

microphone_interface = gr.Interface(
    fn=compare_samples,
    inputs=inputs,
    outputs=gr.HTML(label=""),
    title="Speaker Verification with TitaNet Embeddings",
    description=description,
    article=article,
    flagging_mode="never",
    live=False,
    examples=examples,
)

upload_interface = gr.Interface(
    fn=compare_samples,
    inputs=upload_inputs,
    outputs=gr.HTML(label=""),
    title="Speaker Verification with TitaNet Embeddings",
    description=description,
    article=article,
    flagging_mode="never",
    live=False,
    examples=examples,
)

demo = gr.TabbedInterface([microphone_interface, upload_interface], ["Microphone", "Upload File"])

demo.queue(max_size=5, default_concurrency_limit=4)

if __name__ == "__main__":
    # Defaults work for local Windows browser + WSL2 backend.
    # Override when needed, e.g. GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python app.py
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)