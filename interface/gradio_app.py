"""Interface Gradio pour TTS Darija M2"""

import os
import gradio as gr
import soundfile as sf
import numpy as np
from inference.generate import load_model_from_hf, synthesize


print("Loading model from HuggingFace Hub...")
model, config = load_model_from_hf("chaimaehde/xtts-darija-M2-1h")


def generate(text, ref_audio):
    if not text or not text.strip():
        return None, "Texte vide"
    if ref_audio is None:
        return None, "Pas d'audio de reference"
    try:
        sr, audio_data = ref_audio
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
        sf.write("ref_input.wav", audio_data, sr)
        synthesize(model, config, text, "ref_input.wav", "output.wav", "ar")
        return "output.wav", "Audio genere"
    except Exception as e:
        return None, "Erreur: " + str(e)


with gr.Blocks(title="TTS Darija M2") as demo:
    gr.Markdown("# TTS Darija Marocain - Locuteur M2")
    gr.Markdown("XTTS-v2 finetune sur 60 min du locuteur M2 (DODa)")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Texte en Darija", placeholder="مرحبا، كيف داير؟", lines=3)
            ref_audio  = gr.Audio(label="Audio de reference", type="numpy", sources=["upload", "microphone"])
            btn        = gr.Button("Generer", variant="primary")
        with gr.Column():
            audio_out  = gr.Audio(label="Voix generee", type="filepath")
            status     = gr.Textbox(label="Statut")

    btn.click(fn=generate, inputs=[text_input, ref_audio], outputs=[audio_out, status])

    gr.Examples(
        examples=[
            ["مرحبا، كيف داير؟ واش كلشي مزيان؟"],
            ["واش نتا مزيان؟ شنو كاين الجديد؟"],
            ["الجو مزيان بزاف اليوم، خرجنا نتفرجو"],
            ["ما فهمتش"],
            ["الله يحفظك، بارك الله فيك"],
        ],
        inputs=[text_input],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
