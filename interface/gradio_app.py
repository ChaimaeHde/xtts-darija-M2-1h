"""Interface Gradio pour TTS Darija M2"""

import os
import gradio as gr
import soundfile as sf
import numpy as np
from inference.generate import load_model_from_hf, synthesize

print("Loading model from Hugging Face Hub...")
model, config = load_model_from_hf("chaimaehde/xtts-darija-M2-1h")


def generate_darija_M2_tts(text, ref_audio):
    if not text or not text.strip():
        return None, "❌ Texte vide"

    if ref_audio is None:
        return None, "❌ Pas d'audio de référence"

    try:
        sr, audio_data = ref_audio

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0

        sf.write("ref_input.wav", audio_data, sr)

        synthesize(
            model=model,
            config=config,
            text=text,
            speaker_wav="ref_input.wav",
            output_path="output_M2.wav",
            language="ar",
        )

        return "output_M2.wav", "✅ Audio généré"

    except Exception as e:
        return None, f"❌ Erreur : {str(e)}"


with gr.Blocks(title="TTS Darija M2") as demo:
    gr.Markdown("# 🎙️ TTS Darija Marocain — Locuteur M2")
    gr.Markdown("XTTS-v2 finetuné sur 60 min du locuteur M2 (DODa)")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Texte en Darija",
                placeholder="مرحبا، كيف داير؟",
                lines=3,
            )

            ref_audio_input = gr.Audio(
                label="Audio de référence (voix à cloner)",
                type="numpy",
                sources=["upload", "microphone"],
            )

            btn = gr.Button("🔊 Générer", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Voix générée", type="filepath")
            status_output = gr.Textbox(label="Statut")

    btn.click(
        fn=generate_darija_M2_tts,
        inputs=[text_input, ref_audio_input],
        outputs=[audio_output, status_output],
    )

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
