import os
import gradio as gr
import soundfile as sf
import numpy as np
import torch

from huggingface_hub import hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

os.environ["COQUI_TOS_AGREED"] = "1"

print("Loading model from Hugging Face...")

MODEL_REPO = "chaimaehde/xtts-darija-M2-1h"

model_path  = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pth")
config_path = hf_hub_download(repo_id=MODEL_REPO, filename="config.json")
vocab_path  = hf_hub_download(repo_id=MODEL_REPO, filename="vocab.json")

config = XttsConfig()
config.load_json(config_path)

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=model_path, vocab_path=vocab_path, eval=True)

if torch.cuda.is_available():
    model.cuda()

print("Model loaded")

def generate(text, ref_audio):
    if not text or not text.strip():
        return None, "Texte vide"
    if ref_audio is None:
        return None, "Audio requis"

    try:
        sr, audio_data = ref_audio

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0

        sf.write("ref.wav", audio_data, sr)

        outputs = model.synthesize(
            text=text,
            config=config,
            speaker_wav="ref.wav",
            language="ar"
        )

        sf.write("out.wav", outputs["wav"], 24000)

        return "out.wav", "OK"

    except Exception as e:
        return None, str(e)


with gr.Blocks() as demo:
    gr.Markdown("# TTS Darija M2")

    text = gr.Textbox(label="Texte")
    audio = gr.Audio(type="numpy")

    btn = gr.Button("Generate")

    out_audio = gr.Audio(type="filepath")
    status = gr.Textbox()

    btn.click(generate, [text, audio], [out_audio, status])

demo.launch(server_name="0.0.0.0", server_port=7860)
