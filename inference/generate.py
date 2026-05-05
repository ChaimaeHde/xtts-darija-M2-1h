"""Génération audio avec le modèle XTTS-v2 finetuné M2"""

import os
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_model_from_hf(repo_id="chaimaehde/xtts-darija-M2-1h"):
    """Charge le modèle finetuné depuis HuggingFace Hub."""
    print("Downloading model from " + repo_id + "...")
    model_path  = hf_hub_download(repo_id=repo_id, filename="best_model.pth")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    vocab_path  = hf_hub_download(repo_id=repo_id, filename="vocab.json")

    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=model_path,
        vocab_path=vocab_path, eval=True,
    )
    if torch.cuda.is_available():
        model.cuda()
    print("Model loaded")
    return model, config


def synthesize(model, config, text, speaker_wav, output_path="output.wav", language="ar"):
    """Génère un audio avec voice cloning."""
    outputs = model.synthesize(
        text=text, config=config,
        speaker_wav=speaker_wav, language=language,
    )
    sf.write(output_path, outputs["wav"], 24000)
    print("Audio saved: " + output_path)
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref",  required=True)
    parser.add_argument("--out",  default="output.wav")
    args = parser.parse_args()

    model, config = load_model_from_hf()
    synthesize(model, config, args.text, args.ref, args.out)
