"""Evaluation WER/CER avec Whisper + MOS humain"""

import os
import numpy as np
from faster_whisper import WhisperModel
from jiwer import wer, cer


def evaluate_wer_cer(audio_files, original_texts, language="ar"):
    """Calcule WER et CER avec Whisper large-v2."""
    print("Loading Whisper large-v2...")
    asr_model = WhisperModel("large-v2", device="cuda", compute_type="float16")

    results = []
    for i, (audio_path, original) in enumerate(zip(audio_files, original_texts)):
        segments, _ = asr_model.transcribe(audio_path, language=language)
        recognized = " ".join([s.text for s in segments]).strip()
        score_wer = wer(original, recognized)
        score_cer = cer(original, recognized)
        results.append({
            "audio": os.path.basename(audio_path),
            "original": original, "recognized": recognized,
            "WER": score_wer, "CER": score_cer,
        })
        print("--- Audio " + str(i+1) + " | WER: " + str(round(score_wer*100, 2)) + "% | CER: " + str(round(score_cer*100, 2)) + "%")

    avg_wer = sum(r["WER"] for r in results) / len(results)
    avg_cer = sum(r["CER"] for r in results) / len(results)
    print("Average WER: " + str(round(avg_wer*100, 2)) + "%")
    print("Average CER: " + str(round(avg_cer*100, 2)) + "%")
    return results, avg_wer, avg_cer


def compute_mos(scores_dict):
    """Calcule MOS depuis scores humains (dict de listes 1-5)."""
    mos_values = []
    for criterion, values in scores_dict.items():
        mos = np.mean(values)
        mos_values.append(mos)
        print(criterion + " : " + str(round(mos, 2)) + " / 5")
    mos_global = np.mean(mos_values)
    print("MOS Global : " + str(round(mos_global, 2)) + " / 5")
    return mos_global
