"""Pipeline de préparation des données pour XTTS finetuning"""

import os
import re
import subprocess
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def normalize_arabic(text):
    """Nettoie le texte arabe (tatweel, espaces multiples)."""
    if pd.isna(text) or text == "":
        return ""
    text = str(text).strip()
    text = re.sub(r"[\u0640]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def convert_wav_to_22050(wav_dir):
    """Convertit tous les WAV vers 22050 Hz mono 16-bit (requis XTTS-v2)."""
    wavs = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    if not wavs:
        return
    info = sf.info(os.path.join(wav_dir, wavs[0]))
    if info.samplerate == 22050 and info.channels == 1:
        print("Already at 22050 Hz mono")
        return

    print(f"Converting {len(wavs)} files to 22050 Hz mono...")
    errors = []
    for fname in tqdm(wavs):
        src = os.path.join(wav_dir, fname)
        tmp = src.replace(".wav", "_tmp.wav")
        ret = subprocess.run([
            "ffmpeg", "-y", "-i", src,
            "-ar", "22050", "-ac", "1", "-sample_fmt", "s16", tmp
        ], capture_output=True)
        if ret.returncode == 0:
            os.replace(tmp, src)
        else:
            errors.append(fname)
            if os.path.exists(tmp):
                os.remove(tmp)
    print(f"Conversion done. Errors: {len(errors)}")


def create_train_csv(data_path, metadata_csv="metadata.csv", output_csv="train.csv"):
    """Convertit metadata.csv (virgule) -> train.csv (LJSpeech format)."""
    df = pd.read_csv(os.path.join(data_path, metadata_csv))
    filename_col = df.columns[0]
    text_col = df.columns[1]

    df["text_norm"] = df[text_col].apply(normalize_arabic)
    df = df[
        (df["text_norm"].str.len() >= 3) &
        (df["text_norm"].str.len() <= 250)
    ].copy()

    df["file_id"] = df[filename_col].astype(str)
    df["file_id"] = df["file_id"].str.replace("wavs/", "", regex=False)
    df["file_id"] = df["file_id"].str.replace(".wav", "", regex=False)

    wav_dir = os.path.join(data_path, "wavs")
    missing = [fid for fid in df["file_id"]
               if not os.path.exists(os.path.join(wav_dir, f"{fid}.wav"))]
    df = df[~df["file_id"].isin(missing)]

    train_csv = os.path.join(data_path, output_csv)
    with open(train_csv, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            line = row["file_id"] + "|" + row["text_norm"] + "|" + row["text_norm"] + "\n"
            f.write(line)

    print(f"{output_csv} created with {len(df)} samples")
    return train_csv


if __name__ == "__main__":
    from config.default_config import DATA_PATH
    convert_wav_to_22050(os.path.join(DATA_PATH, "wavs"))
    create_train_csv(DATA_PATH)
