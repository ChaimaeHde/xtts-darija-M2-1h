"""Configuration centralisée pour XTTS Darija M2"""

# === Paths ===
DATA_PATH = "/content/doda_M2_60min/"
OUT_PATH  = "/content/M2_outputs/"
MODEL_DIR = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"

# === Run identification ===
RUN_NAME     = "xtts_darija_M2"
PROJECT_NAME = "XTTS_M2_finetuning"

# === Hyperparameters ===
BATCH_SIZE       = 2
GRAD_ACUMM_STEPS = 126
NUM_EPOCHS       = 10
LR               = 5e-6
SAVE_STEP        = 1000
PRINT_STEP       = 50

# === Audio ===
SAMPLE_RATE        = 22050
OUTPUT_SAMPLE_RATE = 24000

# === Dataset ===
DATASET_NAME = "atlasia/DODa-audio-dataset"
TEXT_COL     = "darija_Arab_new"
LANGUAGE     = "ar"

# === HuggingFace ===
HF_USERNAME   = "chaimaehde"
HF_MODEL_REPO = "chaimaehde/xtts-darija-M2-1h"
HF_SPACE_REPO = "chaimaehde/tts-darija-M2-1h"

# === GitHub ===
GITHUB_USERNAME = "chaimaeHde"
GITHUB_REPO     = "xtts-darija-M2-1h"
