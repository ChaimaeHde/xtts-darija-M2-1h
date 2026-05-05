"""Script de finetuning XTTS-v2 sur darija marocaine (locuteur M2, 60 min)"""

import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.utils.manage import ModelManager

from config.default_config import (
    DATA_PATH, OUT_PATH, MODEL_DIR,
    RUN_NAME, PROJECT_NAME,
    BATCH_SIZE, GRAD_ACUMM_STEPS, NUM_EPOCHS, LR,
    SAVE_STEP, PRINT_STEP,
    SAMPLE_RATE, OUTPUT_SAMPLE_RATE, LANGUAGE,
)


def download_base_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = [
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
        "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    ]
    for url in files:
        fname = os.path.basename(url)
        dest = os.path.join(MODEL_DIR, fname)
        if not os.path.isfile(dest):
            ModelManager._download_model_files([url], MODEL_DIR, progress_bar=True)


def main():
    download_base_model()

    TOKENIZER_FILE  = os.path.join(MODEL_DIR, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(MODEL_DIR, "model.pth")
    DVAE_CHECKPOINT = os.path.join(MODEL_DIR, "dvae.pth")
    MEL_NORM_FILE   = os.path.join(MODEL_DIR, "mel_stats.pth")

    dataset_config = BaseDatasetConfig(
        dataset_name="doda_M2", path=DATA_PATH,
        meta_file_train="train.csv", meta_file_val="",
        ignored_speakers=None, formatter="ljspeech", language=LANGUAGE,
    )
    train_samples, eval_samples = load_tts_samples(
        [dataset_config], eval_split=True,
        eval_split_max_size=256, eval_split_size=0.1,
    )

    audio_config = XttsAudioConfig(
        sample_rate=SAMPLE_RATE, dvae_sample_rate=SAMPLE_RATE,
        output_sample_rate=OUTPUT_SAMPLE_RATE,
    )
    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        debug_loading_failures=False, max_wav_length=255995, max_text_length=200,
        mel_norm_file=MEL_NORM_FILE, dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT, tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
    )
    config = GPTTrainerConfig(
        output_path=OUT_PATH, model_args=model_args,
        run_name=RUN_NAME, project_name=PROJECT_NAME,
        run_description="XTTS-v2 finetuning M2 60min " + str(NUM_EPOCHS) + " epochs",
        dashboard_logger="tensorboard", logger_uri=None,
        audio=audio_config, epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE, batch_group_size=48, eval_batch_size=BATCH_SIZE,
        num_loader_workers=2, eval_split_max_size=256,
        print_step=PRINT_STEP, plot_step=100, log_model_step=100,
        save_step=SAVE_STEP, save_n_checkpoints=1, save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LR, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[], datasets=[dataset_config],
    )

    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False,
                    start_with_eval=False, grad_accum_steps=GRAD_ACUMM_STEPS),
        config, output_path=OUT_PATH + "run/training/",
        model=model, train_samples=train_samples, eval_samples=eval_samples,
    )
    trainer.fit()
    print("Training done. Model in " + OUT_PATH)


if __name__ == "__main__":
    main()
