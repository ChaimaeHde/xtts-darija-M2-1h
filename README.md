#  XTTS-v2 Darija Marocain — Locuteur M2 (1h)

Finetuning du modèle [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) sur 60 minutes du locuteur M2 du dataset [DODa](https://huggingface.co/datasets/atlasia/DODa-audio-dataset) pour la synthèse vocale en darija marocaine.

##  Résultats

| Métrique | Valeur |
|----------|--------|
| Données d'entraînement | 60 min (1102 samples M2) |
| Epochs | 10 |
| Batch effectif | 252 (2 × 126 grad_accum) |
| Learning rate | 5e-6 |
| Loss finale (mel_ce) | 3.59 |
| Temps de training | ~1h05 sur GPU T4 |

##  Ressources

-  **Modèle** : [chaimaehde/xtts-darija-M2-1h](https://huggingface.co/chaimaehde/xtts-darija-M2-1h)
-  **Dataset source** : [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)

##  Structure
```
xtts-darija-M2-1h/
├── config/              # Configuration centralisée
├── data/                # Pipeline de préparation
├── training/            # Script de finetuning
├── inference/           # Génération audio
├── evaluation/          # WER, CER, MOS
├── interface/           # Interface Gradio
└── notebooks/           # Demo Colab
```

### Tester sur colab directement 
```python
!pip install -q coqui-tts gradio huggingface_hub soundfile
!git clone https://github.com/ChaimaeHde/xtts-darija-M2-1h.git
%cd xtts-darija-M2-1h
!python app.py
```




### Installation
```bash
pip install -r requirements.txt
```

### Inférence rapide
```python
from inference.generate import load_model_from_hf, synthesize

model, config = load_model_from_hf()
synthesize(
    model, config,
    text="مرحبا، كيف دايرين؟",
    speaker_wav="reference.wav",
    output_path="output.wav",
)
```

##  Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| `batch_size` | 2 |
| `grad_accum_steps` | 126 |
| `epochs` | 10 |
| `lr` | 5e-6 |
| `optimizer` | AdamW |
| `sample_rate` | 22050 Hz |

##  Licence

Coqui Public Model License (CPML), cohérent avec XTTS-v2 original.

## 👤 Auteur

Chaimae Haddouche - Loubna Haouach  —  2026
