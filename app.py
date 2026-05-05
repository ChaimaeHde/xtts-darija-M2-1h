"""
Point d'entrée principal — Interface Gradio TTS Darija M2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interface.gradio_app import demo

if __name__ == "__main__":
    print(" Lancement de l'interface TTS Darija M2...")
    print(" Le modèle sera téléchargé automatiquement depuis Hugging Face")
    print(" Modèle : chaimaehde/xtts-darija-M2-1h")
    print("🌍 Lien public Gradio activé avec share=True")

    demo.launch(
        share=True,
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
