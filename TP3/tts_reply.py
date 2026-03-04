import os
import time

import numpy as np
import torch
from transformers import pipeline

import soundfile as sf  # <-- au lieu de torchaudio.save


def main():
    os.makedirs("TP3/outputs", exist_ok=True)

    text = (
        "Thanks for calling. I am sorry your order arrived damaged. "
        "I can offer a replacement or a refund. "
        "Please confirm your preferred option."
    )

    # Modèle TTS léger (anglais)
    tts_model_id = "facebook/mms-tts-eng"

    device = 0 if torch.cuda.is_available() else -1
    tts = pipeline(
        task="text-to-speech",
        model=tts_model_id,
        device=device
    )

    print("\n--- TTS GENERATION ---")

    t0 = time.time()
    out = tts(text)
    t1 = time.time()

    audio = np.asarray(out["audio"], dtype=np.float32)
    sr = int(out["sampling_rate"])

    elapsed_s = t1 - t0

    # audio peut être [T] ou [1, T] ou [T, 1]
    if audio.ndim == 1:
        audio_1d = audio
        audio_dur_s = audio.shape[0] / sr
    elif audio.ndim == 2:
        # [1, T]
        if audio.shape[0] == 1:
            audio_1d = audio[0]
            audio_dur_s = audio.shape[1] / sr
        # [T, 1]
        elif audio.shape[1] == 1:
            audio_1d = audio[:, 0]
            audio_dur_s = audio.shape[0] / sr
        else:
            # cas multi-canaux -> on prend canal 0
            audio_1d = audio[0]
            audio_dur_s = audio.shape[1] / sr
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    rtf = elapsed_s / max(audio_dur_s, 1e-9)

    out_wav = "TP3/outputs/tts_reply_call_01.wav"

    # Écriture WAV sans torchaudio/torchcodec
    sf.write(out_wav, audio_1d, sr)

    print("tts_model_id:", tts_model_id)
    print("device:", "cuda" if device == 0 else "cpu")
    print("audio_dur_s:", round(audio_dur_s, 2))
    print("elapsed_s:", round(elapsed_s, 2))
    print("rtf:", round(rtf, 3))
    print("saved:", out_wav)


if __name__ == "__main__":
    main()
