import json
import subprocess
import time

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def load_wav_mono_16k_ffmpeg(path: str):
    """
    Decode audio using system ffmpeg into mono 16kHz float32 PCM (no torchaudio / no torchcodec).
    Returns: wav (torch.Tensor [T]), sr (int)
    """
    sr = 16000

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    audio = np.frombuffer(raw, dtype=np.float32)  # [T]
    wav = torch.from_numpy(audio)  # [T]
    return wav, sr


def main():
    wav_path = "TP3/outputs/tts_reply_call_01.wav"
    model_id = "openai/whisper-small"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("\n--- ASR TTS CHECK (manual, ffmpeg decode) ---")
    print("wav_path:", wav_path)
    print("model_id:", model_id)
    print("device:", device)

    # Load wav without torchaudio
    wav, sr = load_wav_mono_16k_ffmpeg(wav_path)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    # Whisper features
    inputs = processor(wav.numpy(), sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(device, dtype=dtype)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language="en",
            task="transcribe",
            max_new_tokens=128,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    t1 = time.time()

    print("elapsed_s:", round(t1 - t0, 2))
    print("text:", text)


if __name__ == "__main__":
    main()
