import os
import json
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def load_wav_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    """
    Charge un WAV via soundfile, force mono, et force sr=16k si nécessaire.
    Retour: wav float32 1D, sr
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)

    # [T, C] -> mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != 16000:
        # resample avec scipy (fallback)
        import scipy.signal
        target_sr = 16000
        num = int(len(wav) * target_sr / sr)
        wav = scipy.signal.resample(wav, num).astype(np.float32)
        sr = target_sr

    return wav, sr


def choose_model_id() -> str:
    # GPU -> small (bon compromis), CPU -> tiny/base
    if torch.cuda.is_available():
        return "openai/whisper-small"
    return "openai/whisper-tiny"


def main():
    print("--- ASR WHISPER (manual, no pipeline/torchcodec) ---")

    audio_path = "TP3/data/call_01.wav"
    vad_path = "TP3/outputs/vad_segments_call_01.json"
    out_path = "TP3/outputs/asr_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    wav, sr = load_wav_mono_16k(audio_path)
    audio_duration_s = len(wav) / sr

    with open(vad_path, "r", encoding="utf-8") as f:
        vad_payload = json.load(f)
    segments = vad_payload["segments"]  # list of {start_s, end_s}

    model_id = choose_model_id()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("model_id:", model_id)
    print("device:", device)
    print("num_segments:", len(segments))

    # Charger processor + model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    model.eval()

    # Warmup léger sur un mini bout
    with torch.no_grad():
        dummy = wav[: min(len(wav), sr)]  # ~1 seconde
        inputs = processor(dummy, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model.generate(**inputs, max_new_tokens=4)

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    # Options de décodage
    gen_kwargs = {
        "language": "english",
        "task": "transcribe",
        "max_new_tokens": 128,
    }

    for i, seg in enumerate(segments):
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])

        start = int(start_s * sr)
        end = int(end_s * sr)
        seg_wav = wav[start:end]

        # Processor -> features
        inputs = processor(seg_wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        results.append({
            "segment_id": i,
            "start_s": start_s,
            "end_s": end_s,
            "text": text
        })

        print(f"[{i+1}/{len(segments)}] {start_s:.2f}-{end_s:.2f}s -> {text[:70]}")

    t1 = time.time()
    elapsed_s = t1 - t0
    rtf = elapsed_s / max(audio_duration_s, 1e-9)

    full_text = " ".join([r["text"] for r in results]).strip()

    payload = {
        "audio_path": audio_path,
        "model_id": model_id,
        "device": device,
        "audio_duration_s": audio_duration_s,
        "elapsed_s": elapsed_s,
        "rtf": rtf,
        "segments": results,
        "full_text": full_text
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n--- SUMMARY ---")
    print("model_id:", model_id)
    print("device:", device)
    print("audio_duration_s:", round(audio_duration_s, 2))
    print("elapsed_s:", round(elapsed_s, 2))
    print("rtf:", round(rtf, 3))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
