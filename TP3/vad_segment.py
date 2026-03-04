# TP3/vad_segment.py
import os
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torchaudio

# silero-vad (VAD prêt à l’emploi)
from silero_vad import get_speech_timestamps


@dataclass
class Segment:
    start_s: float
    end_s: float


def load_wav_mono_16k(path: str) -> Tuple[torch.Tensor, int]:
    """
    Charge un WAV en mono 16kHz en évitant torchcodec/ffmpeg.
    - essaie soundfile (recommandé)
    - fallback wave (stdlib)
    Retour: (wav[T] float32, sr=16000)
    """
    # 1) Try soundfile
    try:
        import soundfile as sf  # pip install soundfile
        audio, sr = sf.read(path, always_2d=True)  # [T, C]
        audio = audio.astype(np.float32)
        audio_mono = audio.mean(axis=1)  # [T]
        wav = torch.from_numpy(audio_mono)  # float32 [T]
    except Exception as e_sf:
        # 2) Fallback stdlib wave (PCM16) — marche pour la plupart des WAV simples
        try:
            import wave
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sampwidth != 2:
                raise RuntimeError(f"Unsupported sample width: {sampwidth} bytes (expected 2 for PCM16)")

            audio_i16 = np.frombuffer(raw, dtype=np.int16)  # interleaved
            audio_i16 = audio_i16.reshape(-1, n_channels)   # [T, C]
            audio_f32 = (audio_i16.astype(np.float32) / 32768.0)
            audio_mono = audio_f32.mean(axis=1)
            wav = torch.from_numpy(audio_mono)
        except Exception as e_wave:
            raise RuntimeError(
                "Impossible de charger l'audio. Installe soundfile (recommandé):\n"
                "  python -m pip install --user soundfile\n"
                f"Erreur soundfile: {e_sf}\nErreur wave: {e_wave}"
            )

    # Resample -> 16kHz si besoin
    if sr != 16000:
        wav = wav.unsqueeze(0)  # [1, T]
        wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        sr = 16000

    return wav, sr


def main():

    import os, time
    print("RUNNING FILE:", os.path.abspath(__file__))
    print("NOW:", time.ctime())
    print("--- VAD SEGMENTATION (silero-vad) ---")
    in_path = "TP3/data/call_01.wav"
    out_path = "TP3/outputs/vad_segments_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    wav, sr = load_wav_mono_16k(in_path)  # wav: [T] float32
    duration_s = wav.numel() / sr

    # Chargement du modèle Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    model.to("cpu").eval()

    # VAD -> timestamps en indices (samples)
    speech_ts = get_speech_timestamps(
        wav.to(torch.float32),   # 1D float32
        model,
        sampling_rate=16000
    )

    # Convertir en segments en secondes
    segments: List[Segment] = []
    for seg in speech_ts:
        start_s = seg["start"] / sr
        end_s = seg["end"] / sr
        segments.append(Segment(start_s=float(start_s), end_s=float(end_s)))

    # Filtrage simple : supprimer segments trop courts
    min_dur_s = 1.0
    segments = [s for s in segments if (s.end_s - s.start_s) >= min_dur_s]

    # Stats
    total_speech_s = sum((s.end_s - s.start_s) for s in segments)
    speech_ratio = total_speech_s / max(duration_s, 1e-9)

    print("duration_s:", round(duration_s, 2))
    print("num_segments:", len(segments))
    print("total_speech_s:", round(total_speech_s, 2))
    print("speech_ratio:", round(speech_ratio, 3))

    payload = {
        "audio_path": in_path,
        "sample_rate": sr,
        "duration_s": float(duration_s),
        "min_segment_s": float(min_dur_s),
        "segments": [{"start_s": s.start_s, "end_s": s.end_s} for s in segments],
        "stats": {
            "num_segments": int(len(segments)),
            "total_speech_s": float(total_speech_s),
            "speech_ratio": float(speech_ratio),
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("saved:", out_path)


if __name__ == "__main__":
    main()
