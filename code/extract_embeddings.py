"""
Extract MedGemma Text+Acoustic Narrative Embeddings
====================================================
Processes audio clips through the full pipeline:
  1. Load pre-existing transcripts (from JSON cache)
  2. Load audio and compute 14 acoustic metrics (librosa)
  3. Build an acoustic narrative prompt embedding the transcript
     and acoustic descriptors as structured natural language
  4. Pass through MedGemma 4B-IT (4-bit quantised, text-only)
  5. Mean-pool the last hidden state -> 2560-dim embedding per clip
  6. Save all clip embeddings as a single NPZ file

The resulting NPZ is the input to train_text_acoustic_narrative_pre_symptoms.py
and evaluate_text_acoustic_narrative_pre_symptoms.py.

Transcripts were pre-extracted using Pyannote precision-2 with
parakeet-tdt-0.6b-v3 transcription and are provided as a JSON cache.
This script does not call any transcription API.

Requirements:
  pip install torch transformers bitsandbytes accelerate librosa numpy pandas

Usage:
  python code/extract_embeddings.py

  MedGemma requires a Hugging Face token with access to
  google/medgemma-4b-it. Set HF_TOKEN in environment or .env.
"""

import os
import re
import gc
import json
import numpy as np
import pandas as pd
import librosa
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "dataset" / "manifest.csv"
AUDIO_DIR = ROOT / "dataset" / "audio"

# Pre-extracted transcripts (Pyannote precision-2 + parakeet-tdt-0.6b-v3)
TRANSCRIPT_CACHE = ROOT / "reproducibility" / "transcripts" / "manifest_transcripts.json"

OUTPUT_NPZ = (
    ROOT / "reproducibility" / "multimodal"
    / "outputs_multimodal_agentic_manifest_text_acoustic_narrative"
    / "multimodal_embeddings_agentic_manifest_text_acoustic_narrative.npz"
)

SAMPLE_RATE = 16000
MAX_TOKEN_LENGTH = 1024


# ── Acoustic metrics (14 descriptors) ──────────────────────────────

def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def compute_acoustic_metrics(y: np.ndarray, sr: int, transcript: str) -> dict | None:
    """
    Compute 14 waveform-derived acoustic descriptors from a mono audio signal.
    Returns None if the audio is too short (< 0.5s).
    """
    if y.size < int(0.5 * sr):
        return None

    duration_s = float(y.size / sr)
    wc = _word_count(transcript)
    speech_rate_wps = float(wc / max(duration_s, 1e-6))

    # Pause ratio via voice activity detection
    intervals = librosa.effects.split(y, top_db=30)
    voiced_samples = int(np.sum([max(0, b - a) for a, b in intervals])) if len(intervals) else 0
    voiced_ratio = float(voiced_samples / max(1, y.size))
    pause_ratio = float(np.clip(1.0 - voiced_ratio, 0.0, 1.0))

    # RMS energy variability (prosodic variation)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_std = float(np.std(rms))

    # Spectral energy distribution across frequency bands
    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total_energy = float(np.sum(S)) + 1e-12

    low_e = float(np.sum(S[freqs < 1000]))
    mid_e = float(np.sum(S[(freqs >= 1000) & (freqs < 4000)]))
    high_e = float(np.sum(S[freqs >= 4000]))
    low_pct = float(100.0 * low_e / total_energy)
    mid_pct = float(100.0 * mid_e / total_energy)
    high_pct = float(100.0 * high_e / total_energy)
    low_mid_ratio = float((low_e + mid_e) / max(high_e, 1e-9))

    # Spectral centroid and bandwidth
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr)[0]

    # Tempo proxy
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]) if onset_env.size else 0.0

    # Fundamental frequency (pitch)
    f0_mean, f0_std = 0.0, 0.0
    try:
        f0 = librosa.yin(y, fmin=60, fmax=350, sr=sr, frame_length=1024, hop_length=hop)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 >= 60) & (f0 <= 350)]
        if f0.size > 0:
            f0_mean, f0_std = float(np.mean(f0)), float(np.std(f0))
    except Exception:
        pass

    # Articulatory transition proxy (delta MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop, n_fft=n_fft)
    d_mfcc = librosa.feature.delta(mfcc)
    formant_transition_proxy = float(
        np.mean(np.abs(d_mfcc[1:4])) if d_mfcc.shape[0] >= 4
        else np.mean(np.abs(d_mfcc))
    )

    return {
        "speech_rate_wps": speech_rate_wps,
        "pause_ratio": pause_ratio,
        "rms_std": rms_std,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "low_pct": low_pct,
        "mid_pct": mid_pct,
        "high_pct": high_pct,
        "low_mid_ratio": low_mid_ratio,
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "tempo_bpm": tempo,
        "formant_transition_proxy": formant_transition_proxy,
    }


# ── Acoustic narrative prompt ──────────────────────────────────────

def build_acoustic_narrative_prompt(transcript: str, m: dict) -> str:
    """
    Build the exact text prompt used during training.
    Acoustic metrics are embedded as structured natural language
    alongside the verbatim transcript — no images, text-only input.
    """
    acoustic_block = (
        "Acoustic profile (waveform-derived): "
        f"speech rate {m['speech_rate_wps']:.2f} words/s; "
        f"pauses {100.0*m['pause_ratio']:.1f}% of total duration; "
        f"pitch mean {m['f0_mean']:.1f} Hz with variability {m['f0_std']:.1f} Hz; "
        f"spectral energy distribution 0-1kHz {m['low_pct']:.1f}%, "
        f"1-4kHz {m['mid_pct']:.1f}%, >4kHz {m['high_pct']:.1f}% "
        f"(low+mid vs high ratio {m['low_mid_ratio']:.2f}); "
        f"centroid {m['centroid_mean']:.0f} Hz (std {m['centroid_std']:.0f}); "
        f"bandwidth {m['bandwidth_mean']:.0f} Hz; "
        f"tempo proxy {m['tempo_bpm']:.1f} BPM; "
        f"prosodic variability RMS std {m['rms_std']:.4f}; "
        f"articulatory transition proxy {m['formant_transition_proxy']:.4f}. "
        "When reasoning, prioritize low and mid frequency bands "
        "(0-1kHz and 1-4kHz) over the top band."
    )
    return (
        "The following is a verbatim transcript of a person's speech. "
        "Assess the linguistic and cognitive characteristics of this speech.\n\n"
        f"Transcript: {transcript}\n\n"
        f"{acoustic_block}"
    )


# ── MedGemma embedding extraction ─────────────────────────────────

def load_medgemma(device="cuda"):
    """Load MedGemma 4B-IT with 4-bit quantisation."""
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    model_name = "google/medgemma-4b-it"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def extract_embedding(model, processor, prompt: str) -> np.ndarray:
    """
    Run a single text prompt through MedGemma and return the
    mean-pooled last hidden state as a 2560-dim float32 vector.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.squeeze(0).float().cpu().numpy()


# ── Main extraction loop ──────────────────────────────────────────

def load_transcript_cache() -> dict:
    """Load pre-extracted transcripts keyed by clip stem (filename without extension)."""
    if not TRANSCRIPT_CACHE.exists():
        print(f"Transcript cache not found: {TRANSCRIPT_CACHE}")
        print("Transcripts were pre-extracted using Pyannote precision-2.")
        return {}

    with open(TRANSCRIPT_CACHE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    by_stem = {}
    for _, v in payload.items():
        if not isinstance(v, dict):
            continue
        anon_filename = str(v.get("anon_filename", "")).strip()
        txt = str(v.get("text", "")).strip()
        if not anon_filename or len(txt) < 5:
            continue
        stem = os.path.splitext(anon_filename)[0]
        if stem not in by_stem or len(txt) > len(by_stem[stem]):
            by_stem[stem] = txt

    print(f"Loaded transcript cache: {len(by_stem)} clips")
    return by_stem


def main():
    import sys

    print("=" * 60)
    print("MedGemma Text+Acoustic Narrative Embedding Extraction")
    print("=" * 60)

    # Load manifest
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest: {len(df)} clips, {df['anon_speaker_id'].nunique()} speakers")

    # Load pre-extracted transcripts
    transcript_map = load_transcript_cache()
    if not transcript_map:
        print("ERROR: No transcripts available. Cannot proceed.")
        sys.exit(1)

    # Build clip list
    clip_stems = []
    audio_paths = []
    transcripts = []

    for _, row in df.iterrows():
        stem = os.path.splitext(str(row["anon_filename"]))[0]
        transcript = transcript_map.get(stem, "")
        if not transcript:
            continue

        audio_path = AUDIO_DIR / row["original_path"].replace("dataset/audio/", "")
        if not audio_path.exists():
            audio_path = AUDIO_DIR / row.get("group", "") / str(row["anon_speaker_id"]) / row["anon_filename"]

        clip_stems.append(stem)
        audio_paths.append(str(audio_path))
        transcripts.append(transcript)

    print(f"Clips with transcripts: {len(clip_stems)}")

    # Load MedGemma
    print("\nLoading MedGemma 4B-IT (4-bit quantised)...")
    model, processor = load_medgemma()
    print(f"Model loaded on {model.device}")

    # Extract embeddings
    embeddings = {}
    n_total = len(clip_stems)
    n_skipped = 0

    for i, (stem, audio_path, transcript) in enumerate(zip(clip_stems, audio_paths, transcripts)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"\nProcessing clip {i+1}/{n_total}: {stem}")

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

            # Compute acoustic metrics
            metrics = compute_acoustic_metrics(y, sr, transcript)
            if metrics is None:
                print(f"  Skipping {stem}: audio too short")
                n_skipped += 1
                continue

            # Build prompt and extract embedding
            prompt = build_acoustic_narrative_prompt(transcript, metrics)
            emb = extract_embedding(model, processor, prompt)
            embeddings[stem] = emb

        except Exception as e:
            print(f"  Error on {stem}: {e}")
            n_skipped += 1
            continue

    # Clean up GPU
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save as NPZ (matching the format expected by train/evaluate scripts)
    print(f"\n{'=' * 60}")
    print(f"Extracted: {len(embeddings)} clips ({n_skipped} skipped)")
    print(f"Embedding dim: 2560")

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        conditions={"text_plus_acoustic_narrative": embeddings},
    )
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"File size: {OUTPUT_NPZ.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
