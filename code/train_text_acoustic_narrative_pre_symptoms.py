"""
Train Text+Acoustic Narrative Model (Pre-Symptoms)
========================================================
Trains StandardScaler + LogisticRegression on speaker-level
[mean, std] = 5120 dims. Saves model PKL + metadata JSON.

Usage:
    python code/train_text_acoustic_narrative_pre_symptoms.py
"""

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "dataset" / "manifest.csv"
TAN_NPZ = (
    ROOT
    / "reproducibility"
    / "multimodal"
    / "outputs_multimodal_agentic_manifest_text_acoustic_narrative"
    / "multimodal_embeddings_agentic_manifest_text_acoustic_narrative.npz"
)

OUT_DIR = ROOT / "reproducibility" / "best_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "text_acoustic_narrative_pre_symptoms.pkl"
META_PATH = OUT_DIR / "text_acoustic_narrative_pre_symptoms_metadata.json"

RANDOM_STATE = 42


def is_after_symptoms_clip(clip_name: str) -> bool:
    return "_after_symptoms_" in str(clip_name).lower()


def load_manifest_map():
    df = pd.read_csv(MANIFEST_PATH)
    mapping = {}
    for _, row in df.iterrows():
        stem = os.path.splitext(str(row["anon_filename"]))[0]
        mapping[stem] = {
            "speaker_id": str(row["anon_speaker_id"]),
            "label": int(row["label"]),
        }
    return mapping


def load_text_acoustic_clip_embeddings():
    pack = np.load(TAN_NPZ, allow_pickle=True)["conditions"].item()
    return {k: np.asarray(v, dtype=np.float32) for k, v in pack["text_plus_acoustic_narrative"].items()}


def aggregate_speaker(X, y, g):
    by_spk = defaultdict(lambda: {"x": [], "y": None})
    for i, sid in enumerate(g):
        by_spk[sid]["x"].append(X[i])
        by_spk[sid]["y"] = int(y[i])

    Xs, ys, gs = [], [], []
    for sid, row in sorted(by_spk.items()):
        arr = np.stack(row["x"], axis=0)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
        Xs.append(np.concatenate([mean, std]).astype(np.float32))
        ys.append(int(row["y"]))
        gs.append(sid)
    return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(gs)


def main():
    manifest_map = load_manifest_map()
    tan_clip = load_text_acoustic_clip_embeddings()

    # Filter
    all_clips = sorted(tan_clip.keys())
    filtered = [c for c in all_clips if not is_after_symptoms_clip(c)]
    n_excluded = len(all_clips) - len(filtered)

    # Build arrays
    X_list, y_list, g_list = [], [], []
    for clip_name in sorted(filtered):
        info = manifest_map.get(clip_name)
        if info is None:
            continue
        X_list.append(tan_clip[clip_name])
        y_list.append(int(info["label"]))
        g_list.append(str(info["speaker_id"]))

    X_clip = np.asarray(X_list, dtype=np.float32)
    y_clip = np.asarray(y_list, dtype=np.int64)
    g_clip = np.asarray(g_list)

    X_spk, y_spk, g_spk = aggregate_speaker(X_clip, y_clip, g_clip)
    n_dementia = int(np.sum(y_spk == 1))
    n_control = int(np.sum(y_spk == 0))

    print(f"Clips: {len(all_clips)} total, {n_excluded} excluded, {len(filtered)} used")
    print(f"Speakers: {len(y_spk)} ({n_dementia} dementia, {n_control} control)")
    print(f"Feature dim: {X_spk.shape[1]} (expected 5120)")

    # Train
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_spk)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        solver="saga",
        penalty="l2",
        random_state=RANDOM_STATE,
    )
    clf.fit(Xs, y_spk)

    p_train = clf.predict_proba(Xs)[:, 1]
    y_hat = (p_train >= 0.5).astype(np.int64)

    # Save model PKL
    package = {
        "scaler": scaler,
        "classifier": clf,
        "feature_spec": {
            "medgemma_clip_dim": 2560,
            "speaker_feature_dim": int(X_spk.shape[1]),
            "aggregation": "[clip_mean(2560), clip_std(2560)]",
            "modalities": ["text_plus_acoustic_narrative"],
            "filter": "pre_symptoms",
            "n_speakers_trained": int(len(y_spk)),
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata JSON (use relative paths for portability)
    meta = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "n_clips_used": int(len(filtered)),
        "n_excluded_after_symptoms": int(n_excluded),
        "n_speakers": int(len(y_spk)),
        "n_control": n_control,
        "n_dementia": n_dementia,
        "train_metrics_apparent": {
            "auc": float(roc_auc_score(y_spk, p_train)),
            "f1": float(f1_score(y_spk, y_hat, zero_division=0)),
            "accuracy": float(accuracy_score(y_spk, y_hat)),
        },
        "training_inputs": {
            "tan_npz": str(TAN_NPZ.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        },
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
