# MedGemma Hackathon Package (Pre-Symptoms Scope)

This folder is a GitHub-ready, reduced publication package for the hackathon scope:

- Dataset: `audio_clips` (deduplicated)
- Best condition focus: `text_acoustic_narrative_pre_symptoms`
- Pipeline: Pyannote transcription -> MedGemma text+acoustic narrative embeddings (2560-dim per clip) -> speaker-level aggregation [mean, std] = 5120 dims -> LogisticRegression classifier
- Outputs included: reproducibility JSONs, trained model artifact, notebook, and plots (AUC/F1/Accuracy + confusion matrix per speaker with after-symptoms clips excluded)

## Folder Layout

- `dataset/audio_clips/`
- `dataset/reference_manifests/manifest_agentic_clean.csv`
- `code/evaluate_text_acoustic_narrative_pre_symptoms.py`
- `code/train_text_acoustic_narrative_pre_symptoms.py`
- `code/generate_pre_symptoms_plots.py`
- `reproducibility/multimodal/outputs_text_acoustic_narrative_pre_symptoms/`
- `reproducibility/best_model/`
- `plots/`
- `notebooks/hackathon_pre_symptoms_repro.ipynb`

## Environment

```bash
pip install -r requirements.txt
```

Large files are tracked with Git LFS (`.gitattributes` included).

## Reproduce Core Outputs

Run from this folder (`gh_hackathon`):

```bash
python code/evaluate_text_acoustic_narrative_pre_symptoms.py
python code/train_text_acoustic_narrative_pre_symptoms.py
python code/generate_pre_symptoms_plots.py
```

## Produced Artifacts

- Metrics JSON:
  - `reproducibility/multimodal/outputs_text_acoustic_narrative_pre_symptoms/classification_metrics_pre_symptoms.json`
- Summary JSON:
  - `reproducibility/multimodal/outputs_text_acoustic_narrative_pre_symptoms/summary_pre_symptoms.json`
- Best model artifact:
  - `reproducibility/best_model/text_acoustic_narrative_pre_symptoms.pkl`
  - `reproducibility/best_model/text_acoustic_narrative_pre_symptoms_metadata.json`
- Speaker OOF + confusion matrix data:
  - `reproducibility/multimodal/outputs_text_acoustic_narrative_pre_symptoms/oof_text_acoustic_narrative_pre_symptoms.csv`
  - `reproducibility/multimodal/outputs_text_acoustic_narrative_pre_symptoms/cm_text_acoustic_narrative_pre_symptoms.json`
- Plots:
  - `plots/auc_f1_acc_pre_symptoms.png`
  - `plots/cm_text_acoustic_narrative_pre_symptoms.png`

## Scope Notes

- The published raw dataset in this package is the deduplicated set: `dataset/audio_clips`.
- To preserve exact compatibility with previously generated artifacts in this package, the default manifest used by scripts is:
  - `dataset/reference_manifests/manifest_agentic_clean.csv`
- If needed, you can switch scripts to `dataset/audio_clips/manifests/manifest.csv` for dedup-only reruns.
