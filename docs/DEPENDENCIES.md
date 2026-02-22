# Dependencies

## Environment

The runs were executed in a conda environment named `audio` on Windows with CUDA.

## Core Python Packages

- `torch`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Optional / Pipeline-Specific

- `pyannote` helper/API utilities (used by diarization scripts)
- `jupyter` (for notebook execution)
- `git-lfs` (required to publish large audio/artifact files to GitHub)

## Notes

- For publication checks, saved artifacts are already included in `reproducibility_artifacts`.
- Reprocessing from raw WAV is optional and significantly slower.
