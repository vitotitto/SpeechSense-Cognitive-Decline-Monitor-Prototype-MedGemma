# Conversation-Based Cognitive Decline Risk Triage Using HAI-DEF Models

![Application screenshot](../plots/app_screenshot.png)

## 1. Problem and Clinical Need

Dementia affects an estimated 57 million people globally (WHO, 2021). Only 20–50% of cases may be recognised in primary care (Alzheimer’s Disease International). Current screening depends on structured clinic visits, trained administrators, and patient compliance — barriers that delay detection until cognitive decline is already advanced.

I built a prototype of a system that estimates cognitive risk from **natural conversational speech**,  that is already occurring in check-in calls and caregiver conversations. It combines linguistic content analysis with acoustic marker profiling to produce a speaker-level risk score that flags individuals who warrant formal cognitive screening. 

Possible target users include primary care clinicians, geriatric care coordinators, and clinical services such as NHS 111.

A lot more analysis went into this project than this file contains, so if you are interested, please visit the doc folder at https://github.com/vitotitto/SpeechSense-Cognitive-Decline-monitoring-MedGemma/tree/main/docs

There will be more about this project. 

Subtle speech changes (rate decline, increased pauses, reduced pitch variability) are below clinical observation thresholds but detectable by ML algorithms. Prior work has shown LLM embeddings from transcripts can predict dementia (Agbavor & Liang, PLOS Digital Health 2022: 80.3% accuracy on ADReSSo using GPT-3 embeddings), and longitudinal public-figure datasets (DementiaNet [Gite et al., GitHub]; ADCeleb [Gao et al., INTERSPEECH 2025]) confirm pre-diagnosis signal exists. 

My contribution is embedding acoustic markers through MedGemma’s language pathway, in addition to an extensive evaluation of HAI-DEF models for this purpose. 

**Potential impact**. The WHO estimates 75% of dementia cases globally remain undiagnosed. In the UK alone, approximately 11 million GP consultations per year involve patients aged 65 or older. If even a passive screening layer flagged 5% of those conversations for further review, that represents ~550,000 additional triage opportunities annually at near-zero cost (the conversations are already happening and very often are recorded and stored). 

## 2. Technical Approach

### 2.1 Dataset and LLM-Assisted Curation Pipeline

The dataset has 196 unique speakers, 92 of whom have a diagnosis of dementia, vascular dementia or Alzheimer's.  The remaining speakers were age-matched controls.

This gave us an agentic dataset of 2,834 speech segments from various interviews. 
More about this part of the project can be found here: 

https://github.com/vitotitto/Speaker_selection_pipeline

Raw media recordings were processed through: (1) speaker diarisation to isolate target speakers, (2) voice activity detection and single-speaker confidence filtering, (3) LLM-based extraction and verification of temporal metadata, and per-segment quality scoring. 

Segments span a median duration of ~13 seconds (range: 4–216s), with a median of ~12 segments per speaker. Dementia-group speakers are represented across temporal categories spanning 5 to 15+ years before diagnosis. 

The results and analyses in the submission exclude post-symptom recordings, the model is evaluated on pre-diagnosis speech, so it cannot exploit speakers discussing their own diagnosis.

### 2.2 Pipeline Architecture

**Transcription (MedASR and later Pyannote Precision-2).** Audio segments are transcribed using MedASR and Pyannote with chunked decoding (20s windows, 2s stride) and beam search. Transcription quality proved to be the single largest determinant of downstream performance. Proper chunking and decoding improved text-based AUC from ~0.63 to ~0.90, a larger effect than any architectural choice in this experiment. 

**Acoustic Feature Extraction.** The production pipeline uses lightweight signal-processing metrics computed via librosa: speech rate, pause-to-speech ratio, pitch variability (F0 standard deviation), and spectral energy distribution,  across prosodic (0–1 kHz), articulatory (1–4 kHz), and high-frequency (4–8 kHz) bands. 

These descriptors are converted into a **natural-language acoustic narrative** rather than a numerical feature vector. For ablation and convergent validity analysis, a HeAR ViT backbone is also probed by extracting attention-derived features (CLS token attention weights across all 24 transformer layers (not the standard 512-dim pooled embedding)), aggregated (mean + standard deviation) to the speaker level. 

Although HeAR was designed for health acoustics (cough, breathing), speech prosody and phonation share acoustic features with respiratory signals.

**Clinical Reasoning (MedGemma 4B-IT).** MedGemma receives each transcript alongside its acoustic narrative as a unified text prompt. For example:

> *"Assess the linguistic and cognitive characteristics of this speech"*
> *Transcript : full verbatim text from MedASR or Pyannote transcription (the whole audio file concatenated)*

> *Acoustic narrative - 14 waveform-derived metrics formatted as natural language:*
>    *- speech_rate_wps - words per second*
>    *- pause_ratio - percentage of silence*
>    *- f0_mean / f0_std - fundamental frequency (pitch) and its variability*
>    *- low_pct / mid_pct / high_pct - spectral energy distribution across 3 bands*
>    *- low_mid_ratio - ratio of low+mid to high frequency energy*
>    *- centroid_mean / centroid_std - spectral centroid*
>    *- bandwidth_mean - spectral bandwidth*
>    *- tempo_bpm - rhythmic tempo proxy*
>    *- rms_std - prosodic (loudness) variability*
>    *- formant_transition_proxy - articulatory smoothness (delta MFCC)*

This design embeds the acoustic modality through MedGemma’s **text pathway** rather than forcing it through a vision encoder.

MedGemma jointly reasons over  linguistic content and acoustic characteristics, producing embeddings that reflect the clinical signal of both modalities. The model runs with 4-bit quantisation on a consumer GPU (NVIDIA RTX 3080, 10 GB VRAM), demonstrating edge deployability.

**Stage 4 — Fusion and Classification.** MedGemma embeddings are extracted as the mean-pooled last hidden state (2560-dimensional) across all input tokens per segment. Speaker-level features are constructed by aggregating these embeddings (mean + standard deviation across all segments). In ablation experiments, HeAR-derived speaker-level attention features are also concatenated. A logistic regression classifier with L2 regularisation produces the final risk score.

Evaluation uses 5-fold GroupKFold cross-validation with speaker ID as the grouping variable, to make sure no speaker appears in both training and test sets. The classification threshold (0.5) is applied to the out-of-fold predicted probabilities. 

Application prototype shown in the video is available : https://github.com/vitotitto/SpeechSense-App-prototype

The primary analysis code : https://github.com/vitotitto/SpeechSense-Cognitive-Decline-monitoring-MedGemma/tree/main/docs

### 2.3 Novel Contribution: Acoustic Narrative Fusion

The key methodological part is that embedding acoustic descriptors as structured natural language within the text prompt shows stronger generalisation to new speakers than adding raw, high-dimensional acoustic features. HeAR-derived attention features appear speaker-discriminative in our setting and may overfit in small-speaker settings.

The acoustic narrative avoids this by describing speech patterns at a clinically interpretable level (speech rate, pause ratio, pitch variability) rather than encoding speaker-specific acoustic signatures. 

Interestingly, it contrasts with the standard ADReSS/ADReSSo paradigm, which treats acoustic features as numerical vectors and feeds them to a separate encoder. The current approach uses MedGemma’s medical text reasoning as a joint clinical interpreter of both content and delivery. It requires no additional model components, no vision encoder, and naturally produces interpretable clinical narratives.

## 3. Results and Robustness

### 3.1 Cross-Validated Performance (Speaker-Level)

All results are reported on the full curated dataset with post-symptom segments excluded ([~188] speakers, pre-diagnosis speech only), using per-speaker out-of-fold predictions from GroupKFold cross-validation.

| Approach                                               | CV AUC    | F1 (at 0.5) | Accuracy  |
| ------------------------------------------------------ | --------- | ----------- | --------- |
| MedGemma text only (transcript)                        | 0.9       | 0.8021      | 0.8112    |
| **MedGemma text + acoustic narrative**                 | **0.911** | **0.829**   | **0.851** |
| Feature concatenation (text + HeAR attention features) | 0.917     | 0.862       | 0.872     |

*F1 and accuracy reported at fixed threshold 0.5 on out-of-fold predictions. HeAR concatenation achieves highest CV AUC but does not generalise to holdout speakers (see §3.2). Attention-derived features from HeAR backbone, not the standard 512-dim pooled embedding.*

### 3.2 Preliminary Holdout Evaluation

The model was evaluated on 6 completely unseen speakers (3 dementia, 3 control) processed through the identical curation pipeline:

| Condition                                           | Holdout AUC |
| --------------------------------------------------- | ----------- |
| Text only                                           | 1.000       |
| **Text + acoustic narrative**                       | **1.000**   |
| Text + acoustic narrative + HeAR attention features | 0.667       |

Text-only (transcript embeddings) and text + acoustic narrative both preserved correct rank-ordering of all 6 speakers (AUC 1.0). At threshold 0.5, text-only achieved 4/6, while text + acoustic narrative achieved 5/6 (one marginal false positive at 0.614). Adding HeAR-derived features degraded ranking (AUC 0.667), with one control speaker pushed high and one dementia speaker pushed low.  Despite a very tiny holdout dataset, this may suggest that raw, high-dimensional acoustic representations introduce speaker-discriminative nuisance, whereas injecting interpretable acoustic descriptors via MedGemma’s text pathway is more robust.

Given the small holdout (6 speakers, 9 pairwise comparisons), this could be treated this as directional evidence. It should be noted that perfect text-only AUC on a small sample warrants caution — topic-adjacent content in transcripts (interview framing, health context) could contribute. The post-symptom exclusion mitigates the most direct form of this leakage. Future work should prioritise expanding speaker holdout coverage and reporting sensitivity at fixed specificity to match clinical triage use.

The recommended production model is text + acoustic narrative (CV AUC 0.911, holdout AUC 1.000).

*Note: the demo video states 6/6; the correct figure is 5/6.*

### 3.3 Robustness Checks

**Topic leakage.** Including post-symptom segments improves AUC by only ~0.01, confirming the signal originates from pre-diagnosis speech.

**Speaker leakage.** GroupKFold on speaker ID, zero cross-speaker or cross-label clip collisions confirmed.

**Acoustic confounds.** ROCKET frequency-band analysis confirmed a diagnostic signal in the prosodic (0–1 kHz) and articulatory (1–4 kHz) ranges, consistent with known speech biomarkers rather than recording artefacts.

### 3.4 Convergent Validity

As an independent acoustic baseline, ROCKET (random convolutional kernels on mel spectrograms) achieved 0.72 AUC on the same dataset using no learned representations,  confirming that a genuine acoustic signal exists in the data. The text pathway breaks well above this ceiling because linguistic content captures cognitive markers that transfer across speakers, while raw acoustic features plateau.

## 4. Product Concept: Longitudinal Cognitive Risk Tracker

The system is designed as a longitudinal monitoring tool. Each conversation produces a risk score with a MedGemma-generated clinical narrative identifying which features raised concern. Over multiple sessions, the system builds a per-patient risk trajectory, score trend drives the triage decision, not any single assessment. A clinician-facing report surfaces the highest-concern segments, acoustic profile changes, and risk score history.

**Deployment:** The production pipeline requires only MedASR and MedGemma 4B, no HeAR model loading needed. Acoustic features are calculated using lightweight librosa signal processing, reducing GPU memory and inference time. MedGemma runs on consumer-grade hardware with 4-bit quantisation (demonstrated on NVIDIA RTX 3080, 10 GB VRAM). On-device inference preserves patient privacy. No audio leaves the local environment in case of MedASR usage or switching to the Pyannote community model. 

**HAI-DEF Models Used**

| **MedGemma 4B-IT**  | Clinical reasoning over transcripts + acoustic narratives; embedding extraction; risk score generation | Core model of the tool                                       |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **MedASR**          | Transcription; compared against Pyannote precision-2         | Evaluated, Pyannote used for primary transcription           |
| **HeAR (backbone)** | Acoustic analysis, convergent validity, ablation study       | Evaluated; attention-derived features overfit on holdout, dropped from production in favour of acoustic narrative approach |

*Full analysis of all three models, including ablation studies, holdout evaluation, and the HeAR overfitting, is documented in the* https://github.com/vitotitto/SpeechSense-Cognitive-Decline-monitoring-MedGemma/tree/main/docs

## 5. Limitations and Future Work

**Dataset.** The dataset is media-driven and heterogeneous. The model may learn conversational context patterns correlated with illness discussion, acceptable for triage settings where people naturally discuss health, but not a standalone biomarker. Results require a check on clinical populations with confirmed diagnoses.

**Transcription dependency.** Performance is heavily dependent on transcript quality. Switching transcription approaches on identical audio improved text-based AUC from ~0.63 to ~0.90.  The production pipeline uses Pyannote precision-2 for primary transcription with MedASR providing complementary medical-domain transcription. Deployment must use a high-quality ASR system.

**Future directions.** Cross-dataset validation (ADReSS/ADReSSo), transcript masking test (health keywords removed) to further rule out topic leakage; ablation of narrative vs. numeric acoustic prompts, fine-tuning on clinical conversation data and multi-language support.

------

*Source code, full experimental log and dataset documentation are available in the accompanying repositories. Released under CC BY 4.0, consistent with competition guidance.*

------

# References

1. **WHO (2021).** Dementia fact sheet. World Health Organization.
   https://www.who.int/news-room/fact-sheets/detail/dementia
2. **Alzheimer’s Disease International.** Dementia statistics.
   https://www.alzint.org/about/dementia-facts-figures/dementia-statistics/
3. **Agbavor, F. & Liang, H. (2022).** Predicting dementia from spontaneous speech using large language models. *PLOS Digital Health*, 1(12), e0000168.
   https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000168
4. **Gite, S. et al.** DementiaNet: A longitudinal spontaneous speech dataset for dementia screening. GitHub.
   https://github.com/shreyasgite/dementianet
5. **Gao, K., Favaro, A., Dehak, N. & Moro Velazquez, L. (2025).** ADCeleb: A Longitudinal Speech Dataset from Public Figures for Early Detection of Alzheimer’s Disease. *Proc. Interspeech 2025*, 5688–5692.
   https://www.isca-archive.org/interspeech_2025/gao25b_interspeech.html
6. **Google Health AI Developer Foundations.** HeAR model card.
   https://developers.google.com/health-ai-developer-foundations/hear/model-card
7. **Google Health AI Developer Foundations.** MedGemma documentation.
   https://developers.google.com/health-ai-developer-foundations/medgemma
8. **Google Health AI Developer Foundations.** MedASR documentation.
   https://developers.google.com/health-ai-developer-foundations/medasr