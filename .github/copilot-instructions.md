# Voice Cloning Detection Review - AI Agent Instructions

## Project Overview
PhD research experiment evaluating open-source AI models for detecting AI-generated voice clones. Compares three detection architectures: Classical ML (SVM), Hybrid Vision Transformers (ViT), and End-to-End Deep Learning (RawNet2).

## Dataset Structure (Must Be Reconstructed)
The project requires 3,600 audio files organized as:
```
datasets/
├── real/          # 400 files: 20 speakers × 20 samples (Common Voice, LJ Speech)
└── fake/          # 3,200 files from 4 vendors (ElevenLabs, RVC, XTTS, Tortoise)
    ├── DT/        # 1,600 files: Different Text scenario (400 per vendor)
    └── ST/        # 1,600 files: Same Text scenario (tests content bias)
```
**Audio specs:** `.wav`/`.mp3`, 20-48kHz sample rates, mono channel.

## Critical Dependencies
This project wraps **6 external GitHub repositories** - do not implement detection models from scratch:
1. **`MattyB95/Jabberjay`** - Primary source for ViT, AST, RawNet2 implementations
2. `Sudarshanng7/Voice-Cloning-and-Fake-Audio-Detection`
3. `WaliMuhammadAhmad/AIVoice-Detection`
4. `Fahad-kacchi/Deep_Fake_Voice_Recognition`
5. `Csun22/Synthetic-Voice-Detection-Vocoder-Artifacts`
6. `Rishabhstar/Deep_fake_detection`

Required Python libraries: `scikit-learn` (metrics), `librosa` (audio processing), `tracemalloc` (memory profiling).

## Three Mandatory Test Scenarios
1. **Test A (Different Text):** Real vs. Fake/DT - measures detection on varied content
2. **Test B (Same Text):** Real vs. Fake/ST - controls for linguistic bias
3. **Test C (Performance):** 250 random files - measures inference time + RAM (include model loading time)

## Evaluation Metrics Pattern
Always use **weighted F1-score** due to class imbalance (1:8 real-to-fake ratio):
```python
f1_score(y_true, y_pred, average='weighted')
```
Report: Accuracy, Precision, Recall, F1, Confusion Matrix (TP/TN/FP/FN).

## Expected Performance Baselines
- **Best:** ViT with MFCC features (~0.705 accuracy)
- **High Precision on XTTS/Tortoise:** ViT MelSpectrogram (>80%)
- **Weak on ElevenLabs/RVC:** Most models struggle (~51% precision)
- **Worst:** Classical/simple CNNs (~0.23-0.48 accuracy)
- **Red flag:** High recall + low precision = model bias (classifying everything as fake)

## ViT Feature Configuration Requirement
Test ViT models with **three separate feature extractors**: MFCC, Mel-Spectrogram, ConstantQ (CQT). These are not interchangeable - each produces different results.

## Performance Measurement Details
For Test C timing: `total_elapsed_time / 250` where total includes model initialization. Use `tracemalloc` for peak RAM, not average.

## Common Pitfalls
- Do not train new models - wrap existing pretrained ones
- Dataset must maintain 20 speakers × 20 samples structure for validity
- ST/DT scenarios are not optional - they test different hypotheses
- Avoid bias toward labeling everything as fake (check confusion matrix)

## Key Reference
See `guide.md` for complete methodology, checklist, and code templates.
