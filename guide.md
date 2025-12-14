# **Project Documentation: Recreation of AI Voice Detection Experiments**

## **1. Context and Problem Statement**
The rapid advancement of AI voice synthesis has created significant cybersecurity challenges. Technologies for voice cloning have evolved from requiring large samples to "Zero-shot" systems capable of cloning a voice from a few seconds of audio.

**The Core Problem:**
*   **Biometric Vulnerability:** Cloned voices can bypass Voice authentication systems (ASV) and Presentation Attack Detection (PAD) systems.
*   **Social Engineering:** Attackers use clones for financial fraud (e.g., "fake boss" scams) and disinformation (deepfakes).
*   **Detection Gap:** While synthesis tools (like ElevenLabs, RVC) are highly advanced, detection mechanisms often lag behind.

**Experiment Goal:**
The objective of this experiment is to evaluate the efficiency of **existing open-source AI models** in detecting modern voice clones. The study compares three architectural approaches:
1.  **Classical:** Machine Learning (SVM) using manual feature extraction.
2.  **Hybrid:** Vision Transformers (ViT) analyzing audio spectrograms as images.
3.  **End-to-End:** Deep Neural Networks (RawNet2) analyzing raw audio waveforms.

---

## **2. Dataset Specification**
*Note: The original audio files are not provided in the source. You must reconstruct the dataset using the following structure to match the experiment's statistical validity.*

**Total Volume:** 3,600 Audio Files.
**Audio Specs:** Formats `.wav`/`.mp3`; Sample rates 20kHz, 24kHz, 48kHz; Mono channel.

### **Directory Structure**
*   **`/dataset/real`** (400 files)
    *   **Content:** 20 unique speakers × 20 samples each.
    *   **Source:** Collect from open datasets (e.g., Common Voice, LJ Speech).
*   **`/dataset/fake`** (3,200 files)
    *   **Vendors:** You must generate clones using **ElevenLabs, RVC, XTTS, and Tortoise**.
    *   **Sub-folder `/DT` (Different Text - 1,600 files):** Clones speaking text *different* from the real samples. (400 per vendor).
    *   **Sub-folder `/ST` (Same Text - 1,600 files):** Clones speaking the *exact same* text as the real samples (to test for content bias).

---

## **3. Required Models and Software**
The experiment relies on pre-trained models hosted on GitHub. You must clone these repositories.

**Python Libraries:** `scikit-learn` (metrics), `librosa` (audio processing), `tracemalloc` (memory monitoring).

**Target Repositories:**
1.  **`MattyB95 / Jabberjay`** (Critical: Contains the ViT, AST, and RawNet2 implementations).
2.  **`Sudarshanng7 / Voice-Cloning-and-Fake-Audio-Detection`**.
3.  **`WaliMuhammadAhmad / AIVoice-Detection`**.
4.  **`Fahad-kacchi / Deep_Fake_Voice_Recognition`**.
5.  **`Csun22 / Synthetic-Voice-Detection-Vocoder-Artifacts`**.
6.  **`Rishabhstar / Deep_fake_detection`**.

---

## **4. Methodology**
You must run the following three distinct tests.

### **Test A: Accuracy on Different Text (Realistic Scenario)**
*   **Input:** Compare `/real` vs. `/fake/DT`.
*   **Goal:** Determine if the model can detect a fake when the spoken content is random.

### **Test B: Accuracy on Same Text (Control Scenario)**
*   **Input:** Compare `/real` vs. `/fake/ST`.
*   **Goal:** Determine if the model is biased by the linguistic content rather than the voice features.

### **Test C: Performance Benchmarking**
*   **Input:** A random mix of 250 files from the dataset.
*   **Metric 1 (Time):** Average inference time per file. *Note: You must include model initialization/loading time in the total calculation, then divide by 250*.
*   **Metric 2 (Memory):** Average RAM consumption during the process.

---

## **5. Implementation Guide (Code Scripts)**
*Since the source text does not provide the code, use these wrappers to orchestrate the models from the repositories above.*

### **Script 1: Evaluation Wrapper (Accuracy/Metrics)**
This script iterates through your reconstructed dataset and calculates the metrics defined in the thesis (Weighted F1, Precision, Recall).

```python
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- ADAPTER: YOU MUST IMPLEMENT THE CONNECTION TO THE REPOS HERE ---
# Example: If testing MattyB95 ViT, import their predict function
def get_model_prediction(repo_name, file_path):
    """
    Returns 0 for REAL, 1 for FAKE.
    You must wrap the specific inference code from the cloned GitHub repos here.
    """
    # Pseudo-code example:
    # if repo_name == "MattyB95_ViT":
    #    audio = preprocess_spectrogram(file_path) # MFCC/MelSpectrogram/CQT
    #    return model.predict(audio)
    return 1 # Placeholder

def run_experiment(repo_name, real_dir, fake_dir):
    y_true = []
    y_pred = []

    print(f"--- Testing {repo_name} ---")
    
    # 1. Process Real Files (Label = 0)
    for f in os.listdir(real_dir):
        if f.endswith(('.wav', '.mp3')):
            pred = get_model_prediction(repo_name, os.path.join(real_dir, f))
            y_true.append(0)
            y_pred.append(pred)

    # 2. Process Fake Files (Label = 1)
    for f in os.listdir(fake_dir):
        if f.endswith(('.wav', '.mp3')):
            pred = get_model_prediction(repo_name, os.path.join(fake_dir, f))
            y_true.append(1)
            y_pred.append(pred)

    # 3. Calculate Metrics [Source: 101-109]
    print(f"Results for {repo_name}:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    # Thesis specifies 'weighted' average for F1 due to class imbalance
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}\n")

# Usage: Run for DT (Different Text) and ST (Same Text) separately
# run_experiment("MattyB95_ViT", "./dataset/real", "./dataset/fake/DT")
```

### **Script 2: Performance Benchmark**
Use this to reproduce the "Time and Memory Consumption" test.

```python
import time
import tracemalloc
import random

def benchmark_performance(repo_name, dataset_path, sample_count=250):
    # Select random files
    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path) for f in filenames]
    sample_files = random.sample(all_files, sample_count)

    tracemalloc.start()
    start_time = time.time()

    # --- START MEASUREMENT (Include Model Loading) ---
    # model = load_model(repo_name) 
    for file_path in sample_files:
        # result = model.predict(file_path)
        pass # Replace with actual call
    # --- END MEASUREMENT ---

    end_time = time.time()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate Average Time [Source: 116]
    total_time = end_time - start_time
    avg_time = total_time / sample_count
    
    print(f"Performance for {repo_name}:")
    print(f"Avg Time per File: {avg_time:.6f} sec")
    print(f"Peak RAM Usage: {peak_mem / (1024*1024):.2f} MB")
```

---

## **6. Expected Results & Review Criteria**
When reviewing your recreation results, compare them against the author's findings to check correctness.

**Accuracy Baselines:**
*   **Best Performer:** The **ViT (Vision Transformer)** models from the *Jabberjay* repo should perform best.
    *   **ViT with MFCC** features should yield the highest general accuracy (approx **0.705**).
    *   **ViT with MelSpectrogram** should have very high precision on XTTS and Tortoise samples (>80%) but struggle with ElevenLabs/RVC (~51%).
*   **Worst Performers:** The classical and simple CNN models (e.g., `csun22`, `sudarshanng7`) generally yielded low accuracy (~0.23 - 0.48) in the author's tests.

**Common Failure Mode:**
*   If your results show high "True Positives" but low overall accuracy, check for **bias**. The author noted that less effective models often classified *everything* as fake (high recall, low precision).

---

## **7. Recreation Checklist**

**Preparation**
- [ ] **Context Review:** Read Section 1.1–1.3 to understand the threat landscape (spoofing vs. PAD).
- [ ] **Data Generation:** 
    - [ ] Source 400 Real files.
    - [ ] Generate 1600 Fake files (Same Text) via 4 vendors.
    - [ ] Generate 1600 Fake files (Different Text) via 4 vendors.
- [ ] **Repo Setup:** Clone `MattyB95` and 5 other listed repos.

**Execution**
- [ ] **Feature Extraction:** Ensure the ViT model is tested with **MFCC**, **Mel-Spectrogram**, and **ConstantQ (CQT)** configurations separately.
- [ ] **Run Test A (DT):** Calculate F1 (weighted).
- [ ] **Run Test B (ST):** Compare results. (Did accuracy drop or stay stable? ViT MFCC was stable at ~0.66-0.70).
- [ ] **Run Test C (Perf):** Measure time/RAM on 250 samples.

**Analysis**
- [ ] Compare your Confusion Matrices with the tables in Section 3.3.1.
- [ ] Verify if XTTS/Tortoise clones are easier to detect than ElevenLabs/RVC (a key finding of the study).