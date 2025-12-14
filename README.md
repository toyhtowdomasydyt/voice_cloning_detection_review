# Voice Cloning Detection Review - Experiment Recreation

PhD research experiment evaluating open-source AI models for detecting AI-generated voice clones. Compares three detection architectures: Classical ML (SVM), Hybrid Vision Transformers (ViT), and End-to-End Deep Learning (RawNet2).

## Project Status

- ✅ Dataset organized (3,600 audio files)
- ✅ Jabberjay model repository installed
- ✅ Test A implementation complete
- ⏳ Additional models to be integrated

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd voice_cloning_detection_review
```

### 2. Install Dependencies

This project uses `uv` for Python package management:

```bash
# Install core dependencies
uv add huggingface-hub joblib librosa matplotlib numpy Pillow PyYAML scikit-learn torch transformers
```

### 3. Clone and Install Model Repositories

#### Method A: Clone All Required Repositories (Recommended)

```bash
# Create models directory
mkdir -p models
cd models

# Clone required repositories
git clone https://github.com/MattyB95/Jabberjay.git

# Optionals
git clone https://github.com/Sudarshanng7/Voice-Cloning-and-Fake-Audio-Detection.git
git clone https://github.com/WaliMuhammadAhmad/AIVoice-Detection.git
git clone https://github.com/Fahad-kacchi/Deep_Fake_Voice_Recognition.git
git clone https://github.com/Csun22/Synthetic-Voice-Detection-Vocoder-Artifacts.git
git clone https://github.com/Rishabhstar/Deep_fake_detection.git

cd ..
```

#### Method B: Install Models in Editable Mode (Pythonic Approach)

After cloning, install each repository as an editable package:

```bash
# Install Jabberjay (Primary - Required for ViT, AST, RawNet2)
uv pip install -e models/Jabberjay

# Install other repositories as needed
# uv pip install -e models/Voice-Cloning-and-Fake-Audio-Detection
# (Add more as you integrate them)
```

**Why editable mode (`-e`)?**
- ✅ Clean imports: `from Jabberjay.jabberjay import Jabberjay`
- ✅ No `sys.path` manipulation needed
- ✅ Better IDE support (autocomplete, type hints)
- ✅ Standard Python package management
- ✅ Easy to update: changes in `models/` are immediately available

### 4. Verify Setup

```bash
# Test Jabberjay installation
uv run python -c "from Jabberjay.jabberjay import Jabberjay; print('✅ Jabberjay installed')"

# Run a simple experiment
uv run jabberjay_models/jabberjay_test_A.py
```

## Project Structure

```
voice_cloning_detection_review/
├── datasets/                  # Audio dataset (3,600 files)
│   ├── real/                 # 400 real voice samples
│   └── fake/
│       ├── DT/               # 1,600 different text samples
│       └── ST/               # 1,600 same text samples
├── models/                    # Cloned GitHub repositories
│   ├── Jabberjay/            # ViT, AST, RawNet2 models
│   └── ...                   # Other model repositories
├── jabberjay_models/          # Experiment scripts
│   └── jabberjay_test_A.py   # Test A: Different Text
├── samples/                   # Original audio samples (source)
├── guide.md                   # Complete experiment methodology
├── DATASET_STATUS.md          # Dataset organization status
└── pyproject.toml            # Project dependencies
```

## Running Experiments

### Test A: Different Text (Real vs Fake/DT)

```bash
uv run jabberjay_models/jabberjay_test_A.py
```

This tests the model's ability to detect fake audio when spoken content varies.

### Configuration

Edit the script to adjust test parameters:

```python
# In jabberjay_test_A.py
MAX_REAL = 5   # Number of real samples to test
MAX_FAKE = 20  # Number of fake samples to test

# Change dataset
dataset=Dataset.ASVspoof2019  # or ASVspoof5, VoxCelebSpoof
```

## Expected Results (from guide.md)

- **Best Performer:** ViT with MFCC (~0.705 accuracy)
- **High Precision on XTTS/Tortoise:** ViT MelSpectrogram (>80%)
- **Struggles with ElevenLabs/RVC:** Most models (~51% precision)
- **Common Issue:** Model bias (classifying everything as fake)

## Dataset Details

- **Total:** 3,600 audio files
- **Real samples:** 400 (20 speakers × 20 samples)
- **Fake samples:** 3,200 (4 vendors × 20 speakers × 40 samples)
  - ElevenLabs: 800 files
  - RVC: 800 files
  - Tortoise: 800 files
  - XTTS: 800 files

## Adding New Models

To integrate additional detection models:

1. **Clone the repository to `models/`:**
   ```bash
   cd models
   git clone <repository-url>
   ```

2. **Install in editable mode:**
   ```bash
   uv pip install -e models/<repository-name>
   ```

3. **Import in your scripts:**
   ```python
   # Clean imports - no sys.path needed!
   from model_package import ModelClass
   ```

4. **Create experiment script in `jabberjay_models/`:**
   ```python
   from pathlib import Path
   from model_package import ModelClass
   
   PROJECT_ROOT = Path(__file__).resolve().parent.parent
   REAL_DIR = PROJECT_ROOT / "datasets" / "real"
   # ... rest of implementation
   ```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Reinstall the package in editable mode
uv pip install -e models/Jabberjay
```

### GPU/MPS Warnings

The "Device set to use mps:0" messages are normal - PyTorch is using your Mac's GPU for acceleration.

### Matplotlib Warnings

The "More than 20 figures" warning is expected during batch processing. It doesn't affect results.

## Documentation

- **`guide.md`** - Complete experiment methodology and requirements
- **`DATASET_STATUS.md`** - Dataset organization and verification
- **`.github/copilot-instructions.md`** - AI agent instructions for project context

## Contributing

When adding new experiments or models:
1. Follow the editable install pattern (`uv pip install -e`)
2. Use `PROJECT_ROOT` for path resolution
3. Import packages cleanly (no `sys.path` manipulation)
4. Document expected results and baselines

## References

Critical repositories used in this project:
- [Jabberjay](https://github.com/MattyB95/Jabberjay) - ViT, AST, RawNet2 implementations
- See `guide.md` for complete list of model repositories
