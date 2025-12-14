from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from Jabberjay.jabberjay import Jabberjay
from Jabberjay.Utilities.enum_handler import Dataset
from Jabberjay.Models.Tranformer.VIT.MFCC import run as VIT_MFCC

# Get project root directory (one level up from jabberjay_models/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration - use absolute paths relative to project root
REAL_DIR = PROJECT_ROOT / "datasets" / "real"
FAKE_DT_DIR = PROJECT_ROOT / "datasets" / "fake" / "DT"
FAKE_ST_DIR = PROJECT_ROOT / "datasets" / "fake" / "ST"

# How many samples to test (set to None for all files)
MAX_REAL = 5  # Start with just 5 real files
MAX_FAKE = 20  # And 20 fake files (to maintain ~1:4 ratio)


def predict_with_vit_mfcc(audio_path, dataset=Dataset.ASVspoof2019):
    """
    Predict using ViT with MFCC features.
    Returns: 0 for Real (Bonafide), 1 for Fake (Spoof)
    """
    jabberjay = Jabberjay()
    audio = jabberjay.load(filename=str(audio_path))

    # Get prediction
    result = VIT_MFCC.predict(audio=audio, dataset=dataset)

    # result is a list like [{'label': 'Bonafide', 'score': 0.999}]
    is_bonafide = result[0].get("label") == "Bonafide"

    return 0 if is_bonafide else 1


def run_simple_test(test_name, real_dir, fake_dir, dataset=Dataset.ASVspoof2019):
    """
    Run a simple test with ViT MFCC model.

    Args:
        test_name: Name of test (e.g., "Test A - DT")
        real_dir: Path to real samples
        fake_dir: Path to fake samples
        dataset: Dataset type (ASVspoof2019, ASVspoof5, VoxCelebSpoof)
    """
    print(f"\n{'=' * 70}")
    print(f"{test_name}")
    print(f"Model: ViT + MFCC")
    print(f"Dataset: {dataset.value}")
    print(f"{'=' * 70}\n")

    y_true = []
    y_pred = []

    # Get real files
    real_files = list(real_dir.glob("*.wav")) + list(real_dir.glob("*.mp3"))
    if MAX_REAL:
        real_files = real_files[:MAX_REAL]

    print(f"Testing {len(real_files)} REAL samples...")
    for i, audio_file in enumerate(real_files, 1):
        print(f"  [{i}/{len(real_files)}] {audio_file.name}...", end=" ", flush=True)

        try:
            prediction = predict_with_vit_mfcc(audio_file, dataset=dataset)
            result_text = "Real" if prediction == 0 else "FAKE"
            status = "✓" if prediction == 0 else "✗"
            print(f"{status} → {result_text}")

            y_true.append(0)  # Real
            y_pred.append(prediction)

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            y_true.append(0)
            y_pred.append(1)  # Default to fake on error

    # Get fake files
    fake_files = list(fake_dir.glob("*.wav")) + list(fake_dir.glob("*.mp3"))
    if MAX_FAKE:
        fake_files = fake_files[:MAX_FAKE]

    print(f"\nTesting {len(fake_files)} FAKE samples...")
    for i, audio_file in enumerate(fake_files, 1):
        print(f"  [{i}/{len(fake_files)}] {audio_file.name}...", end=" ", flush=True)

        try:
            prediction = predict_with_vit_mfcc(audio_file, dataset=dataset)
            result_text = "Real" if prediction == 0 else "FAKE"
            status = "✓" if prediction == 1 else "✗"
            print(f"{status} → {result_text}")

            y_true.append(1)  # Fake
            y_pred.append(prediction)

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            y_true.append(1)
            y_pred.append(1)

    # Calculate metrics
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall:    {recall:.4f} (weighted)")
    print(f"F1 Score:  {f1:.4f} (weighted)")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {tp:3d} - Fake correctly identified as fake")
    print(f"  True Negatives (TN):  {tn:3d} - Real correctly identified as real")
    print(f"  False Positives (FP): {fp:3d} - Real incorrectly labeled as fake")
    print(f"  False Negatives (FN): {fn:3d} - Fake incorrectly labeled as real")

    print(f"{'=' * 70}\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def main():
    print("\n" + "=" * 70)
    print("SIMPLE EXPERIMENT - Voice Cloning Detection")
    print("=" * 70)
    print(f"\nTesting with {MAX_REAL} real + {MAX_FAKE} fake samples")
    print("(Edit MAX_REAL and MAX_FAKE in the script to test more)\n")

    # Test A: Real vs Fake/DT with ViT MFCC
    run_simple_test(
        test_name="Test A: Different Text (ViT + MFCC)",
        real_dir=REAL_DIR,
        fake_dir=FAKE_DT_DIR,
        dataset=Dataset.ASVspoof2019,
    )


if __name__ == "__main__":
    main()
