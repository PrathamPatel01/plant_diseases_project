# src/pretty_predict.py
# Run:
#   python -m src.pretty_predict
# Or predict a specific image:
#   python -m src.pretty_predict "/path/to/image.jpg"

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TensorFlow hardware/info logs

import sys
import random
import numpy as np
from pathlib import Path
from tensorflow import keras

from src.split_data import split_dataset
from src.dataset_utils import list_class_dirs
from src.labels import build_label_map
from src.preprocess import preprocess_image
from src.config import IMAGE_SIZE


def pretty_predict_one(img_path: Path, model, image_size, idx_to_class):
    img_path = Path(img_path)

    print("\n" + "=" * 70)
    print("✅ PLANT DISEASE MVP — INFERENCE (INPUT → OUTPUT)")
    print("=" * 70)

    # STEP 1: Raw input
    print("\n🔹 STEP 1: RAW INPUT")
    print("Image file path:", str(img_path))
    print("Folder name (acts like true label in PlantVillage):", img_path.parent.name)

    # STEP 2: Preprocess image
    x = preprocess_image(img_path, image_size=image_size)  # (H,W,3) float32 in [0,1]

    print("\n🔹 STEP 2: AFTER PREPROCESSING")
    print("Array shape (H,W,C):", x.shape)
    print("Data type:", x.dtype)
    print("Pixel range:", f"{float(x.min()):.4f} to {float(x.max()):.4f}")

    # STEP 3: Model expects batch dimension
    x_batch = np.expand_dims(x, axis=0)  # (1,H,W,3)

    print("\n🔹 STEP 3: MODEL INPUT")
    print("Batch shape passed to model:", x_batch.shape)

    # STEP 4: Model output probabilities
    probs = model.predict(x_batch, verbose=0)  # (1, 38)

    print("\n🔹 STEP 4: RAW MODEL OUTPUT")
    print("Output shape:", probs.shape)
    print("First 10 probabilities:", np.round(probs[0][:10], 6))
    print("Sum of probabilities (should be ~1.0):", float(probs[0].sum()))

    # STEP 5: Decode prediction
    pred_idx = int(np.argmax(probs[0]))
    pred_class = idx_to_class[pred_idx]
    confidence = float(probs[0][pred_idx])

    print("\n🔹 STEP 5: FINAL DECISION")
    print("Predicted class index:", pred_idx)
    print("Predicted class name:", pred_class)
    print("Confidence:", f"{confidence * 100:.2f}%")

    # Top-5 predictions (extra clarity)
    topk = 5
    top_indices = np.argsort(probs[0])[::-1][:topk]
    print(f"\n🔹 TOP {topk} CLASSES")
    for rank, i in enumerate(top_indices, start=1):
        print(f"{rank}. {idx_to_class[int(i)]:45s} -> {probs[0][int(i)]*100:6.2f}%")

    print("\n" + "=" * 70 + "\n")


def main():
    # Load model
    model_path = Path("outputs/mvp_model.keras")
    model = keras.models.load_model(model_path)
    print(f"Loaded model ✅ ({model_path})")

    # Build label maps (idx -> class)
    class_dirs = list_class_dirs()
    _, idx_to_class = build_label_map(class_dirs)

    # Use TEST set for honest check (unseen images)
    _, _, test_files = split_dataset()

    # Pick 10 random test images
    k = 10
    samples = random.sample(test_files, k)

    correct = 0
    print("\n" + "=" * 70)
    print(f"✅ QUICK MVP CHECK — {k} RANDOM TEST IMAGES")
    print("=" * 70)

    for i, (img_path, true_label) in enumerate(samples, start=1):
        img_path = Path(img_path)

        # Preprocess + predict
        x = preprocess_image(img_path, image_size=IMAGE_SIZE)
        x_batch = np.expand_dims(x, axis=0)
        probs = model.predict(x_batch, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_class[pred_idx]
        conf = float(probs[pred_idx])

        ok = (pred_label == true_label)
        correct += int(ok)

        mark = "✅" if ok else "❌"
        print(f"\n{i}) {mark}")
        print("   True :", true_label)
        print("   Pred :", pred_label, f"({conf*100:.2f}%)")
        print("   File :", str(img_path.name))

    print("\n" + "-" * 70)
    print(f"RESULT: {correct}/{k} correct  ->  Accuracy: {(correct/k)*100:.2f}%")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    main()
