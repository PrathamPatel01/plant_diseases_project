# # src/pretty_predict.py
# # Run:
# #   python -m src.pretty_predict
# # Or predict a specific image:
# #   python -m src.pretty_predict "/path/to/image.jpg"

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TensorFlow hardware/info logs

# import sys
# import random
# import numpy as np
# from pathlib import Path
# from tensorflow import keras

# from src.split_data import split_dataset
# from src.dataset_utils import list_class_dirs
# from src.labels import build_label_map
# from src.preprocess import preprocess_image
# from src.config import IMAGE_SIZE


# def pretty_predict_one(img_path: Path, model, image_size, idx_to_class):
#     img_path = Path(img_path)

#     print("\n" + "=" * 70)
#     print("✅ PLANT DISEASE MVP — INFERENCE (INPUT → OUTPUT)")
#     print("=" * 70)

#     # STEP 1: Raw input
#     print("\n🔹 STEP 1: RAW INPUT")
#     print("Image file path:", str(img_path))
#     print("Folder name (acts like true label in PlantVillage):", img_path.parent.name)

#     # STEP 2: Preprocess image
#     x = preprocess_image(img_path, image_size=image_size)  # (H,W,3) float32 in [0,1]

#     print("\n🔹 STEP 2: AFTER PREPROCESSING")
#     print("Array shape (H,W,C):", x.shape)
#     print("Data type:", x.dtype)
#     print("Pixel range:", f"{float(x.min()):.4f} to {float(x.max()):.4f}")

#     # STEP 3: Model expects batch dimension
#     x_batch = np.expand_dims(x, axis=0)  # (1,H,W,3)

#     print("\n🔹 STEP 3: MODEL INPUT")
#     print("Batch shape passed to model:", x_batch.shape)

#     # STEP 4: Model output probabilities
#     probs = model.predict(x_batch, verbose=0)  # (1, 38)

#     print("\n🔹 STEP 4: RAW MODEL OUTPUT")
#     print("Output shape:", probs.shape)
#     print("First 10 probabilities:", np.round(probs[0][:10], 6))
#     print("Sum of probabilities (should be ~1.0):", float(probs[0].sum()))

#     # STEP 5: Decode prediction
#     pred_idx = int(np.argmax(probs[0]))
#     pred_class = idx_to_class[pred_idx]
#     confidence = float(probs[0][pred_idx])

#     print("\n🔹 STEP 5: FINAL DECISION")
#     print("Predicted class index:", pred_idx)
#     print("Predicted class name:", pred_class)
#     print("Confidence:", f"{confidence * 100:.2f}%")

#     # Top-5 predictions (extra clarity)
#     topk = 5
#     top_indices = np.argsort(probs[0])[::-1][:topk]
#     print(f"\n🔹 TOP {topk} CLASSES")
#     for rank, i in enumerate(top_indices, start=1):
#         print(f"{rank}. {idx_to_class[int(i)]:45s} -> {probs[0][int(i)]*100:6.2f}%")

#     print("\n" + "=" * 70 + "\n")


# def main():
#     # Load model
#     model_path = Path("outputs/mvp_model.keras")
#     model = keras.models.load_model(model_path)
#     print(f"Loaded model ✅ ({model_path})")

#     # Build label maps (idx -> class)
#     class_dirs = list_class_dirs()
#     _, idx_to_class = build_label_map(class_dirs)

#     # Use TEST set for honest check (unseen images)
#     _, _, test_files = split_dataset()

#     # Pick 10 random test images
#     k = 200
#     samples = random.sample(test_files, k)

#     correct = 0
#     print("\n" + "=" * 70)
#     print(f"✅ QUICK MVP CHECK — {k} RANDOM TEST IMAGES")
#     print("=" * 70)

#     for i, (img_path, true_label) in enumerate(samples, start=1):
#         img_path = Path(img_path)

#         # Preprocess + predict
#         x = preprocess_image(img_path, image_size=IMAGE_SIZE)
#         x_batch = np.expand_dims(x, axis=0)
#         probs = model.predict(x_batch, verbose=0)[0]

#         pred_idx = int(np.argmax(probs))
#         pred_label = idx_to_class[pred_idx]
#         conf = float(probs[pred_idx])

#         ok = (pred_label == true_label)
#         correct += int(ok)

#         mark = "✅" if ok else "❌"
#         print(f"\n{i}) {mark}")
#         print("   True :", true_label)
#         print("   Pred :", pred_label, f"({conf*100:.2f}%)")
#         print("   File :", str(img_path.name))

#     print("\n" + "-" * 70)
#     print(f"RESULT: {correct}/{k} correct  ->  Accuracy: {(correct/k)*100:.2f}%")
#     print("=" * 70 + "\n")



# if __name__ == "__main__":
#     main()


import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from src.config import MODEL_PATH, LABELS_PATH, IMAGE_SIZE


def load_labels(labels_path: Path = LABELS_PATH) -> Dict[int, str]:
    """
    Load index-to-label mapping from labels.json.

    Returns:
    {
        0: "Apple___Apple_scab",
        1: "Apple___Black_rot"
    }
    """
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}. Train the model first."
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    index_to_label = label_data["index_to_label"]

    return {
        int(index): label
        for index, label in index_to_label.items()
    }


def load_trained_model(model_path: Path = MODEL_PATH):
    """
    Load saved Keras model.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )

    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def preprocess_image_for_prediction(image) -> np.ndarray:
    """
    Preprocess one image for MobileNetV2 prediction.

    Input can be:
    - file path
    - PIL Image
    - Streamlit uploaded file

    Output shape:
    (1, height, width, 3)
    """
    import tensorflow as tf

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = Image.open(image).convert("RGB")

    img = img.resize(IMAGE_SIZE)

    arr = np.asarray(img, dtype=np.float32)

    # MobileNetV2 expects values in [-1, 1]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    return arr


def predict_image(
    image,
    model=None,
    index_to_label: Dict[int, str] = None,
    top_k: int = 3,
) -> Dict:
    """
    Predict disease class for a single image.

    Returns:
    {
        "predicted_class": "...",
        "confidence": 0.94,
        "top_predictions": [
            {"class_name": "...", "confidence": 0.94},
            ...
        ]
    }
    """
    if model is None:
        model = load_trained_model()

    if index_to_label is None:
        index_to_label = load_labels()

    processed_image = preprocess_image_for_prediction(image)

    probabilities = model.predict(processed_image, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    top_predictions: List[Dict] = []

    for index in top_indices:
        top_predictions.append(
            {
                "class_name": index_to_label[int(index)],
                "confidence": float(probabilities[index]),
            }
        )

    best_prediction = top_predictions[0]

    return {
        "predicted_class": best_prediction["class_name"],
        "confidence": best_prediction["confidence"],
        "top_predictions": top_predictions,
    }


def format_class_name(class_name: str) -> Dict[str, str]:
    """
    Convert PlantVillage class name into readable plant and disease names.

    Example:
    "Tomato___Early_blight"

    Returns:
    {
        "plant": "Tomato",
        "condition": "Early blight",
        "is_healthy": False
    }
    """
    if "___" in class_name:
        plant, condition = class_name.split("___", 1)
    else:
        plant, condition = "Unknown", class_name

    plant = plant.replace("_", " ").title()
    condition = condition.replace("_", " ").title()

    is_healthy = "healthy" in condition.lower()

    return {
        "plant": plant,
        "condition": condition,
        "is_healthy": is_healthy,
    }


def risk_level(confidence: float, is_healthy: bool) -> str:
    """
    Simple confidence-aware risk level.
    """
    if is_healthy:
        return "Low"

    if confidence >= 0.85:
        return "High"

    if confidence >= 0.60:
        return "Medium"

    return "Uncertain"


def recommendation_for_prediction(class_name: str) -> str:
    """
    Simple recommendation text for UI.

    This is not medical/agricultural expert advice.
    It is a general guidance layer for product-style output.
    """
    readable = format_class_name(class_name)

    if readable["is_healthy"]:
        return (
            "The leaf appears healthy. Continue regular monitoring, proper watering, "
            "and balanced sunlight exposure."
        )

    return (
        "The model detected a possible disease pattern. Isolate affected leaves if possible, "
        "avoid overhead watering, improve airflow, and consult a local agricultural expert "
        "before applying treatment."
    )