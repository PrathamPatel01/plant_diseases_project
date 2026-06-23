# import numpy as np
# from tensorflow.keras.utils import Sequence
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow import keras
# import numpy as np


# from src.split_data import split_dataset
# from src.dataset_utils import list_class_dirs
# from src.labels import build_label_map
# from src.encode_labels import encode_labels
# from src.preprocess import preprocess_image
# from src.config import IMAGE_SIZE
# from src.model import build_cnn


# class DataGen(Sequence):
#     def __init__(self, encoded_list, batch_size=32, image_size=(128, 128), shuffle=True):
#         self.data = encoded_list
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.data))
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.ceil(len(self.data) / self.batch_size))

#     def __getitem__(self, idx):
#         batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch = [self.data[i] for i in batch_idx]

#         # PIL uses (width, height); our IMAGE_SIZE is (W,H)
#         W, H = self.image_size
#         X = np.zeros((len(batch), H, W, 3), dtype=np.float32)
#         y = np.zeros((len(batch),), dtype=np.int32)

#         for i, (img_path, label_idx) in enumerate(batch):
#             X[i] = preprocess_image(img_path, image_size=self.image_size)
#             y[i] = label_idx

#         return X, y

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)


# def compute_class_weights(y, num_classes):
#     counts = np.bincount(y, minlength=num_classes)
#     total = len(y)
#     weights = {}
#     for c in range(num_classes):
#         weights[c] = (total / (num_classes * counts[c])) if counts[c] > 0 else 0.0
#     return weights


# def main():
#     # 1) Split dataset
#     train_files, val_files, test_files = split_dataset()

#     # 2) Label maps
#     class_dirs = list_class_dirs()
#     class_to_idx, idx_to_class = build_label_map(class_dirs)
#     num_classes = len(class_to_idx)

#     # 3) Encode labels
#     encoded_train = encode_labels(train_files, class_to_idx)
#     encoded_val = encode_labels(val_files, class_to_idx)

#     # 4) Build model (your build_cnn should already compile the model)
#     model = build_cnn(num_classes)
#     print("Model built ✅")

#     # 5) Class weights for imbalance
#     y_train = np.array([y for _, y in encoded_train], dtype=np.int32)
#     class_weights = compute_class_weights(y_train, num_classes)
#     print("Class weights ready ✅")

#     # 6) Generators (no RAM explosion)
#     batch_size = 32
#     train_gen = DataGen(encoded_train, batch_size=batch_size, image_size=IMAGE_SIZE, shuffle=True)
#     val_gen = DataGen(encoded_val, batch_size=batch_size, image_size=IMAGE_SIZE, shuffle=False)

#     # 7) Train + save best model
#     callbacks = [
#         EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
#         ModelCheckpoint("outputs/mvp_model.keras", monitor="val_loss", save_best_only=True)
#     ]

#     history = model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=12,
#         class_weight=class_weights,
#         callbacks=callbacks,
#         verbose=1
#     )

#     print("\n✅ MVP training complete.")
#     print("Best model saved at: outputs/mvp_model.keras")

#     # ---------- SINGLE IMAGE PREDICTION (MVP DEMO) ----------

#     # Pick ONE image path manually (any image from dataset)
#     img_path = train_files[0][0]  # using first training image

#     # Load trained model
#     model = keras.models.load_model("outputs/mvp_model.keras")

#     # Preprocess image
#     img = preprocess_image(img_path, image_size=IMAGE_SIZE)
#     img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)

#     # Predict
#     probs = model.predict(img, verbose=0)[0]

#     pred_idx = np.argmax(probs)
#     confidence = np.max(probs)

#     pred_class = idx_to_class[pred_idx]

#     print("\n=== SINGLE IMAGE PREDICTION ===")
#     print("Image path:", img_path)
#     print("Predicted class:", pred_class)
#     print("Confidence:", round(float(confidence) * 100, 2), "%")



# if __name__ == "__main__":
#     main()
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    MODEL_DIR,
    REPORTS_DIR,
    MODEL_PATH,
    LABELS_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    EPOCHS,
)
from src.dataset import (
    collect_image_records,
    create_label_mapping,
    split_dataset,
    make_tf_dataset,
    dataset_summary,
)
from src.model import build_model


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_labels(label_mapping):
    index_to_label = {
        str(index): label
        for label, index in label_mapping.items()
    }

    label_data = {
        "label_to_index": label_mapping,
        "index_to_label": index_to_label,
        "num_classes": len(label_mapping),
    }

    save_json(label_data, LABELS_PATH)


def compute_training_class_weights(train_records, label_mapping):
    """
    Handles class imbalance.

    Without this, the model may focus too much on large classes
    and ignore smaller classes.
    """
    label_indices = [
        label_mapping[label]
        for _, label in train_records
    ]

    classes = np.array(sorted(set(label_indices)))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(label_indices),
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    return class_weights


def evaluate_model(model, test_ds, label_mapping):
    index_to_label = {
        index: label
        for label, index in label_mapping.items()
    }

    class_names = [
        index_to_label[i]
        for i in range(len(index_to_label))
    ]

    y_true = []

    for _, labels in test_ds:
        y_true.extend(labels.numpy().tolist())

    probabilities = model.predict(test_ds)
    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred).tolist()

    save_json(report, METRICS_PATH)
    save_json(
        {
            "class_names": class_names,
            "confusion_matrix": matrix,
        },
        CONFUSION_MATRIX_PATH,
    )

    return report, matrix


def main():
    import tensorflow as tf

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Plant Disease Intelligence - Training Pipeline")
    print("=" * 70)

    summary = dataset_summary()

    print("\nDataset Summary")
    print("-" * 70)
    print("Dataset path:", summary["dataset_path"])
    print("Total classes:", summary["total_classes"])
    print("Total images:", summary["total_images"])
    print("Imbalance ratio:", summary["imbalance_ratio"])

    records = collect_image_records()

    if not records:
        raise ValueError("No image records found. Check DATASET_DIR in config.py.")

    label_mapping = create_label_mapping(records)
    num_classes = len(label_mapping)

    print("\nNumber of classes:", num_classes)

    train_records, val_records, test_records = split_dataset(records)

    print("\nSplit Summary")
    print("-" * 70)
    print("Train images:", len(train_records))
    print("Validation images:", len(val_records))
    print("Test images:", len(test_records))

    save_labels(label_mapping)

    class_weights = compute_training_class_weights(
        train_records=train_records,
        label_mapping=label_mapping,
    )

    print("\nClass weights calculated.")

    train_ds = make_tf_dataset(
        train_records,
        label_mapping,
        shuffle=True,
    )

    val_ds = make_tf_dataset(
        val_records,
        label_mapping,
        shuffle=False,
    )

    test_ds = make_tf_dataset(
        test_records,
        label_mapping,
        shuffle=False,
    )

    print("\nBuilding model...")
    model = build_model(num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    print("\nTraining started...")
    print("-" * 70)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    history_data = {
        "created_at": datetime.now().isoformat(),
        "epochs": EPOCHS,
        "history": history.history,
    }

    save_json(history_data, HISTORY_PATH)

    print("\nEvaluating on test set...")
    report, _ = evaluate_model(
        model=model,
        test_ds=test_ds,
        label_mapping=label_mapping,
    )

    print("\nFinal Test Accuracy:", round(report["accuracy"], 4))

    print("\nSaved files")
    print("-" * 70)
    print("Model:", MODEL_PATH)
    print("Labels:", LABELS_PATH)
    print("History:", HISTORY_PATH)
    print("Metrics:", METRICS_PATH)
    print("Confusion Matrix:", CONFUSION_MATRIX_PATH)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()