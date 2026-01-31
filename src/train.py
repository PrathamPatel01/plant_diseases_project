import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
import numpy as np


from src.split_data import split_dataset
from src.dataset_utils import list_class_dirs
from src.labels import build_label_map
from src.encode_labels import encode_labels
from src.preprocess import preprocess_image
from src.config import IMAGE_SIZE
from src.model import build_cnn


class DataGen(Sequence):
    def __init__(self, encoded_list, batch_size=32, image_size=(128, 128), shuffle=True):
        self.data = encoded_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.data[i] for i in batch_idx]

        # PIL uses (width, height); our IMAGE_SIZE is (W,H)
        W, H = self.image_size
        X = np.zeros((len(batch), H, W, 3), dtype=np.float32)
        y = np.zeros((len(batch),), dtype=np.int32)

        for i, (img_path, label_idx) in enumerate(batch):
            X[i] = preprocess_image(img_path, image_size=self.image_size)
            y[i] = label_idx

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    weights = {}
    for c in range(num_classes):
        weights[c] = (total / (num_classes * counts[c])) if counts[c] > 0 else 0.0
    return weights


def main():
    # 1) Split dataset
    train_files, val_files, test_files = split_dataset()

    # 2) Label maps
    class_dirs = list_class_dirs()
    class_to_idx, idx_to_class = build_label_map(class_dirs)
    num_classes = len(class_to_idx)

    # 3) Encode labels
    encoded_train = encode_labels(train_files, class_to_idx)
    encoded_val = encode_labels(val_files, class_to_idx)

    # 4) Build model (your build_cnn should already compile the model)
    model = build_cnn(num_classes)
    print("Model built ✅")

    # 5) Class weights for imbalance
    y_train = np.array([y for _, y in encoded_train], dtype=np.int32)
    class_weights = compute_class_weights(y_train, num_classes)
    print("Class weights ready ✅")

    # 6) Generators (no RAM explosion)
    batch_size = 32
    train_gen = DataGen(encoded_train, batch_size=batch_size, image_size=IMAGE_SIZE, shuffle=True)
    val_gen = DataGen(encoded_val, batch_size=batch_size, image_size=IMAGE_SIZE, shuffle=False)

    # 7) Train + save best model
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint("outputs/mvp_model.keras", monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\n✅ MVP training complete.")
    print("Best model saved at: outputs/mvp_model.keras")

    # ---------- SINGLE IMAGE PREDICTION (MVP DEMO) ----------

    # Pick ONE image path manually (any image from dataset)
    img_path = train_files[0][0]  # using first training image

    # Load trained model
    model = keras.models.load_model("outputs/mvp_model.keras")

    # Preprocess image
    img = preprocess_image(img_path, image_size=IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)  # (1, 128, 128, 3)

    # Predict
    probs = model.predict(img, verbose=0)[0]

    pred_idx = np.argmax(probs)
    confidence = np.max(probs)

    pred_class = idx_to_class[pred_idx]

    print("\n=== SINGLE IMAGE PREDICTION ===")
    print("Image path:", img_path)
    print("Predicted class:", pred_class)
    print("Confidence:", round(float(confidence) * 100, 2), "%")



if __name__ == "__main__":
    main()
