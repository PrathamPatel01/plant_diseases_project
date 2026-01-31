# from src.dataset_utils import dataset_summary
# from src.split_data import split_dataset
# from src.preprocess import preprocess_image

# def main():
#     # Step 2 summary (counts + imbalance)
#     dataset_summary()

#     # Step 4 split
#     train_files, val_files, test_files = split_dataset()
#     print("\nSplit sizes:")
#     print("Train:", len(train_files))
#     print("Val:", len(val_files))
#     print("Test:", len(test_files))

#     # Sanity preprocess test
#     sample_img, sample_label = train_files[0]
#     arr = preprocess_image(sample_img)
#     print("\nSanity preprocess check:")
#     print("Image shape:", arr.shape)
#     print("Pixel range:", arr.min(), arr.max())
#     print("Label:", sample_label)



# #sanity check. for label mapping
# from src.labels import build_label_map
# from src.dataset_utils import list_class_dirs

# class_dirs = list_class_dirs()
# class_to_idx, idx_to_class = build_label_map(class_dirs)

# print("First 5 mappings:")
# for k in list(class_to_idx.keys())[:5]:
#     print(k, "→", class_to_idx[k])

# print("Index 0 maps back to:", idx_to_class[0])




# if __name__ == "__main__":
#     main()
# src/main.py
import numpy as np

from src.dataset_utils import dataset_summary, list_class_dirs, make_minibatch
from src.split_data import split_dataset
from src.preprocess import preprocess_image
from src.labels import build_label_map
from src.encode_labels import encode_labels
from src.config import IMAGE_SIZE
from src.model import build_cnn


def main():
    # -------------------------
    # Step 2: Dataset summary
    # -------------------------
    dataset_summary()

    # -------------------------
    # Step 4: Train/Val/Test split
    # -------------------------
    train_files, val_files, test_files = split_dataset()
    print("\nSplit sizes:")
    print("Train:", len(train_files))
    print("Val:", len(val_files))
    print("Test:", len(test_files))

    # -------------------------
    # Sanity preprocess test
    # -------------------------
    sample_img, sample_label = train_files[0]
    arr = preprocess_image(sample_img)
    print("\nSanity preprocess check:")
    print("Image shape:", arr.shape)
    print("Pixel range:", float(arr.min()), float(arr.max()))
    print("Label:", sample_label)

    # -------------------------
    # Step 5.1: Label mapping
    # -------------------------
    class_dirs = list_class_dirs()
    class_to_idx, idx_to_class = build_label_map(class_dirs)

    print("\nFirst 5 mappings:")
    for k in list(class_to_idx.keys())[:5]:
        print(k, "→", class_to_idx[k])
    print("Index 0 maps back to:", idx_to_class[0])

    # -------------------------
    # Step 5.2: Encode labels
    # -------------------------
    encoded_train = encode_labels(train_files, class_to_idx)
    encoded_val = encode_labels(val_files, class_to_idx)

    print("\nOriginal:", train_files[0])
    print("Encoded:", encoded_train[0])

    # -------------------------
    # Step 5.3: Mini-batch sanity
    # -------------------------
    X_small, y_small = make_minibatch(encoded_train, batch_size=8, image_size=IMAGE_SIZE)
    print("\nMini-batch check:")
    print("X shape:", X_small.shape)
    print("y shape:", y_small.shape)
    print("y values:", y_small)
    print("X min/max:", float(X_small.min()), float(X_small.max()))

    # -------------------------
    # Step 5.4/5.6: Build model (compiled inside build_cnn)
    # -------------------------
    num_classes = len(class_to_idx)
    model = build_cnn(num_classes)
    print("\nCompiled successfully ✅")

    print("\nModel summary:")
    model.summary()

    # -------------------------
    # Step 5.7 + 5.8: Smoke test training + validation
    # -------------------------
    print("\nStarting smoke test training (1 epoch on small subset)...")

    # Train subset
    train_subset_size = 2000  # reduce to 500 if your laptop feels slow
    train_idx = np.random.choice(len(encoded_train), train_subset_size, replace=False)
    train_subset = [encoded_train[i] for i in train_idx]
    X_train_small, y_train_small = make_minibatch(
        train_subset, batch_size=train_subset_size, image_size=IMAGE_SIZE
    )

    # Val subset
    val_subset_size = 500
    val_idx = np.random.choice(len(encoded_val), val_subset_size, replace=False)
    val_subset = [encoded_val[i] for i in val_idx]
    X_val_small, y_val_small = make_minibatch(
        val_subset, batch_size=val_subset_size, image_size=IMAGE_SIZE
    )

    history = model.fit(
        X_train_small,
        y_train_small,
        validation_data=(X_val_small, y_val_small),
        epochs=1,
        batch_size=32,
        verbose=1
    )

    print("\nSmoke test done ✅")
    print("Train loss:", history.history["loss"][-1])
    print("Train accuracy:", history.history["accuracy"][-1])
    print("Val loss:", history.history["val_loss"][-1])
    print("Val accuracy:", history.history["val_accuracy"][-1])


if __name__ == "__main__":
    main()
