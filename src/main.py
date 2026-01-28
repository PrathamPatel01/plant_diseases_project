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

from src.dataset_utils import dataset_summary, list_class_dirs, make_minibatch
from src.split_data import split_dataset
from src.preprocess import preprocess_image
from src.labels import build_label_map
from src.encode_labels import encode_labels
from src.config import IMAGE_SIZE


def main():
    # Step 2 summary (counts + imbalance)
    dataset_summary()

    # Step 4 split
    train_files, val_files, test_files = split_dataset()
    print("\nSplit sizes:")
    print("Train:", len(train_files))
    print("Val:", len(val_files))
    print("Test:", len(test_files))

    # Sanity preprocess test
    sample_img, sample_label = train_files[0]
    arr = preprocess_image(sample_img)
    print("\nSanity preprocess check:")
    print("Image shape:", arr.shape)
    print("Pixel range:", arr.min(), arr.max())
    print("Label:", sample_label)

    # Step 5.1: label mapping
    class_dirs = list_class_dirs()
    class_to_idx, idx_to_class = build_label_map(class_dirs)

    print("\nFirst 5 mappings:")
    for k in list(class_to_idx.keys())[:5]:
        print(k, "→", class_to_idx[k])
    print("Index 0 maps back to:", idx_to_class[0])

    # Step 5.2: encode labels
    encoded_train = encode_labels(train_files, class_to_idx)
    print("\nOriginal:", train_files[0])
    print("Encoded:", encoded_train[0])

    # Step 5.3: mini-batch (NOW encoded_train exists ✅)
    X_small, y_small = make_minibatch(encoded_train, batch_size=8, image_size=IMAGE_SIZE)
    print("\nMini-batch check:")
    print("X shape:", X_small.shape)
    print("y shape:", y_small.shape)
    print("y values:", y_small)
    print("X min/max:", X_small.min(), X_small.max())


if __name__ == "__main__":
    main()
