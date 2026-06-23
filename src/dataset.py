from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    ALLOWED_EXTENSIONS,
)


ImageRecord = Tuple[str, str]


def list_class_dirs(dataset_dir: Path = DATASET_DIR) -> List[Path]:
    """
    Return all class folders inside the dataset directory.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    return sorted(class_dirs, key=lambda p: p.name)


def count_images_per_class(class_dirs: List[Path]) -> Dict[str, int]:
    """
    Count valid images inside each class folder.
    """
    image_counts = {}

    for class_dir in class_dirs:
        images = [
            img
            for img in class_dir.iterdir()
            if img.is_file() and img.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        image_counts[class_dir.name] = len(images)

    return image_counts


def dataset_summary() -> Dict:
    """
    Return useful dataset statistics.
    This function does not need TensorFlow.
    """
    class_dirs = list_class_dirs()
    image_counts = count_images_per_class(class_dirs)

    total_classes = len(class_dirs)
    total_images = sum(image_counts.values())

    sorted_classes = sorted(
        image_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    if image_counts:
        max_count = max(image_counts.values())
        min_count = min(image_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
    else:
        imbalance_ratio = 0

    return {
        "dataset_path": str(DATASET_DIR),
        "total_classes": total_classes,
        "total_images": total_images,
        "image_counts": image_counts,
        "top_classes": sorted_classes[:5],
        "bottom_classes": sorted_classes[-5:],
        "imbalance_ratio": round(imbalance_ratio, 2),
    }


def collect_image_records() -> List[ImageRecord]:
    """
    Collect all image paths with their class names.

    Returns:
    [
        ("path/to/image1.jpg", "Apple___Apple_scab"),
        ("path/to/image2.jpg", "Tomato___healthy"),
    ]
    """
    records = []

    for class_dir in list_class_dirs():
        images = [
            img
            for img in class_dir.iterdir()
            if img.is_file() and img.suffix.lower() in ALLOWED_EXTENSIONS
        ]

        for img in images:
            records.append((str(img), class_dir.name))

    return records


def create_label_mapping(records: List[ImageRecord]) -> Dict[str, int]:
    """
    Create class-name to integer-label mapping.

    Example:
    {
        "Apple___Apple_scab": 0,
        "Apple___Black_rot": 1
    }
    """
    class_names = sorted(set(label for _, label in records))
    return {class_name: index for index, class_name in enumerate(class_names)}


def split_dataset(records: List[ImageRecord]):
    """
    Split dataset into train, validation, and test sets.

    Stratify keeps class distribution balanced across splits.
    """
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9

    paths = [path for path, _ in records]
    labels = [label for _, label in records]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    val_ratio_from_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_ratio_from_temp,
        random_state=RANDOM_SEED,
        stratify=temp_labels,
    )

    train_records = list(zip(train_paths, train_labels))
    val_records = list(zip(val_paths, val_labels))
    test_records = list(zip(test_paths, test_labels))

    return train_records, val_records, test_records


def preprocess_for_training(image_path, label):
    """
    Load image, resize it, and prepare it for MobileNetV2.

    TensorFlow is imported inside this function so normal dataset
    checks do not crash if TensorFlow has environment issues.
    """
    import tensorflow as tf

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # MobileNetV2 expects image values in [-1, 1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


def make_tf_dataset(
    records: List[ImageRecord],
    label_mapping: Dict[str, int],
    shuffle: bool = False,
):
    """
    Convert image paths and labels into a TensorFlow Dataset.

    TensorFlow is imported only here because this function is used
    during training, not during simple dataset inspection.
    """
    import tensorflow as tf

    image_paths = [path for path, _ in records]
    labels = [label_mapping[label] for _, label in records]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        preprocess_for_training,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def print_dataset_summary():
    """
    Print dataset overview in the terminal.
    """
    summary = dataset_summary()

    print("Dataset path:", summary["dataset_path"])
    print("Total classes:", summary["total_classes"])
    print("Total images:", summary["total_images"])

    print("\nTop 5 classes:")
    for cls, count in summary["top_classes"]:
        print(f"- {cls}: {count}")

    print("\nBottom 5 classes:")
    for cls, count in summary["bottom_classes"]:
        print(f"- {cls}: {count}")

    print("\nImbalance ratio:", summary["imbalance_ratio"])