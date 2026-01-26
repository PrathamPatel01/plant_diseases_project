from pathlib import Path
from .config import DATASET_DIR, ALLOWED_EXTENSIONS

def list_class_dirs(dataset_dir: Path = DATASET_DIR):
    """Return a list of class directories (folders) inside dataset_dir."""
    return [p for p in dataset_dir.iterdir() if p.is_dir()]

def count_images_per_class(class_dirs):
    """
    Returns:
      image_counts: dict {class_name: count}
    """
    image_counts = {}
    for class_dir in class_dirs:
        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        image_counts[class_dir.name] = len(files)
    return image_counts

def dataset_summary():
    """
    Prints:
      - total classes
      - total images
      - top/bottom classes
      - imbalance ratio
    """
    class_dirs = list_class_dirs()
    image_counts = count_images_per_class(class_dirs)

    total_classes = len(class_dirs)
    total_images = sum(image_counts.values())

    sorted_classes = sorted(image_counts.items(), key=lambda x: x[1], reverse=True)

    max_count = max(image_counts.values())
    min_count = min(image_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    print("Dataset path:", DATASET_DIR)
    print("Total classes:", total_classes)
    print("Total images:", total_images)

    print("\nTop 5 classes:")
    for cls, c in sorted_classes[:5]:
        print(f"- {cls}: {c}")

    print("\nBottom 5 classes:")
    for cls, c in sorted_classes[-5:]:
        print(f"- {cls}: {c}")

    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.2f}")

    return class_dirs, image_counts, sorted_classes
