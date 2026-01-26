from sklearn.model_selection import train_test_split
from .config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, ALLOWED_EXTENSIONS
from .dataset_utils import list_class_dirs

def split_dataset():
    """
    Returns:
      train_files, val_files, test_files
    Each is a list of tuples: (img_path, class_name)
    """
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9, "Split ratios must sum to 1."

    class_dirs = list_class_dirs()

    train_files, val_files, test_files = [], [], []

    # For each class, split its images independently (keeps class distribution)
    for class_dir in class_dirs:
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
        ]

        # 70% train, 30% temp
        train, temp = train_test_split(
            images,
            test_size=(1.0 - TRAIN_RATIO),
            random_state=RANDOM_SEED
        )

        # temp -> val/test split equally (15/15)
        val_size_from_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # = 0.5 when 15/15
        val, test = train_test_split(
            temp,
            test_size=(1.0 - val_size_from_temp),
            random_state=RANDOM_SEED
        )

        train_files.extend([(img, class_dir.name) for img in train])
        val_files.extend([(img, class_dir.name) for img in val])
        test_files.extend([(img, class_dir.name) for img in test])

    return train_files, val_files, test_files
