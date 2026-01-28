def build_label_map(class_dirs):
    # Extract class names from folder paths
    class_names = [p.name for p in class_dirs]

    # Sort to ensure stable ordering
    class_names = sorted(class_names)

    # String → number
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Number → string (for decoding predictions later)
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    return class_to_idx, idx_to_class
