def encode_labels(file_label_list, class_to_idx):
    """
    Converts:
      [(img_path, 'ClassName'), ...]
    to:
      [(img_path, class_index), ...]
    """
    encoded = []

    for img_path, label_str in file_label_list:
        label_idx = class_to_idx[label_str]
        encoded.append((img_path, label_idx))

    return encoded
