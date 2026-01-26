import numpy as np
from PIL import Image
from .config import IMAGE_SIZE

def preprocess_image(img_path, image_size=IMAGE_SIZE) -> np.ndarray:
    """
    Loads an image, converts to RGB, resizes, and normalizes to [0,1].
    Returns: numpy array of shape (H, W, 3) with float32 dtype.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr
