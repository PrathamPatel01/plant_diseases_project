# 🌿 Plant Disease Classification using CNN

A deep learning project for automatic plant disease identification using the PlantVillage dataset and TensorFlow/Keras.

The system learns visual patterns from leaf images and classifies them into healthy and diseased categories across multiple plant species.

---

## 📌 Overview

Plant diseases can significantly affect crop yield and food production. Early detection allows farmers to take preventive actions before diseases spread.

This project builds a Convolutional Neural Network (CNN) that classifies leaf images into disease categories using the **PlantVillage dataset**.

The entire pipeline was implemented from scratch, including:

* Dataset analysis
* Data preprocessing
* Label encoding
* Train/validation/test splitting
* Batch generation
* CNN model construction
* Training with class weighting
* Model checkpointing
* Inference and prediction

---

# Dataset

Dataset used:

**PlantVillage**

The dataset contains:

* **38 classes**
* Healthy and diseased leaves
* Multiple crop species
* Images stored in class-specific folders

Example:

```
PlantVillage/
│
├── Apple___Black_rot
├── Apple___healthy
├── Tomato___Early_blight
├── Tomato___Late_blight
├── Potato___healthy
...
```

---

# Exploratory Data Analysis

The dataset statistics include:

* Number of classes
* Total number of images
* Top 5 largest classes
* Bottom 5 smallest classes
* Class imbalance ratio

A class distribution histogram is generated using Matplotlib.

Example analysis:

```python
imbalance_ratio = max_count / min_count
```

This helps understand whether some classes dominate the dataset.

---

# Project Structure

```
plant_disease_project/
│
├── data_raw/
│     └── PlantVillage/
│
├── outputs/
│     └── mvp_model.keras
│
├── src/
│     ├── config.py
│     ├── dataset_utils.py
│     ├── preprocess.py
│     ├── labels.py
│     ├── encode_labels.py
│     ├── split_data.py
│     ├── minibatch.py
│     ├── model.py
│     ├── train.py
│     └── pretty_predict.py
│
└── README.md
```

---

# Preprocessing Pipeline

Each image undergoes:

### 1. RGB conversion

```python
Image.open(img_path).convert("RGB")
```

### 2. Resize

All images are resized to:

```
128 × 128
```

### 3. Normalization

Pixel values are scaled from:

```
0–255
```

to

```
0–1
```

using:

```python
arr = np.asarray(img, dtype=np.float32) / 255.0
```

Output tensor shape:

```
(H, W, 3)
```

---

# Dataset Split

The dataset is split using:

| Set        | Ratio |
| ---------- | ----- |
| Train      | 70%   |
| Validation | 15%   |
| Test       | 15%   |

Random seed:

```python
RANDOM_SEED = 42
```

ensures reproducibility.

---

# Label Encoding

Folder names are converted into integer labels.

Example:

```
Apple___Black_rot → 0
Apple___healthy → 1
Tomato___Early_blight → 2
...
```

Two mappings are maintained:

```python
class_to_idx
idx_to_class
```

---

# Mini-Batch Generator

A custom data generator based on:

```python
tensorflow.keras.utils.Sequence
```

loads images batch-by-batch instead of storing the entire dataset in RAM.

Advantages:

* Memory efficient
* Supports shuffling
* Scalable to larger datasets

Batch shape:

```
(batch_size, 128, 128, 3)
```

Default:

```python
batch_size = 32
```

---

# CNN Architecture

The model consists of three convolution blocks.

## Block 1

```python
Conv2D(32)
MaxPooling2D()
```

## Block 2

```python
Conv2D(64)
MaxPooling2D()
```

## Block 3

```python
Conv2D(128)
MaxPooling2D()
```

---

## Classification Head

```python
GlobalAveragePooling2D()

Dense(128)

Dropout(0.3)

Dense(num_classes, activation="softmax")
```

---

# Loss Function

Sparse categorical cross entropy:

```python
loss="sparse_categorical_crossentropy"
```

---

# Optimizer

Adam optimizer:

```python
optimizer="adam"
```

---

# Training Strategy

The model is trained using:

```python
model.fit()
```

with:

* Training generator
* Validation generator
* Class weights
* Early stopping
* Model checkpointing

---

## Class Imbalance Handling

Class weights are computed using:

```python
weight = total_samples / (num_classes × class_count)
```

This prevents majority classes from dominating training.

---

## Early Stopping

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```

Stops training when validation loss stops improving.

---

## Model Checkpoint

```python
ModelCheckpoint(
    "outputs/mvp_model.keras",
    save_best_only=True
)
```

Only the best-performing model is saved.

---

# Inference Pipeline

For a new image:

```
Input image
        ↓
RGB conversion
        ↓
Resize (128×128)
        ↓
Normalization
        ↓
Expand dimensions
        ↓
CNN model
        ↓
Softmax probabilities
        ↓
Argmax
        ↓
Predicted disease class
```

---

# Prediction Output

Example:

```
Predicted class:

Tomato___Late_blight

Confidence:

98.74%
```

The system also displays:

### Top-5 predictions

```
1. Tomato___Late_blight
2. Tomato___Early_blight
3. Tomato___healthy
4. Potato___Late_blight
5. Pepper___healthy
```

along with confidence scores.

---

# Technologies Used

* Python
* NumPy
* TensorFlow
* Keras
* Pillow
* Matplotlib

---

# Future Improvements

* Data augmentation
* Transfer learning with EfficientNet
* ResNet50
* MobileNetV3 deployment
* Grad-CAM visualization
* Streamlit web application
* ONNX/TFLite export
* Real-time mobile inference
* Disease treatment recommendations

---

# Key Learnings

Through this project, the following concepts were implemented from scratch:

* Computer Vision
* Image preprocessing
* Convolutional Neural Networks
* Batch generation
* Label encoding
* Handling class imbalance
* Early stopping
* Model checkpointing
* Multi-class classification
* TensorFlow/Keras training pipeline
* Model inference and probability decoding

