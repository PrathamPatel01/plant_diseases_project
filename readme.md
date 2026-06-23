# 🌿 Plant Disease Intelligence

A deep learning project for **plant disease classification** using the **PlantVillage dataset**, **TensorFlow/Keras**, and **MobileNetV2 transfer learning**.

The project trains a model to classify leaf images into healthy or diseased plant categories and provides a clean **Streamlit dashboard** to explore dataset insights and model performance.

---

## 📌 Key Features

* 38 plant disease classes
* 54,305 leaf images
* MobileNetV2 transfer learning
* Stratified train/validation/test split
* Class-weighted training for imbalance handling
* Model evaluation with accuracy, precision, recall, and F1-score
* Streamlit dashboard with dataset and performance insights

---

## 📊 Result

Training for 3 epochs produced:

| Metric              |  Value |
| ------------------- | -----: |
| Validation Accuracy | 91.12% |
| Test Accuracy       | 91.66% |
| Classes             |     38 |
| Images              | 54,305 |

---

## 🏗️ Project Structure

```text
plant_diseases_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── data_raw/
│   └── PlantVillage/
│
├── models/
│   ├── plant_disease_model.keras
│   └── labels.json
│
└── reports/
    ├── history.json
    ├── metrics.json
    └── confusion_matrix.json
```

---

## ⚙️ Setup

Create environment:

```bash
conda create -n plant-disease python=3.11 -y
conda activate plant-disease
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train Model

```bash
python -m src.train
```

This saves the trained model and evaluation files inside:

```text
models/
reports/
```

---

## 🖥️ Run Dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

---

## 📦 Requirements

For Apple Silicon Mac:

```text
tensorflow-macos==2.16.2
tensorflow-metal
streamlit
scikit-learn
pillow
numpy
pandas
matplotlib
plotly
```

For Windows/Linux, replace `tensorflow-macos` and `tensorflow-metal` with `tensorflow`.

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* Scikit-learn
* Pandas
* Plotly
* Streamlit


