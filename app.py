import json
import random
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    DATASET_DIR,
    ALLOWED_EXTENSIONS,
    HISTORY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    LABELS_PATH,
)


# ============================================================
# Page Setup
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
)

px.defaults.template = "plotly_dark"


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f5132, #198754);
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(25, 135, 84, 0.22);
    }

    .hero h1 {
        font-size: 2.7rem;
        font-weight: 850;
        margin-bottom: 0.4rem;
        letter-spacing: -0.04em;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.92;
        margin-bottom: 0;
    }

    .card {
        padding: 1.4rem;
        border-radius: 18px;
        background: #161b22;
        border: 1px solid #30363d;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .card h3, .card h4 {
        color: #f8fafc;
        margin-bottom: 0.6rem;
    }

    .card p {
        color: #c9d1d9;
        line-height: 1.6;
    }

    .small-text {
        color: #8b949e;
        font-size: 0.92rem;
    }

    .status-ready {
        color: #3fb950;
        font-weight: 800;
        font-size: 2rem;
    }

    .status-missing {
        color: #f2cc60;
        font-weight: 800;
        font-size: 2rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helper Functions
# ============================================================

@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def get_summary():
    """
    Deployment-safe dataset summary.

    Streamlit Cloud does not contain the full PlantVillage dataset.
    So the app first reads reports/dataset_summary.json.
    If that file is missing, it uses a small fallback summary.
    """
    summary_path = Path("reports/dataset_summary.json")

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "dataset_path": "PlantVillage dataset not included in deployed demo",
        "total_classes": 38,
        "total_images": 54305,
        "image_counts": {
            "Orange___Haunglongbing_(Citrus_greening)": 5507,
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 5357,
            "Soybean___healthy": 5090,
            "Peach___Bacterial_spot": 2297,
            "Tomato___Bacterial_spot": 2127,
            "Tomato___Tomato_mosaic_virus": 373,
            "Raspberry___healthy": 371,
            "Peach___healthy": 360,
            "Apple___Cedar_apple_rust": 275,
            "Potato___healthy": 152,
        },
        "top_classes": [
            ["Orange___Haunglongbing_(Citrus_greening)", 5507],
            ["Tomato___Tomato_Yellow_Leaf_Curl_Virus", 5357],
            ["Soybean___healthy", 5090],
            ["Peach___Bacterial_spot", 2297],
            ["Tomato___Bacterial_spot", 2127],
        ],
        "bottom_classes": [
            ["Tomato___Tomato_mosaic_virus", 373],
            ["Raspberry___healthy", 371],
            ["Peach___healthy", 360],
            ["Apple___Cedar_apple_rust", 275],
            ["Potato___healthy", 152],
        ],
        "imbalance_ratio": 36.23,
    }


def clean_class_name(class_name: str) -> str:
    return class_name.replace("___", " - ").replace("_", " ")


def split_class_name(class_name: str):
    if "___" in class_name:
        plant, condition = class_name.split("___", 1)
    else:
        plant, condition = "Unknown", class_name

    plant = plant.replace("_", " ").title()
    condition = condition.replace("_", " ").title()

    return plant, condition


def build_class_dataframe(summary: dict) -> pd.DataFrame:
    rows = []

    for class_name, count in summary["image_counts"].items():
        plant, condition = split_class_name(class_name)

        rows.append(
            {
                "Class": class_name,
                "Readable Class": clean_class_name(class_name),
                "Plant": plant,
                "Condition": condition,
                "Type": "Healthy" if "healthy" in condition.lower() else "Disease",
                "Images": count,
            }
        )

    return pd.DataFrame(rows).sort_values("Images", ascending=False)


def get_sample_images(class_name: str, limit: int = 6):
    """
    Local-only sample images.

    On Streamlit Cloud, data_raw/ is not deployed, so this returns [].
    """
    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        return []

    images = [
        image_path
        for image_path in class_dir.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not images:
        return []

    return random.sample(images, min(limit, len(images)))


def make_chart_clean(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ============================================================
# Load Data
# ============================================================

summary = get_summary()
history_data = load_json(HISTORY_PATH)
metrics_data = load_json(METRICS_PATH)
class_df = build_class_dataframe(summary)


# ============================================================
# Header
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


tabs = st.tabs(
    [
        "Dashboard",
        "Dataset Insights",
        "Model Performance",
    ]
)


# ============================================================
# Tab 1: Dashboard
# ============================================================

with tabs[0]:
    st.subheader("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", f"{summary['total_images']:,}")

    with col2:
        st.metric("Classes", summary["total_classes"])

    with col3:
        st.metric("Imbalance Ratio", summary["imbalance_ratio"])

    with col4:
        if metrics_data and "accuracy" in metrics_data:
            st.metric("Test Accuracy", f"{metrics_data['accuracy'] * 100:.2f}%")
        else:
            st.metric("Test Accuracy", "Pending")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown(
            """
            <div class="card">
                <h3>Project Summary</h3>
                <p>
                    This project detects plant diseases from leaf images using deep learning.
                    It uses the PlantVillage dataset and a MobileNetV2 transfer learning model.
                </p>
                <p>
                    The pipeline includes dataset analysis, stratified splitting, image preprocessing,
                    class-imbalance handling, model training, and final performance evaluation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        model_ready = MODEL_PATH.exists() and LABELS_PATH.exists()

        status_html = (
            '<div class="status-ready">Ready</div>'
            if model_ready
            else '<div class="status-missing">Demo Mode</div>'
        )

        st.markdown(
            f"""
            <div class="card">
                <h3>System Status</h3>
                <p class="small-text">Dashboard deployment status</p>
                {status_html}
                <p class="small-text">
                    The deployed dashboard uses saved reports and does not include
                    the full dataset or model file.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ML Pipeline")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown(
            """
            <div class="card">
                <h4>1. Dataset</h4>
                <p>Leaf images are loaded from class-wise PlantVillage folders.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            """
            <div class="card">
                <h4>2. Preprocessing</h4>
                <p>Images are resized to 224×224 and normalized for MobileNetV2.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            """
            <div class="card">
                <h4>3. Training</h4>
                <p>Transfer learning is used with class weights for imbalance handling.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with p4:
        st.markdown(
            """
            <div class="card">
                <h4>4. Evaluation</h4>
                <p>Accuracy, precision, recall, F1-score, and per-class metrics are saved.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Top 5 Largest Classes")

    top_df = pd.DataFrame(summary["top_classes"], columns=["Class", "Images"])
    top_df["Class"] = top_df["Class"].apply(clean_class_name)

    fig = px.bar(
        top_df,
        x="Images",
        y="Class",
        orientation="h",
        text="Images",
        title="Largest dataset classes",
    )
    fig.update_layout(
        height=420,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(make_chart_clean(fig), width="stretch")


# ============================================================
# Tab 2: Dataset Insights
# ============================================================

with tabs[1]:
    st.subheader("Dataset Insights")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Images", f"{summary['total_images']:,}")

    with c2:
        st.metric("Total Classes", summary["total_classes"])

    with c3:
        st.metric("Disease Classes", len(class_df[class_df["Type"] == "Disease"]))

    with c4:
        st.metric("Healthy Classes", len(class_df[class_df["Type"] == "Healthy"]))

    f1, f2, f3 = st.columns(3)

    with f1:
        selected_plant = st.selectbox(
            "Plant",
            ["All"] + sorted(class_df["Plant"].unique().tolist()),
        )

    with f2:
        selected_type = st.selectbox(
            "Class Type",
            ["All", "Disease", "Healthy"],
        )

    filtered_df = class_df.copy()

    if selected_plant != "All":
        filtered_df = filtered_df[filtered_df["Plant"] == selected_plant]

    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["Type"] == selected_type]

    with f3:
        st.metric("Visible Classes", len(filtered_df))

    if filtered_df.empty:
        st.warning("No classes match the selected filters.")
    else:
        st.markdown("### Class Distribution")

        fig = px.bar(
            filtered_df,
            x="Images",
            y="Readable Class",
            color="Type",
            orientation="h",
            text="Images",
            title="Images per class",
        )
        fig.update_layout(
            height=max(500, min(950, len(filtered_df) * 30)),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(make_chart_clean(fig), width="stretch")

        st.markdown("### Class Details")

        selected_class_readable = st.selectbox(
            "Choose class",
            filtered_df["Readable Class"].tolist(),
        )

        selected_class = filtered_df[
            filtered_df["Readable Class"] == selected_class_readable
        ]["Class"].iloc[0]

        plant, condition = split_class_name(selected_class)

        i1, i2, i3 = st.columns(3)

        with i1:
            st.metric("Plant", plant)

        with i2:
            st.metric("Condition", condition)

        with i3:
            st.metric("Images", f"{summary['image_counts'][selected_class]:,}")

        st.markdown("### Sample Images")

        sample_images = get_sample_images(selected_class, limit=6)

        if sample_images:
            image_cols = st.columns(3)

            for index, image_path in enumerate(sample_images):
                with image_cols[index % 3]:
                    image = Image.open(image_path).convert("RGB")
                    st.image(
                        image,
                        caption=image_path.name,
                        width="stretch",
                    )
        else:
            st.info(
                "Sample images are available locally only. The deployed demo does not "
                "include the full PlantVillage dataset because it is too large for GitHub."
            )

        st.markdown("### Dataset Table")

        st.dataframe(
            filtered_df[
                ["Plant", "Condition", "Type", "Images", "Readable Class"]
            ],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# Tab 3: Model Performance
# ============================================================

with tabs[2]:
    st.subheader("Model Performance")

    if history_data is None:
        st.warning("Training history not found. Add reports/history.json to show trends.")
    else:
        history_df = pd.DataFrame(history_data["history"])
        history_df["Epoch"] = range(1, len(history_df) + 1)

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Train Accuracy",
                f"{history_df['accuracy'].iloc[-1] * 100:.2f}%",
            )

        with m2:
            st.metric(
                "Validation Accuracy",
                f"{history_df['val_accuracy'].iloc[-1] * 100:.2f}%",
            )

        with m3:
            if metrics_data and "accuracy" in metrics_data:
                st.metric(
                    "Test Accuracy",
                    f"{metrics_data['accuracy'] * 100:.2f}%",
                )
            else:
                st.metric("Test Accuracy", "Pending")

        with m4:
            st.metric("Epochs", len(history_df))

        st.markdown("### Accuracy Trend")

        acc_df = history_df[["Epoch", "accuracy", "val_accuracy"]].melt(
            id_vars="Epoch",
            var_name="Metric",
            value_name="Accuracy",
        )

        acc_df["Metric"] = acc_df["Metric"].replace(
            {
                "accuracy": "Training Accuracy",
                "val_accuracy": "Validation Accuracy",
            }
        )

        fig_acc = px.line(
            acc_df,
            x="Epoch",
            y="Accuracy",
            color="Metric",
            markers=True,
            title="Training vs validation accuracy",
        )
        fig_acc.update_yaxes(tickformat=".0%")
        st.plotly_chart(make_chart_clean(fig_acc), width="stretch")

        st.markdown("### Loss Trend")

        loss_df = history_df[["Epoch", "loss", "val_loss"]].melt(
            id_vars="Epoch",
            var_name="Metric",
            value_name="Loss",
        )

        loss_df["Metric"] = loss_df["Metric"].replace(
            {
                "loss": "Training Loss",
                "val_loss": "Validation Loss",
            }
        )

        fig_loss = px.line(
            loss_df,
            x="Epoch",
            y="Loss",
            color="Metric",
            markers=True,
            title="Training vs validation loss",
        )
        st.plotly_chart(make_chart_clean(fig_loss), width="stretch")

    st.markdown("### Classification Report")

    if metrics_data is None:
        st.warning("Metrics file not found. Add reports/metrics.json to show evaluation.")
    else:
        macro_avg = metrics_data.get("macro avg", {})
        weighted_avg = metrics_data.get("weighted avg", {})

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric("Macro Precision", f"{macro_avg.get('precision', 0) * 100:.2f}%")

        with r2:
            st.metric("Macro Recall", f"{macro_avg.get('recall', 0) * 100:.2f}%")

        with r3:
            st.metric("Macro F1", f"{macro_avg.get('f1-score', 0) * 100:.2f}%")

        with r4:
            st.metric("Weighted F1", f"{weighted_avg.get('f1-score', 0) * 100:.2f}%")

        rows = []

        for class_name, values in metrics_data.items():
            if isinstance(values, dict) and "precision" in values:
                rows.append(
                    {
                        "Class": clean_class_name(class_name),
                        "Precision": values["precision"],
                        "Recall": values["recall"],
                        "F1-score": values["f1-score"],
                        "Support": values["support"],
                    }
                )

        metrics_df = pd.DataFrame(rows)

        if not metrics_df.empty:
            st.markdown("### Lowest F1-score Classes")

            low_f1_df = metrics_df.sort_values("F1-score").head(10)

            fig_f1 = px.bar(
                low_f1_df,
                x="F1-score",
                y="Class",
                orientation="h",
                text="F1-score",
                title="Classes with lowest F1-score",
            )
            fig_f1.update_layout(
                height=520,
                yaxis={"categoryorder": "total descending"},
            )
            st.plotly_chart(make_chart_clean(fig_f1), width="stretch")

            st.markdown("### Full Per-Class Metrics")

            st.dataframe(
                metrics_df.sort_values("F1-score"),
                width="stretch",
                hide_index=True,
            )

    st.info(
        """
        Technical summary: MobileNetV2 transfer learning was used with ImageNet weights.
        The dataset was split using stratified sampling, and class weights were applied
        during training to reduce the impact of dataset imbalance.
        """
    )