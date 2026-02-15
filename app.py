"""
ML Assignment 2 - Streamlit App
Interactive app to upload test data (CSV), select a model, and view metrics + confusion matrix.
Uses Score.csv dataset (Credit Score classification: Good / Standard / Poor).
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "saved")
TARGET_COL = "Credit_Score"


def load_models_and_artifacts():
    """Load preprocessor, feature names, label encoder, class names, and model list."""
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    input_fn_path = os.path.join(MODEL_DIR, "input_feature_names.joblib")
    if not os.path.isfile(preprocessor_path) or not os.path.isfile(input_fn_path):
        return None, None, None, None, {}
    preprocessor = joblib.load(preprocessor_path)
    input_feature_names = joblib.load(input_fn_path)
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    class_names = joblib.load(os.path.join(MODEL_DIR, "class_names.joblib"))

    display_names = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)",
    ]
    name_to_file = {}
    for d in display_names:
        fname = d.replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        path = os.path.join(MODEL_DIR, fname)
        if os.path.isfile(path):
            name_to_file[d] = fname
    return preprocessor, input_feature_names, label_encoder, class_names, name_to_file


def load_single_model(display_name, name_to_file):
    """Load one model."""
    fname = name_to_file.get(display_name)
    if not fname:
        return None
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(path):
        return None
    data = joblib.load(path)
    return data["model"]


def get_target_column(df):
    """Return target column name."""
    if TARGET_COL in df.columns:
        return TARGET_COL
    return df.columns[-1]


def run_inference(model, X_transformed):
    """Run prediction and probability."""
    y_pred = model.predict(X_transformed)
    y_proba = model.predict_proba(X_transformed) if hasattr(model, "predict_proba") else None
    return y_pred, y_proba


def main():
    st.set_page_config(page_title="ML Assignment 2 - Credit Score Classification", layout="wide")
    st.title("Credit Score Classification - Model Demo")
    st.markdown("Upload a **test** CSV with the same columns as **Score.csv** (including `Credit_Score`). Select a model to see metrics and confusion matrix.")

    preprocessor, input_feature_names, label_encoder, class_names, name_to_file = load_models_and_artifacts()
    if not name_to_file:
        st.error(
            "Models not found. Run `python model/train.py` from the project root (with Score.csv in the folder), then restart the app."
        )
        return

    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to evaluate a model on your test data.")
        metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
        if os.path.isfile(metrics_path):
            st.subheader("Pre-computed metrics (on default train/test split)")
            df_metrics = pd.read_csv(metrics_path)
            st.dataframe(df_metrics, use_container_width=True)
        return

    df = pd.read_csv(uploaded_file)
    target_col = get_target_column(df)
    if target_col not in df.columns:
        st.error(f"Expected target column '{TARGET_COL}'. Found: {list(df.columns)}")
        return

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    missing = set(input_feature_names) - set(X_df.columns)
    if missing:
        st.error(f"Missing columns in CSV: {list(missing)}. Required: {input_feature_names}")
        return
    X_df = X_df[input_feature_names]

    try:
        X_transformed = preprocessor.transform(X_df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    try:
        y_true_encoded = label_encoder.transform(y_raw.astype(str))
    except Exception:
        st.error(f"Target values must be in {list(label_encoder.classes_)}. Found unique: {y_raw.astype(str).unique().tolist()[:20]}")
        return

    selected_model = st.selectbox("Select model", list(name_to_file.keys()))
    model = load_single_model(selected_model, name_to_file)
    if model is None:
        st.error("Could not load the selected model.")
        return

    y_pred, y_proba = run_inference(model, X_transformed)

    # Metrics (multi-class)
    acc = accuracy_score(y_true_encoded, y_pred)
    prec = precision_score(y_true_encoded, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true_encoded, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true_encoded, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true_encoded, y_pred)
    if y_proba is not None and len(np.unique(y_true_encoded)) >= 2:
        try:
            auc = roc_auc_score(y_true_encoded, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc = 0.0
    else:
        auc = 0.0

    st.subheader("Evaluation metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall", f"{rec:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true_encoded, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close()

    st.subheader("Classification report")
    report = classification_report(
        y_true_encoded, y_pred,
        target_names=class_names,
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)


if __name__ == "__main__":
    main()
