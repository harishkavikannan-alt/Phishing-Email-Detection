"""
Phishing Email Detection Model - Training Script
=================================================
This script trains a Machine Learning model to classify emails as
either 'Phishing' or 'Safe' using:
  1. TF-IDF (Term Frequency - Inverse Document Frequency) for text vectorization
  2. Logistic Regression for binary classification

Author: Antigravity Pair Programmer
Project: Phishing Email Detection Model
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------
DATASET_PATH = "emails.csv"
MODEL_OUTPUT_PATH = "phishing_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.25  # 25% for testing, 75% for training


def print_banner(title: str):
    """Utility function to print styled section headers in terminal."""
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads and validates the email dataset from CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset '{file_path}' not found! Please ensure emails.csv exists."
        )

    df = pd.read_csv(file_path)

    # Basic data validation
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Drop missing values if any
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def train_and_evaluate():
    """
    Main training and evaluation pipeline.
    """
    print_banner("1. LOADING DATASET")
    df = load_dataset(DATASET_PATH)
    print(f"Total email samples loaded: {len(df)}")
    print("\nClass Distribution:")
    for label, count in df["label"].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  - {label:<10}: {count} samples ({percentage:.1f}%)")

    # Features and Target
    X = df["text"]
    y = df["label"]

    print_banner("2. SPLITTING DATA INTO TRAIN & TEST SETS")
    # Stratified split ensures equal proportion of Phishing and Safe in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Training set size : {len(X_train)} samples")
    print(f"Testing set size  : {len(X_test)} samples")

    print_banner("3. BUILDING TF-IDF & LOGISTIC REGRESSION PIPELINE")
    # We construct a Scikit-Learn Pipeline combining vectorization and classification
    # - TfidfVectorizer: Converts raw text strings into numerical n-gram importance scores
    # - LogisticRegression: Learns linear decision boundary to classify emails
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),        # Consider unigrams and bigrams (e.g. "urgent", "verify account")
                stop_words="english",       # Remove common English stop words
                lowercase=True,            # Normalize all text to lowercase
                max_features=2500          # Limit maximum vocabulary size
            )
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                C=1.0                      # Standard L2 regularization
            )
        )
    ])

    print("Fitting model to training data...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    print_banner("4. MODEL EVALUATION ON UNSEEN TEST DATA")
    y_pred = pipeline.predict(X_test)

    # Compute key metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="Phishing")
    recall = recall_score(y_test, y_pred, pos_label="Phishing")
    f1 = f1_score(y_test, y_pred, pos_label="Phishing")
    cm = confusion_matrix(y_test, y_pred, labels=["Safe", "Phishing"])

    # Display Metrics Summary Table
    print(f"Accuracy  : {accuracy * 100:.2f}%  (Overall correctly predicted emails)")
    print(f"Precision : {precision * 100:.2f}%  (When predicted Phishing, how often is it correct?)")
    print(f"Recall    : {recall * 100:.2f}%  (Out of all actual Phishing emails, how many were caught?)")
    print(f"F1-Score  : {f1 * 100:.2f}%  (Harmonic mean of precision and recall)")

    print("\nConfusion Matrix:")
    print(f"{'':>12} | {'Predicted Safe':>15} | {'Predicted Phishing':>18}")
    print("-" * 52)
    print(f"{'Actual Safe':>12} | {cm[0][0]:>15} | {cm[0][1]:>18}")
    print(f"{'Actual Phishing':>12} | {cm[1][0]:>15} | {cm[1][1]:>18}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print_banner("5. SAVING TRAINED MODEL")
    # Package pipeline along with metrics metadata for the Streamlit dashboard
    model_payload = {
        "pipeline": pipeline,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "classes": list(pipeline.classes_)
    }

    joblib.dump(model_payload, MODEL_OUTPUT_PATH)
    print(f"Model and metrics saved successfully to: '{MODEL_OUTPUT_PATH}'")

    print_banner("CRITICAL EDUCATIONAL DISCLAIMER")
    print(
        "PLEASE NOTE:\n"
        "This project is trained on an educational sample dataset.\n"
        "While it effectively demonstrates the mechanics of text vectorization (TF-IDF)\n"
        "and supervised classification (Logistic Regression), high scores on a small\n"
        "sample dataset DO NOT prove real-world accuracy or readiness for production.\n\n"
        "Real-world phishing attacks often employ:\n"
        "  - Zero-day spear phishing techniques\n"
        "  - Obfuscated URLs, look-alike domains (typosquatting), and QR codes (quishing)\n"
        "  - Header spoofing bypassing SPF/DKIM/DMARC\n"
        "  - Evasive language generated by modern LLMs\n\n"
        "Production email security systems utilize multi-layered defenses including\n"
        "deep domain reputation scoring, behavioral sandboxing, and millions of samples."
    )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train_and_evaluate()
