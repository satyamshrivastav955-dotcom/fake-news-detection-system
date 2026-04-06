"""
Fake News Detection - Model Training Script
============================================
Trains a TF-IDF + Logistic Regression classifier on the ISOT dataset
(True.csv + Fake.csv) and saves the artifacts as model.pkl / vectorizer.pkl.

Usage:
    python train_model.py

Dataset:
    Place True.csv and Fake.csv inside backend/data/
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""

import os
import re
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection   import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import classification_report, accuracy_score

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
TRUE_CSV        = os.path.join(DATA_DIR, "True.csv")
FAKE_CSV        = os.path.join(DATA_DIR, "Fake.csv")
MODEL_OUT       = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_OUT  = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

TFIDF_MAX_FEAT  = 50_000      # vocabulary size
TFIDF_NGRAM     = (1, 2)      # unigrams + bigrams
TEST_SIZE       = 0.20
RANDOM_STATE    = 42


# ─── Text Cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip HTML, remove non-alpha characters."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Load & Prepare Data ──────────────────────────────────────────────────────
def load_data():
    print("📂  Loading dataset...")

    if not os.path.exists(TRUE_CSV) or not os.path.exists(FAKE_CSV):
        raise FileNotFoundError(
            f"Dataset CSVs not found in {DATA_DIR}.\n"
            "Download True.csv and Fake.csv from Kaggle and place them in backend/data/"
        )

    df_true = pd.read_csv(TRUE_CSV)
    df_fake = pd.read_csv(FAKE_CSV)

    df_true["label"] = 0   # 0 = Real
    df_fake["label"] = 1   # 1 = Fake

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Combine title + text for richer signal
    df["content"] = (df.get("title", "").fillna("") + " " +
                     df.get("text",  "").fillna(""))

    df = df[["content", "label"]].dropna()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"   Total samples  : {len(df):,}")
    print(f"   Real (0) count : {(df.label==0).sum():,}")
    print(f"   Fake (1) count : {(df.label==1).sum():,}")

    return df


# ─── Train ────────────────────────────────────────────────────────────────────
def train(df):
    print("\n🔤  Cleaning text...")
    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["label"]

    print("✂️   Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n🔢  Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEAT,
        ngram_range  = TFIDF_NGRAM,
        stop_words   = "english",
        sublinear_tf = True,          # apply log normalization to term frequency
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("\n🤖  Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter     = 1000,
        C            = 1.0,
        solver       = "lbfgs",
        multi_class  = "auto",
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    lr_model.fit(X_train_vec, y_train)

    # ── Evaluate ──
    y_pred = lr_model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n📊  Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print("\n" + classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    return lr_model, vectorizer


# ─── Save Artifacts ───────────────────────────────────────────────────────────
def save_artifacts(model, vectorizer):
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_OUT, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"💾  Saved model      → {MODEL_OUT}")
    print(f"💾  Saved vectorizer → {VECTORIZER_OUT}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df               = load_data()
    model, vectorizer = train(df)
    save_artifacts(model, vectorizer)
    print("\n✅  Training complete. You can now start the Flask server.")
