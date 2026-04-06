"""
Fake News Detection - Flask API Backend
========================================
Serves a trained ML model (TF-IDF + Logistic Regression) via REST API.
Endpoint: POST /predict
"""

import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

# Paths to saved model artifacts
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

# ─── Load Model & Vectorizer ──────────────────────────────────────────────────
def load_artifacts():
    """Load the trained model and TF-IDF vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model artifacts not found. Please run train_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    model, vectorizer = None, None


# ─── Helper Functions ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip HTML tags, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def get_top_words(text: str, n: int = 8) -> list:
    """
    Return the top-N TF-IDF weighted words from the input text.
    These act as a lightweight 'explanation' of the prediction.
    """
    if vectorizer is None:
        return []

    # Transform just this document
    tfidf_vec = vectorizer.transform([clean_text(text)])

    # Get feature names and their scores for this document
    feature_names = vectorizer.get_feature_names_out()
    scores        = tfidf_vec.toarray()[0]

    # Sort by score descending, pick top-N non-zero features
    top_indices = scores.argsort()[::-1]
    top_words   = []
    for idx in top_indices:
        if scores[idx] > 0 and len(top_words) < n:
            word = feature_names[idx]
            if len(word) > 2:          # skip very short tokens
                top_words.append(word)
    return top_words


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Health-check endpoint."""
    return jsonify({"status": "ok", "message": "Fake News Detection API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "text": "..." }
    Returns JSON: { "prediction": "Fake|Real", "confidence": 0.94,
                    "explanation": [...] }
    """
    # ── Validate request ──
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must include a 'text' field."}), 400

    raw_text = data["text"].strip()
    if len(raw_text) < 10:
        return jsonify({"error": "Text is too short. Please provide more content."}), 422

    # ── Preprocess & vectorize ──
    cleaned = clean_text(raw_text)
    X       = vectorizer.transform([cleaned])

    # ── Predict ──
    pred_label  = int(model.predict(X)[0])          # 0 = Real, 1 = Fake
    pred_proba  = model.predict_proba(X)[0]          # [P(Real), P(Fake)]

    label       = "Fake" if pred_label == 1 else "Real"
    confidence  = round(float(pred_proba[pred_label]), 4)   # probability of chosen class
    explanation = get_top_words(raw_text)

    return jsonify({
        "prediction":  label,
        "confidence":  confidence,
        "explanation": explanation
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
