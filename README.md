# 🔍 TruthLens — Fake News Detector

> An AI-powered web application that analyzes news article authenticity using **TF-IDF + Logistic Regression** machine learning.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📸 Features

- **Instant analysis** — Paste any headline or article and get results in under 1 second
- **Confidence score** — See how certain the model is (0–100%)
- **Explanation keywords** — Top TF-IDF words that influenced the prediction
- **Premium UI** — Dark, professional SaaS-grade interface
- **REST API** — Clean JSON API ready for integration

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.9+ installed
- pip

### 1. Clone & enter project
```bash
cd fake_news_detection
```

### 2. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Generate the demo model (runs immediately, no dataset needed)
```bash
python create_demo_model.py
```

### 4. Start the Flask server
```bash
python app.py
```
The API will be live at `http://127.0.0.1:5000`

### 5. Open the frontend
Open `frontend/index.html` in your browser (double-click or use a Live Server extension).

---

## 🗃️ Using the Real Dataset (Recommended for production)

For maximum accuracy (~98%), train on the full ISOT Fake News Dataset:

1. Download **True.csv** and **Fake.csv** from:
   - Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. Place both files inside `backend/data/`

3. Run the training script:
   ```bash
   cd backend
   python train_model.py
   ```
   Training takes ~2–5 minutes on a standard laptop.

4. Training output example:
   ```
   Test Accuracy : 0.9872 (98.72%)
   Real: precision 0.99  recall 0.98
   Fake: precision 0.98  recall 0.99
   ```

---

## 📁 Project Structure

```
fake_news_detection/
├── .gitignore
├── render.yaml               # Render deployment config
├── README.md
│
├── frontend/
│   ├── index.html            # Main UI page
│   ├── style.css             # Premium dark theme styles
│   └── script.js             # Fetch API + result rendering
│
└── backend/
    ├── app.py                # Flask REST API
    ├── train_model.py        # Full ISOT dataset trainer
    ├── create_demo_model.py  # Instant demo model (no dataset needed)
    ├── requirements.txt      # Python dependencies
    ├── Procfile              # Render/Heroku process file
    ├── model.pkl             # Saved model (generated)
    ├── vectorizer.pkl        # Saved TF-IDF (generated)
    └── data/                 # (gitignored) Place CSVs here
        ├── True.csv
        └── Fake.csv
```

---

## 🌐 API Reference

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok", "message": "Fake News Detection API is running." }
```

---

### `POST /predict`
Analyze a text for authenticity.

**Request:**
```json
{ "text": "Scientists at NASA confirmed water ice near lunar south pole." }
```

**Response:**
```json
{
  "prediction":  "Real",
  "confidence":  0.9312,
  "explanation": ["scientists", "nasa", "confirmed", "lunar", "water"]
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | `"Fake"` \| `"Real"` | Model verdict |
| `confidence` | `float` (0–1) | Probability of the predicted class |
| `explanation` | `string[]` | Top TF-IDF keywords driving the decision |

---

## 🤖 ML Pipeline

```
User Input
    │
    ▼
Text Cleaning         (lowercase, strip HTML, remove non-alpha)
    │
    ▼
TF-IDF Vectorizer     (50,000 features, unigrams + bigrams)
    │
    ▼
Logistic Regression   (trained on 44,000+ ISOT articles)
    │
    ▼
Prediction + Confidence + Top Keywords
```

---

## ☁️ Deployment

### Backend → Render (free tier)
1. Push the repo to GitHub (the actual highly accurate ML models will be pushed via Git LFS!)
2. Go to [render.com](https://render.com), create a new Web Service
3. Point to your repo, set **Root Directory** to `backend`
4. Build command: `pip install -r requirements.txt` (the model is already there!)
5. Start command: `gunicorn app:app`

### Frontend → Vercel / Netlify
1. Update `API_BASE` in `frontend/script.js` to your Render URL
2. Drag-and-drop the `frontend/` folder to Netlify or Vercel

---

## 📄 License
MIT — free to use for educational and commercial purposes.
