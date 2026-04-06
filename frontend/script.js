/**
 * TruthLens — Fake News Detector
 * Frontend Logic (script.js)
 * ─────────────────────────────────────────────────────────────
 * Responsibilities:
 *   1. Send user text to the Flask /predict API via fetch()
 *   2. Render the result card with prediction, confidence, keywords
 *   3. Manage loading / error / empty-input states
 *   4. Live character counter and clear button
 */

"use strict";

// ── Config ─────────────────────────────────────────────────────────────────
/** Base URL of the Flask backend. Change to your deployed URL on Render. */
const API_BASE = "http://127.0.0.1:5000";

// ── DOM references ──────────────────────────────────────────────────────────
const newsInput       = document.getElementById("news-input");
const charCounter     = document.getElementById("input-hint");
const clearBtn        = document.getElementById("clear-btn");
const analyzeBtn      = document.getElementById("analyze-btn");
const btnLabel        = document.getElementById("btn-label");
const btnSpinner      = document.getElementById("btn-spinner");
const errorBanner     = document.getElementById("error-banner");
const errorText       = document.getElementById("error-text");
const resultCard      = document.getElementById("result-card");
const verdictBanner   = document.getElementById("verdict-banner");
const verdictIcon     = document.getElementById("verdict-icon");
const verdictLabel    = document.getElementById("verdict-label");
const confidenceBar   = document.getElementById("confidence-bar");
const confidenceValue = document.getElementById("confidence-value");
const keywordChips    = document.getElementById("keyword-chips");
const analyzeAgainBtn = document.getElementById("analyze-again-btn");

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Show or hide the error banner with a given message.
 * @param {string|null} message - Pass null or empty to hide banner.
 */
function setError(message) {
  if (message) {
    errorText.textContent = message;
    errorBanner.hidden    = false;
  } else {
    errorBanner.hidden = true;
    errorText.textContent = "";
  }
}

/**
 * Toggle the loading state on the Analyze button.
 * @param {boolean} loading
 */
function setLoading(loading) {
  analyzeBtn.disabled = loading;
  btnLabel.textContent = loading ? "Analyzing…" : "Analyze Article";
  btnSpinner.hidden    = !loading;
}

/** Smoothly reveal the result card. */
function showResultCard() {
  resultCard.hidden = false;
  // Scroll the result card into view after a tiny delay
  setTimeout(() => {
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 80);
}

/** Build keyword chip elements from an array of words. */
function renderKeywords(words) {
  keywordChips.innerHTML = "";                    // clear old chips

  if (!words || words.length === 0) {
    const empty = document.createElement("span");
    empty.className   = "keyword-chip";
    empty.textContent = "No key terms identified";
    keywordChips.appendChild(empty);
    return;
  }

  words.forEach((word) => {
    const chip = document.createElement("span");
    chip.className   = "keyword-chip";
    chip.textContent = word;
    keywordChips.appendChild(chip);
  });
}

// ── Core — API call ─────────────────────────────────────────────────────────

/**
 * Send text to the /predict endpoint and render the response.
 */
async function analyzeText() {
  const text = newsInput.value.trim();

  // ── Client-side validation ──
  if (!text) {
    setError("Please enter a news article or headline before analyzing.");
    newsInput.focus();
    return;
  }
  if (text.length < 10) {
    setError("Your input is too short. Please provide more context (at least 10 characters).");
    newsInput.focus();
    return;
  }

  // ── Reset previous state ──
  setError(null);
  resultCard.hidden = true;
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    // ── HTTP-level error handling ──
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server responded with status ${response.status}.`);
    }

    const data = await response.json();

    // ── Validate response shape ──
    if (!data.prediction || data.confidence === undefined) {
      throw new Error("Unexpected response from the server. Please try again.");
    }

    // ── Render result ──
    renderResult(data);

  } catch (err) {
    // Network errors, CORS issues, JSON parse errors, etc.
    if (err.name === "TypeError" && err.message.includes("fetch")) {
      setError(
        "Could not connect to the backend. Make sure the Flask server is running on port 5000."
      );
    } else {
      setError(err.message || "An unexpected error occurred. Please try again.");
    }
  } finally {
    setLoading(false);
  }
}

/**
 * Populate and display the result card.
 * @param {{ prediction: string, confidence: number, explanation: string[] }} data
 */
function renderResult(data) {
  const isFake       = data.prediction.toLowerCase() === "fake";
  const pct          = Math.round(data.confidence * 100);

  // ── Verdict banner ──
  verdictBanner.className  = `verdict-banner ${isFake ? "is-fake" : "is-real"}`;
  verdictIcon.textContent  = isFake ? "🚨" : "✅";
  verdictLabel.textContent = isFake ? "Fake News" : "Real News";

  // ── Confidence bar ──
  // Reset width first (so animation triggers even on re-analysis)
  confidenceBar.style.width = "0%";
  confidenceBar.setAttribute("aria-valuenow", pct);
  confidenceBar.className =
    `confidence-bar ${isFake ? "bar-fake" : "bar-real"}`;

  // Trigger animation in next frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceBar.style.width = `${pct}%`;
    });
  });

  confidenceValue.textContent = `${pct}% confident`;
  confidenceValue.style.color = isFake
    ? "var(--color-fake)"
    : "var(--color-real)";

  // ── Keywords ──
  renderKeywords(data.explanation);

  // ── Show card ──
  showResultCard();
}

// ── Event Listeners ─────────────────────────────────────────────────────────

/** Live character counter. */
newsInput.addEventListener("input", () => {
  const count = newsInput.value.length;
  charCounter.textContent = `${count.toLocaleString()} character${count !== 1 ? "s" : ""}`;
});

/** Trigger analysis on button click. */
analyzeBtn.addEventListener("click", analyzeText);

/** Also trigger on Ctrl+Enter / Cmd+Enter for power-users. */
newsInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    analyzeText();
  }
});

/** Clear everything and start fresh. */
clearBtn.addEventListener("click", () => {
  newsInput.value          = "";
  charCounter.textContent  = "0 characters";
  setError(null);
  resultCard.hidden = true;
  newsInput.focus();
});

/** "Analyze Another Article" scrolls back and focuses the textarea. */
analyzeAgainBtn.addEventListener("click", () => {
  resultCard.hidden = true;
  newsInput.focus();
  newsInput.scrollIntoView({ behavior: "smooth", block: "center" });
});
