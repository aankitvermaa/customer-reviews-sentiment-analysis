# Streamlit UI for the sentiment classifier
# Run from project root: streamlit run app/app.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.predict import SentimentPredictor

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")


@st.cache_resource
def get_predictor() -> SentimentPredictor:
    return SentimentPredictor()


st.title("Sentiment Analysis")
st.caption("Classifies a product review as Positive or Negative (TF-IDF + linear classifier trained on Amazon reviews).")

try:
    predictor = get_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

review = st.text_area(
    "Enter a review:",
    height=160,
    placeholder="The product quality is excellent and delivery was very fast.",
)

if st.button("Analyze Sentiment", type="primary"):
    try:
        result = predictor.predict(review)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        if result["label_id"] == 1:
            st.success(f"Prediction: {result['label'].upper()}")
        else:
            st.error(f"Prediction: {result['label'].upper()}")
        label = "Confidence" if result["calibrated"] else "Confidence (approximate score)"
        st.metric(label, f"{result['confidence'] * 100:.1f}%")
        st.progress(result["probability_positive"], text=f"P(positive) = {result['probability_positive']:.3f}")
