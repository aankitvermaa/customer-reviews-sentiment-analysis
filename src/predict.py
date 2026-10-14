# Inference wrapper — loads the saved pipeline and runs predictions

import math
from pathlib import Path

import joblib

import src.preprocessing  # noqa: F401  (ensures TextCleaner is importable when unpickling)
from src.data_loader import LABEL_NAMES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_pipeline.pkl"
MAX_CHARS = 10_000


class SentimentPredictor:
    """Loads the saved pipeline and returns label + confidence for a review."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{path}'. Run `python -m src.train --data <dataset>` first."
            )
        self.pipeline = joblib.load(path)

    @staticmethod
    def validate_input(text):
        if not isinstance(text, str):
            raise ValueError("Review must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("Review text is empty.")
        return text[:MAX_CHARS]

    def predict(self, text):
        text = self.validate_input(text)
        if hasattr(self.pipeline, "predict_proba"):
            p_pos = float(self.pipeline.predict_proba([text])[0][1])
            calibrated = True
        else:
            # LinearSVC doesn't have predict_proba, so use sigmoid on the decision function
            # not a real probability but good enough for a confidence score
            p_pos = 1.0 / (1.0 + math.exp(-float(self.pipeline.decision_function([text])[0])))
            calibrated = False
        label = 1 if p_pos >= 0.5 else 0
        return {
            "label": LABEL_NAMES[label],
            "label_id": label,
            "confidence": p_pos if label == 1 else 1.0 - p_pos,
            "probability_positive": p_pos,
            "calibrated": calibrated,
        }
