import joblib
import pytest

from src.features import build_final_pipeline
from src.predict import SentimentPredictor

TEXTS = ["great product love it", "excellent quality works well", "wonderful and fast",
         "terrible waste of money", "broke after one day awful", "not good at all never again"] * 5
LABELS = [1, 1, 1, 0, 0, 0] * 5
VEC = dict(ngram_range=(1, 2), max_features=1000, min_df=1, max_df=1.0, sublinear_tf=True)


@pytest.fixture(params=["logistic_regression", "linear_svm"])
def predictor(request, tmp_path):
    pipe = build_final_pipeline("minimal", request.param, VEC, {})
    pipe.fit(TEXTS, LABELS)
    path = tmp_path / "model.pkl"
    joblib.dump(pipe, path)
    return SentimentPredictor(path)


def test_valid_prediction(predictor):
    out = predictor.predict("excellent product, love it")
    assert out["label"] == "Positive"
    assert 0.5 <= out["confidence"] <= 1.0


def test_special_characters_and_long_input(predictor):
    assert predictor.predict("!!! <b>great</b> $$$ 😀")["label"] in {"Positive", "Negative"}
    assert predictor.predict("great " * 20000)["label"] in {"Positive", "Negative"}


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_invalid_input_raises(predictor, bad):
    with pytest.raises(ValueError):
        predictor.predict(bad)


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SentimentPredictor(tmp_path / "missing.pkl")
