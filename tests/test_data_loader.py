import pandas as pd
import pytest

from src.data_loader import LABEL_COL, TEXT_COL, load_dataset, validate


def test_validate_drops_bad_rows():
    df = pd.DataFrame(
        {
            TEXT_COL: ["good", "good", "", None, "bad", "ok"],
            LABEL_COL: [1, 1, 1, 0, 0, None],
        }
    )
    clean, report = validate(df)
    assert list(clean[TEXT_COL]) == ["good", "bad"]
    assert report["duplicate_reviews"] == 1
    assert report["empty_reviews"] == 1
    assert report["rows_clean"] == 2


def test_fasttext_format(tmp_path):
    f = tmp_path / "reviews.txt"
    f.write_text("__label__2 Great: loved it\n__label__1 Awful: broke fast\n")
    df, _ = load_dataset(f)
    assert sorted(df[LABEL_COL]) == [0, 1]


def test_fine_food_format(tmp_path):
    f = tmp_path / "Reviews.csv"
    pd.DataFrame({"Score": [5, 1, 3], "Text": ["yum", "yuck", "meh"]}).to_csv(f, index=False)
    df, _ = load_dataset(f)
    assert len(df) == 2


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_dataset(tmp_path / "nope.csv")
