import pytest

from src.preprocessing import TextCleaner, clean_text


def test_html_and_entities_removed():
    assert clean_text("<br />Great &amp; cheap<br/>") == "great cheap"


def test_whitespace_and_special_characters():
    assert clean_text("  Love   it!!!   #1 @@ ") == "love it 1"


def test_urls_removed():
    assert "http" not in clean_text("see https://example.com/x now")


def test_contractions_expand_to_negation():
    assert clean_text("It doesn't work and I can't use it") == "it does not work and i can not use it"


def test_stopword_mode_keeps_negation():
    out = clean_text("This is not good", mode="stopwords")
    assert "not" in out.split() and "good" in out.split()


def test_stem_mode_keeps_negation():
    out = clean_text("never working", mode="stem")
    assert out.split()[0] == "never"


def test_lemma_mode_reduces_words():
    # lemmatizer should turn "running" -> "run", "better" stays as is (needs POS)
    out = clean_text("the cats are running quickly", mode="lemma")
    assert "cat" in out.split()  # plural -> singular
    assert "run" in out.split()  # gerund -> base form


def test_lemma_mode_keeps_negation():
    out = clean_text("not working properly", mode="lemma")
    assert "not" in out.split()


@pytest.mark.parametrize("value", [None, float("nan"), "", "   "])
def test_empty_values_return_empty_string(value):
    assert clean_text(value) == ""


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        clean_text("x", mode="bogus")


def test_transformer_output_length():
    assert len(TextCleaner().fit_transform(["a b", "c d", None])) == 3

