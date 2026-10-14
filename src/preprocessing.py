# Preprocessing — text cleaning that can live inside a saved sklearn pipeline

import html
import re
from functools import lru_cache

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# download NLTK data if not already present
for _res in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "wordnet"):
    nltk.download(_res, quiet=True)

MODES = ("minimal", "stopwords", "stem", "lemma")

# I keep negation words even when removing stopwords — dropping "not" would flip sentiment
NEGATIONS = frozenset({"no", "not", "never", "nor", "neither", "cannot", "none", "nobody", "nothing", "nowhere"})
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_NON_TEXT_RE = re.compile(r"[^a-z0-9'\s]")
_SPACE_RE = re.compile(r"\s+")
_STEMMER = SnowballStemmer("english")
_LEMMATIZER = WordNetLemmatizer()

# mapping Penn Treebank POS tags to WordNet POS for the lemmatizer
_POS_MAP = {"J": "a", "V": "v", "R": "r"}  # adjective, verb, adverb; default = noun


@lru_cache(maxsize=200_000)
def _stem(token):
    return token if token in NEGATIONS else _STEMMER.stem(token)


@lru_cache(maxsize=200_000)
def _lemmatize(token, pos_tag_first_char):
    if token in NEGATIONS:
        return token
    wn_pos = _POS_MAP.get(pos_tag_first_char, "n")
    return _LEMMATIZER.lemmatize(token, pos=wn_pos)


def _expand_contractions(text):
    text = text.replace("can't", "can not").replace("won't", "will not").replace("cannot", "can not")
    return re.sub(r"n't\b", " not", text)


def clean_text(text, mode="minimal"):
    """Normalise one review.

    minimal   : unescape/strip HTML, drop URLs, lowercase, expand n't, strip punctuation.
    stopwords : minimal + stopword removal (negations are preserved).
    stem      : minimal + Snowball stemming (negations are preserved).
    lemma     : minimal + WordNet lemmatization using POS tags (negations preserved).
    """
    if mode not in MODES:
        raise ValueError(f"Unknown preprocessing mode '{mode}'. Choose from {MODES}.")
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = html.unescape(str(text))
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _expand_contractions(text.lower())
    text = _NON_TEXT_RE.sub(" ", text).replace("'", "")
    tokens = _SPACE_RE.sub(" ", text).strip().split()
    if mode == "stopwords":
        tokens = [t for t in tokens if t not in STOPWORDS]
    elif mode == "stem":
        tokens = [_stem(t) for t in tokens]
    elif mode == "lemma":
        # use NLTK POS tagger to get better lemmas (e.g. "running" + verb -> "run")
        tagged = pos_tag(tokens)
        tokens = [_lemmatize(tok, tag[0]) for tok, tag in tagged]
    return " ".join(tokens)


def get_pos_distribution(texts, sample_size=5000):
    """Get POS tag distribution from a sample of texts.
    
    Useful for understanding the linguistic structure of reviews -
    e.g. do negative reviews use more adjectives?
    """
    from collections import Counter
    pos_counts = Counter()
    for text in texts[:sample_size]:
        if not isinstance(text, str):
            continue
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        pos_counts.update(tag for _, tag in tagged)
    return dict(pos_counts.most_common(20))


class TextCleaner(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer wrapping `clean_text`."""

    def __init__(self, mode="minimal"):
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode not in MODES:
            raise ValueError(f"Unknown preprocessing mode '{self.mode}'.")
        return self

    def transform(self, X):
        return [clean_text(x, self.mode) for x in X]
