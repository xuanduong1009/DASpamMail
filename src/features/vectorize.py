from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.config import (
    USE_ENGLISH_STOPWORDS,
    USE_VIETNAMESE_STOPWORDS,
    VECTORIZER_PARAMS,
    VIETNAMESE_STOPWORDS_PATH,
)
from src.utils.text_clean import clean_text


def load_vietnamese_stopwords(path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_stopwords():
    stopwords = set()
    if USE_ENGLISH_STOPWORDS:
        stopwords.update(ENGLISH_STOP_WORDS)
    if USE_VIETNAMESE_STOPWORDS:
        stopwords.update(load_vietnamese_stopwords(VIETNAMESE_STOPWORDS_PATH))
    return sorted(stopwords) if stopwords else None


def build_vectorizer(params=None) -> TfidfVectorizer:
    cfg = dict(VECTORIZER_PARAMS)
    if params:
        cfg.update(params)
    analyzer = cfg.get("analyzer", "word")
    if analyzer == "word" and "stop_words" not in cfg:
        stopwords = build_stopwords()
        if stopwords:
            cfg["stop_words"] = stopwords
    vectorizer_kwargs = {
        "preprocessor": clean_text,
        "lowercase": False,
        **cfg,
    }
    if analyzer == "word":
        vectorizer_kwargs["token_pattern"] = r"(?u)\b\w+\b"
    return TfidfVectorizer(**vectorizer_kwargs)
