import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MONEY_RE = re.compile(r"[\$€£]\s?\d+(?:[\.,]\d+)?")
NUM_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{40,}={0,2})")
TAG_RE = re.compile(r"<[^>]+>")
REPEATED_PUNCT_RE = re.compile(r"([!$?])\1{2,}")
HEADER_LINE_RE = re.compile(r"^(from|to|subject|cc|bcc|date):.*$", re.IGNORECASE | re.MULTILINE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HEADER_LINE_RE.sub(" ", text)
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = BASE64_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1\1\1", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()
