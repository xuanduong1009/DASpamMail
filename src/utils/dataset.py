import hashlib
from pathlib import Path

import pandas as pd

from src.utils.email_parse import parse_email_bytes
from src.utils.io import read_bytes

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def load_emails(data_path, dedup=False, limit=None):
    path = Path(data_path)
    if path.is_dir():
        ham_dir = path / "ham"
        spam_dir = path / "spam"
        if ham_dir.exists() and spam_dir.exists():
            return _load_from_folders(path, dedup=dedup, limit=limit)
    if path.is_file() and path.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(path)
        return _load_from_frame(df)
    raise ValueError(f"Unsupported data path: {path}")


def _load_from_folders(base_dir, dedup=False, limit=None):
    records = []
    seen = set()
    for label_name, label in (("ham", 0), ("spam", 1)):
        folder = Path(base_dir) / label_name
        files = [p for p in folder.rglob("*") if p.is_file()]
        for path in tqdm(files, desc=f"load {label_name}"):
            raw = read_bytes(path)
            subject, body = parse_email_bytes(raw)
            text = f"{subject}\n{body}".strip()
            if not text:
                continue
            if dedup:
                digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
            records.append(
                {
                    "id": hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest(),
                    "label": label,
                    "label_name": label_name,
                    "text": text,
                    "path": str(path),
                }
            )
            if limit and len(records) >= limit:
                break
    return pd.DataFrame(records)


def _load_from_frame(df):
    if "text" in df.columns:
        text_col = "text"
    elif "raw_text" in df.columns:
        text_col = "raw_text"
    else:
        raise ValueError("CSV must contain a text or raw_text column")
    if "label" not in df.columns:
        raise ValueError("CSV must contain a label column")
    out = df.copy()
    out["text"] = out[text_col].astype(str)
    out["label"] = out["label"].astype(int)
    return out[["text", "label"]]
