import argparse
from pathlib import Path

import joblib

from src.utils.email_parse import parse_email_bytes
from src.utils.io import read_bytes


def build_input_text(email_file, text):
    if email_file:
        raw = read_bytes(Path(email_file))
        subject, body = parse_email_bytes(raw)
        return f"{subject}\n{body}".strip()
    if text:
        return text
    raise ValueError("Provide --email-file or --text")


def get_score(model, X):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        return float(prob[1])
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return float(score[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="Predict spam or ham for a new email.")
    parser.add_argument("--model", type=str, default="models/best_model.joblib")
    parser.add_argument("--email-file", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    model = joblib.load(args.model)
    text = build_input_text(args.email_file, args.text)
    label = int(model.predict([text])[0])
    label_name = "spam" if label == 1 else "ham"
    score = get_score(model, [text])

    if score is None:
        print(f"prediction: {label_name}")
    else:
        print(f"prediction: {label_name} (score={score:.4f})")


if __name__ == "__main__":
    main()
