import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.utils.io import ensure_dir


def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision_spam": float(precision),
        "recall_spam": float(recall),
        "f1_spam": float(f1),
        "accuracy": float(acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved spam filter model.")
    parser.add_argument("--model", type=str, default="models/best_model.joblib")
    parser.add_argument(
        "--test-csv", type=str, default=str(Path(PROCESSED_DIR) / "test.csv")
    )
    parser.add_argument(
        "--out", type=str, default=str(Path(RESULTS_DIR) / "eval_metrics.csv")
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.test_csv)
    X = df["text"].astype(str)
    y_true = df["label"].astype(int)
    y_pred = model.predict(X)

    metrics = compute_metrics(y_true, y_pred)
    metrics["model"] = Path(args.model).stem
    ordered = ["model", "precision_spam", "recall_spam", "f1_spam", "accuracy", "tn", "fp", "fn", "tp"]
    metrics = {key: (round(value, 6) if isinstance(value, float) else value) for key, value in metrics.items()}
    print(json.dumps(metrics, indent=2))

    ensure_dir(Path(RESULTS_DIR))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[metrics.get(col) for col in ordered]], columns=ordered).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
