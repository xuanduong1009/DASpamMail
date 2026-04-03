import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.config import PROCESSED_DIR, RESULTS_DIR
from src.utils.io import ensure_dir


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        return prob[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_metrics(y_true, y_pred, y_score=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0
    false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
    false_negative_rate = float(fn / (fn + tp)) if (fn + tp) else 0.0
    macro_f1 = float(
        precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    )
    weighted_f1 = float(
        precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    )
    roc_auc = None
    if y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = None
    return {
        "precision_ham": float(precision[0]),
        "recall_ham": float(recall[0]),
        "f1_ham": float(f1[0]),
        "support_ham": int(support[0]),
        "precision_spam": float(precision[1]),
        "recall_spam": float(recall[1]),
        "f1_spam": float(f1[1]),
        "support_spam": int(support[1]),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": float(balanced_acc),
        "specificity": specificity,
        "npv": npv,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "roc_auc": roc_auc,
        "accuracy": float(acc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_classification_report(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for key in ["ham", "spam", "macro avg", "weighted avg"]:
        row = report.get(key, {})
        rows.append(
            {
                "class": key,
                "precision": round(float(row.get("precision", 0.0)), 6),
                "recall": round(float(row.get("recall", 0.0)), 6),
                "f1_score": round(float(row.get("f1-score", 0.0)), 6),
                "support": int(row.get("support", 0)),
            }
        )
    rows.append(
        {
            "class": "accuracy",
            "precision": None,
            "recall": None,
            "f1_score": round(float(report.get("accuracy", 0.0)), 6),
            "support": int(len(y_true)),
        }
    )
    return pd.DataFrame(rows)


def build_error_analysis(df, y_true, y_pred, y_score=None):
    errors = df.copy()
    errors["true_label"] = y_true
    errors["predicted_label"] = y_pred
    errors["true_name"] = errors["true_label"].map({0: "ham", 1: "spam"})
    errors["predicted_name"] = errors["predicted_label"].map({0: "ham", 1: "spam"})
    if y_score is not None:
        errors["score"] = y_score
        errors["confidence"] = errors["score"].abs()
    else:
        errors["score"] = None
        errors["confidence"] = None
    errors["length_chars"] = errors["text"].astype(str).str.len()
    errors["length_words"] = errors["text"].astype(str).str.split().str.len()
    errors["text_preview"] = (
        errors["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
    )
    errors = errors[errors["true_label"] != errors["predicted_label"]].copy()
    if not errors.empty:
        errors = errors.sort_values(["confidence", "length_words"], ascending=[False, False])
    keep_cols = [
        col
        for col in [
            "id",
            "path",
            "true_name",
            "predicted_name",
            "score",
            "confidence",
            "length_chars",
            "length_words",
            "text_preview",
        ]
        if col in errors.columns
    ]
    if "score" in errors.columns:
        errors["score"] = errors["score"].round(6)
    if "confidence" in errors.columns:
        errors["confidence"] = errors["confidence"].round(6)
    return errors[keep_cols].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved spam filter model.")
    parser.add_argument("--model", type=str, default="models/best_model.joblib")
    parser.add_argument(
        "--test-csv", type=str, default=str(Path(PROCESSED_DIR) / "test.csv")
    )
    parser.add_argument(
        "--out", type=str, default=str(Path(RESULTS_DIR) / "eval_metrics.csv")
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default=str(Path(RESULTS_DIR) / "classification_report.csv"),
    )
    parser.add_argument(
        "--out-errors",
        type=str,
        default=str(Path(RESULTS_DIR) / "misclassified_examples.csv"),
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.test_csv)
    X = df["text"].astype(str)
    y_true = df["label"].astype(int)
    y_pred = model.predict(X)
    y_score = get_scores(model, X)

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["model"] = Path(args.model).stem
    ordered = [
        "model",
        "precision_ham",
        "recall_ham",
        "f1_ham",
        "support_ham",
        "precision_spam",
        "recall_spam",
        "f1_spam",
        "support_spam",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "specificity",
        "npv",
        "false_positive_rate",
        "false_negative_rate",
        "roc_auc",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    metrics = {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }
    print(json.dumps(metrics, indent=2))

    ensure_dir(Path(RESULTS_DIR))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[metrics.get(col) for col in ordered]], columns=ordered).to_csv(out_path, index=False)
    build_classification_report(y_true, y_pred).to_csv(Path(args.out_report), index=False)
    build_error_analysis(df, y_true, y_pred, y_score).to_csv(Path(args.out_errors), index=False)


if __name__ == "__main__":
    main()
