import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    DATASET_DIR,
    INTERIM_DIR,
    KEYWORD_BASELINE,
    MODELS_DIR,
    NB_PARAM_GRID,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SVM_PARAM_GRID,
    TEST_SIZE,
    VECTORIZER_PARAMS,
)
from src.features.vectorize import build_vectorizer
from src.models.evaluate import compute_metrics
from src.utils.dataset import load_emails
from src.utils.io import ensure_dir


def prepare_data(data_dir, dedup=False, limit=None):
    df = load_emails(data_dir, dedup=dedup, limit=limit)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return df, X_train, X_test, y_train, y_test


def run_baselines(X_train, y_train, X_test, y_test):
    results = []

    majority_label = int(y_train.mode().iloc[0])
    y_pred = [majority_label] * len(y_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "majority"
    results.append(metrics)

    keywords = [k.lower() for k in KEYWORD_BASELINE]
    y_pred = []
    for text in X_test:
        text_l = str(text).lower()
        is_spam = any(k in text_l for k in keywords)
        y_pred.append(1 if is_spam else 0)
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "keyword"
    results.append(metrics)

    return results


def train_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    vectorizer_params=None,
    param_grid=None,
):
    if model_name == "nb":
        clf = MultinomialNB()
        grid = NB_PARAM_GRID if param_grid is None else param_grid
    elif model_name == "svm":
        clf = LinearSVC()
        grid = SVM_PARAM_GRID if param_grid is None else param_grid
    else:
        raise ValueError("Unknown model name")

    params = VECTORIZER_PARAMS if vectorizer_params is None else vectorizer_params
    pipeline = Pipeline(
        [
            ("tfidf", build_vectorizer(params)),
            ("clf", clf),
        ]
    )

    search = GridSearchCV(
        pipeline,
        grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = model_name
    metrics["best_params"] = json.dumps(search.best_params_)
    metrics["cv_best_f1"] = float(search.best_score_)

    return best, metrics


def main():
    parser = argparse.ArgumentParser(description="Train spam filter models.")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--dedup", action="store_true", help="Deduplicate by content hash")
    parser.add_argument("--limit", type=int, default=None, help="Limit emails for quick tests")
    parser.add_argument(
        "--out-metrics", type=str, default=str(Path(RESULTS_DIR) / "metrics.csv")
    )
    parser.add_argument("--out-model-dir", type=str, default=str(MODELS_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ensure_dir(INTERIM_DIR)
    ensure_dir(PROCESSED_DIR)
    ensure_dir(Path(args.out_model_dir))
    ensure_dir(RESULTS_DIR)

    df, X_train, X_test, y_train, y_test = prepare_data(
        args.data_dir, dedup=args.dedup, limit=args.limit
    )

    raw_path = Path(INTERIM_DIR) / "emails_raw.csv"
    df.to_csv(raw_path, index=False)

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})
    train_df.to_csv(Path(PROCESSED_DIR) / "train.csv", index=False)
    test_df.to_csv(Path(PROCESSED_DIR) / "test.csv", index=False)

    results = []
    results.extend(run_baselines(X_train, y_train, X_test, y_test))

    best_models = []
    for name in ("nb", "svm"):
        model, metrics = train_model(name, X_train, y_train, X_test, y_test)
        model_path = Path(args.out_model_dir) / f"{name}_best.joblib"
        joblib.dump(model, model_path)
        best_models.append((metrics["f1_spam"], model))
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_metrics, index=False)

    best_models.sort(key=lambda x: x[0], reverse=True)
    if best_models:
        best_model = best_models[0][1]
        best_path = Path(args.out_model_dir) / "best_model.joblib"
        joblib.dump(best_model, best_path)
        logging.info("Saved best model to %s", best_path)

    logging.info("Saved metrics to %s", args.out_metrics)


if __name__ == "__main__":
    main()
