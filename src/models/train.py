import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.config import (
    DATASET_DIR,
    EXPERIMENTS,
    INTERIM_DIR,
    KEYWORD_BASELINE,
    LR_PARAM_GRID,
    MODELS_DIR,
    NB_PARAM_GRID,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PARAM_GRID,
    SVM_PARAM_GRID,
    TEST_SIZE,
    VECTORIZER_PARAMS,
    XGB_PARAM_GRID,
)
from src.features.vectorize import build_vectorizer
from src.models.evaluate import compute_metrics, get_scores
from src.utils.dataset import load_emails
from src.utils.io import ensure_dir

METRIC_COLUMNS = [
    "model",
    "precision_spam",
    "recall_spam",
    "f1_spam",
    "accuracy",
    "tn",
    "fp",
    "fn",
    "tp",
    "cv_best_f1",
    "best_params",
]


def prepare_data(data_dir, dedup=False, limit=None, return_frames=False):
    df = load_emails(data_dir, dedup=dedup, limit=limit)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    X_train = train_df["text"]
    X_test = test_df["text"]
    y_train = train_df["label"]
    y_test = test_df["label"]
    if return_frames:
        return df, train_df, test_df, X_train, X_test, y_train, y_test
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


def run_search(clf, grid, vectorizer_params, X_train, y_train):
    pipeline = Pipeline(
        [
            ("tfidf", build_vectorizer(vectorizer_params)),
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
    return search


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
        grid = SVM_PARAM_GRID if param_grid is None else param_grid
        experiments = (
            [{"name": "custom", "vectorizer": vectorizer_params}]
            if vectorizer_params is not None
            else EXPERIMENTS
        )

        best_search = None
        best_experiment = None
        for experiment in experiments:
            search = run_search(
                LinearSVC(),
                grid,
                experiment["vectorizer"],
                X_train,
                y_train,
            )
            if best_search is None or search.best_score_ > best_search.best_score_:
                best_search = search
                best_experiment = experiment["name"]

        best = best_search.best_estimator_
        y_pred = best.predict(X_test)

        y_score = get_scores(best, X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics["model"] = model_name
        metrics["best_params"] = json.dumps(
            {
                "experiment": best_experiment,
                **best_search.best_params_,
            }
        )
        metrics["cv_best_f1"] = float(best_search.best_score_)

        return best, metrics

    # Omotehinwa & Oyewola, Appl. Sci. 2023 — Random Forest + GridSearchCV
    elif model_name == "rf":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        grid = RF_PARAM_GRID if param_grid is None else param_grid

    # Si et al., arXiv:2402.15537 — Logistic Regression (TF-IDF benchmark)
    elif model_name == "lr":
        clf = LogisticRegression(random_state=RANDOM_STATE)
        grid = LR_PARAM_GRID if param_grid is None else param_grid

    # Mustapha et al., arXiv:2012.14430 — XGBoost
    elif model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed in the current environment.")
        clf = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
        grid = XGB_PARAM_GRID if param_grid is None else param_grid

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    params = VECTORIZER_PARAMS if vectorizer_params is None else vectorizer_params
    search = run_search(clf, grid, params, X_train, y_train)
    best = search.best_estimator_
    y_pred = best.predict(X_test)

    y_score = get_scores(best, X_test)
    metrics = compute_metrics(y_test, y_pred, y_score)
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

    df, train_df, test_df, X_train, X_test, y_train, y_test = prepare_data(
        args.data_dir, dedup=args.dedup, limit=args.limit, return_frames=True
    )

    raw_path = Path(INTERIM_DIR) / "emails_raw.csv"
    df.to_csv(raw_path, index=False)

    train_df.to_csv(Path(PROCESSED_DIR) / "train.csv", index=False)
    test_df.to_csv(Path(PROCESSED_DIR) / "test.csv", index=False)

    results = []
    results.extend(run_baselines(X_train, y_train, X_test, y_test))

    best_models = []
    model_names = ["nb", "svm", "rf", "lr"]
    if XGBClassifier is not None:
        model_names.append("xgb")
    else:
        logging.warning("xgboost is not installed; skipping xgb benchmark.")

    for name in model_names:
        logging.info("Training model: %s", name)
        model, metrics = train_model(name, X_train, y_train, X_test, y_test)
        model_path = Path(args.out_model_dir) / f"{name}_best.joblib"
        joblib.dump(model, model_path)
        best_models.append((metrics["f1_spam"], model))
        results.append(metrics)
        logging.info(
            "%s → F1: %.4f | Acc: %.4f | best_params: %s",
            name,
            metrics["f1_spam"],
            metrics["accuracy"],
            metrics["best_params"],
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        float_cols = results_df.select_dtypes(include="float").columns
        results_df[float_cols] = results_df[float_cols].round(6)
        results_df = results_df.sort_values(
            ["f1_spam", "accuracy"],
            ascending=[False, False],
            kind="stable",
        )
        ordered_cols = [col for col in METRIC_COLUMNS if col in results_df.columns]
        extra_cols = [col for col in results_df.columns if col not in ordered_cols]
        results_df = results_df[ordered_cols + extra_cols]
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
