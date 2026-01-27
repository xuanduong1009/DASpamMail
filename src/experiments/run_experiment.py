import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import DATASET_DIR, EXPERIMENTS, NB_PARAM_GRID, RESULTS_DIR, SVM_PARAM_GRID
from src.models.train import prepare_data, train_model


def main():
    parser = argparse.ArgumentParser(description="Run experiment grid for spam filter.")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument(
        "--out", type=str, default=str(Path(RESULTS_DIR) / "experiment_results.csv")
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _, X_train, X_test, y_train, y_test = prepare_data(args.data_dir)

    results = []
    for exp in EXPERIMENTS:
        vec_params = dict(exp["vectorizer"])
        for model_name in ("nb", "svm"):
            param_grid = NB_PARAM_GRID if model_name == "nb" else SVM_PARAM_GRID
            _, metrics = train_model(
                model_name,
                X_train,
                y_train,
                X_test,
                y_test,
                vectorizer_params=vec_params,
                param_grid=param_grid,
            )
            metrics["experiment"] = exp["name"]
            results.append(metrics)

    df = pd.DataFrame(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info("Saved experiment results to %s", out_path)


if __name__ == "__main__":
    main()
