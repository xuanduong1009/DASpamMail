from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "bilingual"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2

VECTORIZER_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "norm": "l2",
}

USE_ENGLISH_STOPWORDS = True
USE_VIETNAMESE_STOPWORDS = True
VIETNAMESE_STOPWORDS_PATH = DATA_DIR / "stopwords_vi.txt"

NB_PARAM_GRID = {
    "clf__alpha": [0.1, 0.5, 1.0],
}

SVM_PARAM_GRID = {
    "clf__C": [0.1, 1.0, 3.0, 10.0],
}

# --- Random Forest (Omotehinwa & Oyewola, Appl. Sci. 2023) ---
RF_PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__max_features": ["sqrt", "log2"],
}

# --- Logistic Regression (Si et al., arXiv 2402.15537) ---
LR_PARAM_GRID = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__solver": ["lbfgs", "liblinear"],
    "clf__max_iter": [1000],
}

# --- XGBoost (Mustapha et al., arXiv 2012.14430) ---
XGB_PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 6],
    "clf__learning_rate": [0.1, 0.3],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
}

EXPERIMENTS = [
    {"name": "word_1_2", "vectorizer": VECTORIZER_PARAMS},
    {
        "name": "char_3_5",
        "vectorizer": {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "min_df": 3,
            "max_df": 0.95,
            "sublinear_tf": True,
            "norm": "l2",
        },
    },
]

KEYWORD_BASELINE = [
    "free",
    "winner",
    "click",
    "urgent",
    "limited",
    "offer",
    "buy",
    "money",
    "credit",
    "viagra",
]
