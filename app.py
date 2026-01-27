from __future__ import annotations

from pathlib import Path
import random

import joblib
import pandas as pd
import streamlit as st

from src.utils.email_parse import parse_email_bytes


MODEL_PATH = Path("models") / "best_model.joblib"
DATASET_DIR = Path("data") / "bilingual"
METRICS_PATH = Path("reports") / "results" / "eval_metrics.csv"
MODELS_TABLE_PATH = Path("reports") / "results" / "metrics.csv"
TRAIN_CSV_PATH = Path("data") / "processed" / "train.csv"
TEST_CSV_PATH = Path("data") / "processed" / "test.csv"


st.set_page_config(
    page_title="Enron Spam Filter Demo",
    page_icon="Email",
    layout="wide",
)

THEMES = {
    "Sunrise": {
        "bg1": "#fff7ed",
        "bg2": "#fff1f2",
        "bg3": "#fef3c7",
        "card": "rgba(255, 255, 255, 0.78)",
        "border": "rgba(252, 211, 77, 0.45)",
        "accent": "#f97316",
        "accent2": "#0ea5a4",
        "text": "#111827",
        "muted": "#6b7280",
    },
    "Ocean": {
        "bg1": "#e0f2fe",
        "bg2": "#ecfeff",
        "bg3": "#dbeafe",
        "card": "rgba(255, 255, 255, 0.78)",
        "border": "rgba(59, 130, 246, 0.2)",
        "accent": "#0284c7",
        "accent2": "#14b8a6",
        "text": "#0f172a",
        "muted": "#64748b",
    },
    "Forest": {
        "bg1": "#f0fdf4",
        "bg2": "#ecfccb",
        "bg3": "#fef9c3",
        "card": "rgba(255, 255, 255, 0.8)",
        "border": "rgba(34, 197, 94, 0.18)",
        "accent": "#16a34a",
        "accent2": "#ca8a04",
        "text": "#0f172a",
        "muted": "#64748b",
    },
}

theme_name = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=0)
theme = THEMES[theme_name]

st.markdown(
    f"""
    <style>
      :root {{
        --bg-1: {theme['bg1']};
        --bg-2: {theme['bg2']};
        --bg-3: {theme['bg3']};
        --card: {theme['card']};
        --border: {theme['border']};
        --accent: {theme['accent']};
        --accent-2: {theme['accent2']};
        --text: {theme['text']};
        --muted: {theme['muted']};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
      html, body, [class*="css"]  {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
      }
      body {
        background: linear-gradient(135deg, var(--bg-1), var(--bg-2), var(--bg-3));
      }
      .stApp {
        background: transparent;
      }
      [data-testid="stAppViewContainer"] {
        background: transparent;
      }
      .bg-layer {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
      }
      .bg-orb {
        position: absolute;
        width: 360px;
        height: 360px;
        border-radius: 50%;
        filter: blur(30px);
        opacity: 0.45;
        animation: float 10s ease-in-out infinite;
      }
      .orb-1 {
        top: -80px;
        left: -120px;
        background: radial-gradient(circle at 30% 30%, var(--accent), transparent 60%);
      }
      .orb-2 {
        bottom: -120px;
        right: -120px;
        background: radial-gradient(circle at 70% 40%, var(--accent-2), transparent 60%);
        animation-delay: 2s;
      }
      .orb-3 {
        top: 35%;
        right: -120px;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle at 40% 40%, #ffffff, transparent 60%);
        opacity: 0.25;
        animation-delay: 4s;
      }
      @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(18px); }
      }
      .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 22px 28px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(6px);
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
        animation: fadeUp 0.7s ease-out;
      }
      .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 6px;
      }
      .hero-gradient {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }
      .muted {
        color: var(--muted);
        font-size: 0.95rem;
      }
      .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        background: #111827;
        color: #f9fafb;
      }
      .metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
      }
      .label-spam {
        background: #ffedd5;
        color: #9a3412;
        border: 1px solid #fdba74;
      }
      .label-ham {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
      }
      .stButton > button {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.15);
      }
      .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px rgba(15, 23, 42, 0.2);
      }
      .stTextArea textarea, .stTextInput input {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(255, 255, 255, 0.9);
      }
      [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        border: 1px dashed rgba(148, 163, 184, 0.45);
        background: rgba(255, 255, 255, 0.7);
      }
      [data-testid="metric-container"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
      }
      [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85);
        border-right: 1px solid rgba(148, 163, 184, 0.2);
      }
      [data-testid="stTabs"] button[role="tab"] {
        border-radius: 999px;
        padding: 8px 16px;
        margin-right: 6px;
      }
      [data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
      }
      .fade-in {
        animation: fadeUp 0.7s ease-out;
      }
      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='bg-layer'><span class='bg-orb orb-1'></span><span class='bg-orb orb-2'></span><span class='bg-orb orb-3'></span></div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
      <h1><span class='hero-gradient'>Enron Spam Filter</span></h1>
      <p class='muted'>TF-IDF + ML classifier with bilingual support (EN + VI).</p>
      <p class='muted'>Paste text or upload an email to predict spam/ham.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_eval_metrics(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


@st.cache_data
def load_metrics_table(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


@st.cache_data
def count_dataset_files(dataset_dir: Path):
    ham_dir = dataset_dir / "ham"
    spam_dir = dataset_dir / "spam"
    ham_count = sum(1 for p in ham_dir.rglob("*") if p.is_file()) if ham_dir.exists() else 0
    spam_count = sum(1 for p in spam_dir.rglob("*") if p.is_file()) if spam_dir.exists() else 0
    total = ham_count + spam_count
    return {"ham": ham_count, "spam": spam_count, "total": total}


@st.cache_data
def load_split_counts(train_path: Path, test_path: Path):
    if not train_path.exists() or not test_path.exists():
        return None
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_counts = train_df["label"].value_counts().to_dict()
    test_counts = test_df["label"].value_counts().to_dict()
    return {"train": train_counts, "test": test_counts}


def get_score(model, text: str):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([text])[0]
        return float(prob[1])
    if hasattr(model, "decision_function"):
        score = model.decision_function([text])
        return float(score[0])
    return None


def predict_with_threshold(model, text: str, threshold: float | None):
    score = get_score(model, text)
    if score is None or threshold is None:
        label = int(model.predict([text])[0])
        return label, score
    label = 1 if score >= threshold else 0
    return label, score


def explain_linear_model(model, text: str, top_k: int = 12):
    if not hasattr(model, "named_steps"):
        return None
    vec = model.named_steps.get("tfidf")
    clf = model.named_steps.get("clf")
    if vec is None or clf is None or not hasattr(clf, "coef_"):
        return None

    X = vec.transform([text])
    if X.nnz == 0:
        return None

    indices = X.indices
    data = X.data
    coef = clf.coef_[0]
    contrib = data * coef[indices]
    feature_names = vec.get_feature_names_out()

    ranked = sorted(zip(indices, contrib), key=lambda x: x[1], reverse=True)
    top_pos = [(feature_names[i], float(v)) for i, v in ranked[:top_k]]
    top_neg = [(feature_names[i], float(v)) for i, v in ranked[-top_k:]][::-1]
    return top_pos, top_neg


def load_random_sample(label: str):
    folder = DATASET_DIR / label
    files = [p for p in folder.rglob("*") if p.is_file()]
    if not files:
        return None
    pick = random.choice(files)
    raw = pick.read_bytes()
    subject, body = parse_email_bytes(raw)
    return f"{subject}\n{body}".strip()


def get_model_info(model):
    if not hasattr(model, "named_steps"):
        return {"model_type": model.__class__.__name__}
    vec = model.named_steps.get("tfidf")
    clf = model.named_steps.get("clf")
    info = {
        "model_type": model.__class__.__name__,
        "classifier": clf.__class__.__name__ if clf else "unknown",
    }
    if vec is not None:
        info.update(
            {
                "ngram_range": str(vec.ngram_range),
                "min_df": vec.min_df,
                "max_df": vec.max_df,
                "sublinear_tf": vec.sublinear_tf,
            }
        )
    return info


model = load_model(MODEL_PATH)

if model is None:
    st.error("Model not found. Train first: `python -m src.models.train --data-dir data/bilingual`")
    st.stop()

eval_metrics = load_eval_metrics(METRICS_PATH)
metrics_table = load_metrics_table(MODELS_TABLE_PATH)
dataset_counts = count_dataset_files(DATASET_DIR)
split_counts = load_split_counts(TRAIN_CSV_PATH, TEST_CSV_PATH)
model_info = get_model_info(model)

st.sidebar.header("Controls")
top_k = st.sidebar.slider("Top tokens", min_value=5, max_value=30, value=12, step=1)
show_explain = st.sidebar.checkbox("Show token contributions", value=True)

threshold = None
score_label = "Score"
if hasattr(model, "predict_proba"):
    threshold = st.sidebar.slider("Spam threshold", 0.0, 1.0, 0.5, 0.01)
    score_label = "Spam probability"
elif hasattr(model, "decision_function"):
    threshold = st.sidebar.number_input("Decision threshold", value=0.0, step=0.1)
    score_label = "Decision score"

st.sidebar.caption(f"Model: {model_info.get('classifier', 'unknown')}")

tabs = st.tabs(["Predict", "Metrics", "Dataset"])

with tabs[0]:
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader("Input")
        mode = st.radio("Choose input mode", ["Paste text", "Upload email file"], horizontal=True)

        if mode == "Paste text":
            if "text_input" not in st.session_state:
                st.session_state["text_input"] = ""
            st.text_area(
                "Email content",
                key="text_input",
                height=260,
                placeholder="Paste raw email content here...",
            )
            sample_cols = st.columns(2)
            with sample_cols[0]:
                if st.button("Load random spam sample"):
                    sample = load_random_sample("spam")
                    if sample:
                        st.session_state["text_input"] = sample
            with sample_cols[1]:
                if st.button("Load random ham sample"):
                    sample = load_random_sample("ham")
                    if sample:
                        st.session_state["text_input"] = sample
            input_text = st.session_state.get("text_input", "")
        else:
            uploaded = st.file_uploader("Upload a raw email (.txt)", type=["txt", "eml"])
            input_text = ""
            if uploaded is not None:
                subject, body = parse_email_bytes(uploaded.getvalue())
                input_text = f"{subject}\n{body}".strip()
                st.text_area("Parsed preview", value=input_text[:4000], height=200, disabled=True)

        predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Prediction")
        if predict_clicked:
            if not input_text.strip():
                st.warning("Please provide email content first.")
            else:
                label, score = predict_with_threshold(model, input_text, threshold)
                label_name = "spam" if label == 1 else "ham"

                label_class = "label-spam" if label == 1 else "label-ham"
                st.markdown(
                    f"<span class='pill {label_class}'>Prediction: {label_name.upper()}</span>",
                    unsafe_allow_html=True,
                )
                if score is not None:
                    st.markdown(
                        f"<p class='metric'>{score_label}: {score:.4f}</p>", unsafe_allow_html=True
                    )
                    if hasattr(model, "predict_proba"):
                        st.progress(min(max(score, 0.0), 1.0))

                if show_explain:
                    explanation = explain_linear_model(model, input_text, top_k=top_k)
                    if explanation:
                        top_pos, top_neg = explanation
                        st.write("Top contributing tokens")
                        pos_col, neg_col = st.columns(2)
                        with pos_col:
                            st.caption("Spam-leaning")
                            st.table(
                                [{"token": tok, "weight": f"{val:.4f}"} for tok, val in top_pos]
                            )
                        with neg_col:
                            st.caption("Ham-leaning")
                            st.table(
                                [{"token": tok, "weight": f"{val:.4f}"} for tok, val in top_neg]
                            )
                    else:
                        st.info("Explainability is available for linear models only.")
        else:
            st.info("Paste or upload an email, then click Run prediction.")

with tabs[1]:
    st.subheader("Evaluation Metrics")
    if eval_metrics:
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Precision (spam)", f"{eval_metrics['precision_spam']:.4f}")
        mcol2.metric("Recall (spam)", f"{eval_metrics['recall_spam']:.4f}")
        mcol3.metric("F1 (spam)", f"{eval_metrics['f1_spam']:.4f}")
        mcol4.metric("Accuracy", f"{eval_metrics['accuracy']:.4f}")

        st.caption("Confusion matrix (rows=Actual, cols=Predicted)")
        cm = pd.DataFrame(
            [
                [eval_metrics["tn"], eval_metrics["fp"]],
                [eval_metrics["fn"], eval_metrics["tp"]],
            ],
            index=["ham", "spam"],
            columns=["ham", "spam"],
        )
        st.table(cm)
    else:
        st.info("Run evaluation first: `python -m src.models.evaluate --model models/best_model.joblib`")

    st.subheader("Model Comparison")
    if metrics_table is not None:
        st.dataframe(metrics_table, width="stretch")
    else:
        st.info("Run training to generate metrics: `python -m src.models.train --data-dir data/bilingual`")

    st.subheader("Model Details")
    st.json(model_info)

with tabs[2]:
    st.subheader("Dataset Summary")
    if dataset_counts["total"] > 0:
        dcol1, dcol2, dcol3 = st.columns(3)
        dcol1.metric("Ham", dataset_counts["ham"])
        dcol2.metric("Spam", dataset_counts["spam"])
        dcol3.metric("Total", dataset_counts["total"])

        df_counts = pd.DataFrame(
            {"label": ["ham", "spam"], "count": [dataset_counts["ham"], dataset_counts["spam"]]}
        )
        st.bar_chart(df_counts, x="label", y="count", width="stretch")
    else:
        st.info("Dataset not found. Make sure data/bilingual/ham and data/bilingual/spam exist.")

    if split_counts:
        st.subheader("Train/Test Split")
        train_spam = split_counts["train"].get(1, 0)
        train_ham = split_counts["train"].get(0, 0)
        test_spam = split_counts["test"].get(1, 0)
        test_ham = split_counts["test"].get(0, 0)
        split_df = pd.DataFrame(
            {
                "split": ["train", "train", "test", "test"],
                "label": ["ham", "spam", "ham", "spam"],
                "count": [train_ham, train_spam, test_ham, test_spam],
            }
        )
        st.bar_chart(split_df, x="split", y="count", color="label", width="stretch")
