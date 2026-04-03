from __future__ import annotations

from pathlib import Path
import random
import warnings

import joblib
import altair as alt
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from src.models.evaluate import (
    build_classification_report,
    build_error_analysis,
    compute_metrics,
    get_scores as get_batch_scores,
)
from src.utils.email_parse import parse_email_bytes


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


MODEL_LABELS = {
    "majority": "Majority Baseline",
    "keyword": "Keyword Baseline",
    "nb": "Naive Bayes",
    "svm": "Linear SVM",
    "rf": "Random Forest",
    "lr": "Logistic Regression",
    "xgb": "XGBoost",
}

DECISION_MARGIN = 0.12

TRAINED_MODEL_PATHS = {
    "svm": Path("models") / "svm_best.joblib",
    "lr": Path("models") / "lr_best.joblib",
    "nb": Path("models") / "nb_best.joblib",
    "rf": Path("models") / "rf_best.joblib",
    "xgb": Path("models") / "xgb_best.joblib",
}
DEFAULT_PRIMARY_MODEL = "svm"
DATASET_DIR = Path("data") / "bilingual"
METRICS_PATH = Path("reports") / "results" / "eval_metrics.csv"
MODELS_TABLE_PATH = Path("reports") / "results" / "metrics.csv"
CLASS_REPORT_PATH = Path("reports") / "results" / "classification_report.csv"
ERROR_ANALYSIS_PATH = Path("reports") / "results" / "misclassified_examples.csv"
TRAIN_CSV_PATH = Path("data") / "processed" / "train.csv"
TEST_CSV_PATH = Path("data") / "processed" / "test.csv"


st.set_page_config(
    page_title="Enron Spam Filter Demo",
    page_icon="Email",
    layout="wide",
)

theme = {
    "bg1": "#17181c",
    "bg2": "#1d1f24",
    "bg3": "#252830",
    "card": "rgba(24, 27, 33, 0.84)",
    "surface": "rgba(31, 35, 42, 0.96)",
    "surface_soft": "rgba(38, 42, 50, 0.82)",
    "border": "rgba(196, 168, 120, 0.16)",
    "border_strong": "rgba(196, 168, 120, 0.30)",
    "accent": "#c4a878",
    "accent2": "#8a93a3",
    "text": "#ece7df",
    "muted": "#a79f96",
    "grid": "rgba(255, 248, 236, 0.025)",
    "grid_strong": "rgba(255, 248, 236, 0.05)",
    "halo": "rgba(196, 168, 120, 0.10)",
    "halo2": "rgba(138, 147, 163, 0.08)",
    "glow": "rgba(255, 244, 224, 0.035)",
    "shadow": "rgba(6, 7, 10, 0.58)",
    "tag": "rgba(20, 22, 27, 0.92)",
    "chip": "rgba(196, 168, 120, 0.10)",
    "input": "rgba(22, 25, 31, 0.92)",
    "input_soft": "rgba(29, 33, 39, 0.78)",
    "sidebar": "rgba(18, 20, 25, 0.96)",
    "contrast": "#f8f3eb",
    "danger_bg": "rgba(110, 40, 48, 0.22)",
    "danger_text": "#e3b0b7",
    "danger_border": "rgba(190, 96, 114, 0.34)",
    "success_bg": "rgba(48, 78, 63, 0.22)",
    "success_text": "#b9d9c2",
    "success_border": "rgba(113, 165, 134, 0.32)",
}

st.markdown(
    f"""
    <style>
      :root {{
        --bg-1: {theme['bg1']};
        --bg-2: {theme['bg2']};
        --bg-3: {theme['bg3']};
        --card: {theme['card']};
        --surface: {theme['surface']};
        --surface-soft: {theme['surface_soft']};
        --border: {theme['border']};
        --border-strong: {theme['border_strong']};
        --accent: {theme['accent']};
        --accent-2: {theme['accent2']};
        --text: {theme['text']};
        --muted: {theme['muted']};
        --grid: {theme['grid']};
        --grid-strong: {theme['grid_strong']};
        --halo: {theme['halo']};
        --halo-2: {theme['halo2']};
        --glow: {theme['glow']};
        --shadow: {theme['shadow']};
        --tag: {theme['tag']};
        --chip: {theme['chip']};
        --input: {theme['input']};
        --input-soft: {theme['input_soft']};
        --sidebar: {theme['sidebar']};
        --contrast: {theme['contrast']};
        --danger-bg: {theme['danger_bg']};
        --danger-text: {theme['danger_text']};
        --danger-border: {theme['danger_border']};
        --success-bg: {theme['success_bg']};
        --success-text: {theme['success_text']};
        --success-border: {theme['success_border']};
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
        color-scheme: dark;
      }
      body {
        background: linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
      }
      .stApp {
        background:
          radial-gradient(circle at top right, rgba(196, 168, 120, 0.08), transparent 28%),
          radial-gradient(circle at bottom left, rgba(138, 147, 163, 0.07), transparent 24%),
          linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
      }
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at top right, rgba(196, 168, 120, 0.08), transparent 28%),
          radial-gradient(circle at bottom left, rgba(138, 147, 163, 0.07), transparent 24%),
          linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 48%, var(--bg-3) 100%);
      }
      [data-testid="stHeader"] {
        background: rgba(18, 20, 25, 0.72);
        backdrop-filter: blur(12px);
      }
      [data-testid="stToolbar"] {
        right: 1rem;
      }
      [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1440px;
        padding-top: 2.2rem;
        padding-bottom: 4.5rem;
        padding-left: 2.8rem;
        padding-right: 2.8rem;
      }
      .bg-layer {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
      }
      .bg-grid {
        position: absolute;
        inset: 0;
        background-image:
          linear-gradient(var(--grid-strong) 1px, transparent 1px),
          linear-gradient(90deg, var(--grid) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: radial-gradient(circle at center, black 30%, transparent 78%);
        opacity: 0.20;
        animation: drift 24s linear infinite;
      }
      .bg-halo {
        position: absolute;
        top: -8%;
        left: 18%;
        width: 42vw;
        height: 42vw;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, var(--halo), transparent 65%);
        filter: blur(52px);
        opacity: 0.18;
        animation: sway 18s ease-in-out infinite;
      }
      .bg-beam {
        position: absolute;
        left: -12%;
        top: 42%;
        width: 72vw;
        height: 16vw;
        border-radius: 999px;
        background: linear-gradient(90deg, transparent, var(--glow), transparent);
        filter: blur(30px);
        opacity: 0.10;
        transform: rotate(-12deg);
        animation: beamMove 22s ease-in-out infinite;
      }
      .bg-cloud {
        position: absolute;
        right: -8%;
        top: 18%;
        width: 32vw;
        height: 24vw;
        border-radius: 50%;
        background: radial-gradient(circle at 50% 50%, var(--halo-2), transparent 68%);
        filter: blur(42px);
        opacity: 0.12;
        animation: sway 24s ease-in-out infinite reverse;
      }
      .bg-ribbon {
        position: absolute;
        right: 12%;
        bottom: 8%;
        width: 34vw;
        height: 34vw;
        border-radius: 50%;
        background: conic-gradient(from 160deg, transparent, var(--glow), transparent 58%);
        filter: blur(32px);
        opacity: 0.08;
        animation: spinSlow 30s linear infinite;
      }
      .bg-ring {
        position: absolute;
        border-radius: 50%;
        border: 1px solid var(--grid-strong);
        opacity: 0.16;
        animation: breathe 12s ease-in-out infinite;
      }
      .ring-1 {
        width: 520px;
        height: 520px;
        right: -120px;
        top: 12%;
      }
      .ring-2 {
        width: 360px;
        height: 360px;
        left: -120px;
        bottom: 4%;
        animation-delay: 2s;
      }
      .spark {
        position: absolute;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--halo-2);
        box-shadow: 0 0 14px var(--halo-2);
        opacity: 0;
        animation: twinkle 7s ease-in-out infinite;
      }
      .spark-1 { top: 16%; left: 16%; animation-delay: 0.6s; }
      .spark-2 { top: 28%; right: 18%; animation-delay: 2.1s; }
      .spark-3 { bottom: 18%; left: 38%; animation-delay: 3.4s; }
      .spark-4 { bottom: 24%; right: 28%; animation-delay: 5.2s; }
      .bg-orb {
        position: absolute;
        width: 360px;
        height: 360px;
        border-radius: 50%;
        filter: blur(30px);
        opacity: 0.18;
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
        background: radial-gradient(circle at 40% 40%, var(--halo), transparent 60%);
        opacity: 0.10;
        animation-delay: 4s;
      }
      .orb-4 {
        top: 18%;
        left: 44%;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle at 50% 50%, var(--halo-2), transparent 62%);
        opacity: 0.08;
        animation-delay: 1s;
      }
      @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(18px); }
      }
      @keyframes drift {
        from { transform: translate3d(0, 0, 0); }
        to { transform: translate3d(24px, 24px, 0); }
      }
      @keyframes sway {
        0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
        33% { transform: translate3d(18px, -14px, 0) scale(1.05); }
        66% { transform: translate3d(-16px, 12px, 0) scale(0.98); }
      }
      @keyframes beamMove {
        0%, 100% { transform: translate3d(0, 0, 0) rotate(-12deg); opacity: 0.18; }
        50% { transform: translate3d(60px, -18px, 0) rotate(-8deg); opacity: 0.42; }
      }
      @keyframes spinSlow {
        from { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.04); }
        to { transform: rotate(360deg) scale(1); }
      }
      @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.18; }
        50% { transform: scale(1.06); opacity: 0.36; }
      }
      @keyframes twinkle {
        0%, 100% { opacity: 0; transform: scale(0.7); }
        20% { opacity: 0; }
        35% { opacity: 1; transform: scale(1.2); }
        52% { opacity: 0.35; transform: scale(0.92); }
        70% { opacity: 0; transform: scale(0.7); }
      }
      @keyframes sheen {
        0% { transform: translateX(-120%) rotate(14deg); opacity: 0; }
        14% { opacity: 0.0; }
        22% { opacity: 0.45; }
        34% { transform: translateX(140%) rotate(14deg); opacity: 0; }
        100% { transform: translateX(140%) rotate(14deg); opacity: 0; }
      }
      @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 18px 38px var(--shadow); }
        50% { box-shadow: 0 24px 48px rgba(2, 6, 23, 0.72); }
      }
      .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 32px 34px;
        box-shadow: 0 26px 70px var(--shadow);
        backdrop-filter: blur(6px);
        margin-bottom: 28px;
        position: relative;
        z-index: 1;
        animation: fadeUp 0.7s ease-out;
        overflow: hidden;
      }
      .hero:before {
        content: "";
        position: absolute;
        inset: auto -80px -120px auto;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, var(--halo), transparent 70%);
        opacity: 0.35;
      }
      .hero:after {
        content: "";
        position: absolute;
        top: -20%;
        bottom: -20%;
        width: 160px;
        background: linear-gradient(90deg, transparent, var(--glow), transparent);
        transform: translateX(-140%) rotate(14deg);
        animation: sheen 9s ease-in-out infinite;
        opacity: 0.55;
      }
      .hero-grid {
        display: grid;
        grid-template-columns: 1.45fr 0.95fr;
        gap: 28px;
        align-items: stretch;
      }
      .hero-copy {
        position: relative;
        z-index: 1;
      }
      .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border-radius: 999px;
        background: var(--surface-soft);
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .hero h1 {
        font-size: 2.6rem;
        line-height: 1.02;
        margin: 16px 0 10px;
      }
      .hero-lead {
        color: var(--muted);
        font-size: 1rem;
        max-width: 40rem;
        margin: 0 0 18px;
      }
      .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .hero-tag {
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        background: var(--tag);
        color: var(--contrast);
        font-size: 0.84rem;
        font-weight: 600;
        box-shadow: 0 8px 18px rgba(8, 10, 14, 0.34);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
      }
      .hero-tag:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 14px 24px rgba(8, 10, 14, 0.42);
      }
      .hero-side {
        position: relative;
        z-index: 1;
        display: grid;
        gap: 18px;
      }
      .signal-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 18px 28px var(--shadow);
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        animation: fadeUp 0.85s ease-out both;
      }
      .signal-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 26px 42px rgba(2, 6, 23, 0.70);
        border-color: var(--border-strong);
      }
      .signal-card:after {
        content: "";
        position: absolute;
        right: -35px;
        top: -35px;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: radial-gradient(circle, var(--glow), transparent 68%);
        opacity: 0.7;
      }
      .signal-title {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
      }
      .signal-value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 8px;
      }
      .signal-copy {
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0;
      }
      .hero h1 {
        margin-bottom: 6px;
      }
      .hero-gradient {
        background: linear-gradient(90deg, var(--accent), #d3c0a0 48%, var(--accent-2));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }
      .overview-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 20px;
        margin: 12px 0 34px;
      }
      .overview-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 20px 20px 18px;
        box-shadow: 0 16px 28px var(--shadow);
        backdrop-filter: blur(8px);
        min-height: 178px;
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        animation: fadeUp 0.9s ease-out both;
      }
      .overview-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 24px 40px rgba(2, 6, 23, 0.70);
        border-color: var(--border-strong);
      }
      .overview-card:before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(140deg, rgba(255,255,255,0.015), transparent 42%, var(--glow));
        pointer-events: none;
      }
      .overview-label {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
      }
      .overview-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.1;
        margin-bottom: 6px;
      }
      .overview-note {
        color: var(--muted);
        font-size: 0.9rem;
      }
      .section-kicker {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
      }
      .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0 0 6px;
      }
      .section-note {
        color: var(--muted);
        font-size: 0.94rem;
        margin-bottom: 14px;
      }
      .soft-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 22px 24px;
        box-shadow: 0 16px 28px var(--shadow);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        animation: fadeUp 0.9s ease-out both;
      }
      .soft-panel:hover {
        transform: translateY(-4px);
        box-shadow: 0 22px 38px rgba(2, 6, 23, 0.70);
      }
      .soft-panel:before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 140px;
        height: 4px;
        background: linear-gradient(90deg, var(--accent), #b69a72 48%, var(--accent-2));
      }
      .result-card {
        background: linear-gradient(135deg, var(--surface), var(--surface-soft));
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 22px 22px 20px;
        box-shadow: 0 20px 36px var(--shadow);
        min-height: 235px;
        position: relative;
        overflow: hidden;
        animation: pulseGlow 7s ease-in-out infinite;
      }
      .result-card:after {
        content: "";
        position: absolute;
        top: -15%;
        bottom: -15%;
        width: 120px;
        background: linear-gradient(90deg, transparent, var(--glow), transparent);
        transform: translateX(-160%) rotate(16deg);
        animation: sheen 8s ease-in-out infinite;
        opacity: 0.28;
      }
      .result-topline {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
      }
      .result-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 16px;
      }
      .result-badge.spam {
        background: var(--danger-bg);
        color: var(--danger-text);
        border: 1px solid var(--danger-border);
      }
      .result-badge.ham {
        background: var(--success-bg);
        color: var(--success-text);
        border: 1px solid var(--success-border);
      }
      .result-headline {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 8px;
      }
      .result-copy {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 16px;
      }
      .result-score {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
        gap: 10px;
      }
      .score-chip {
        padding: 12px 14px;
        border-radius: 18px;
        background: var(--chip);
      }
      .score-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 4px;
      }
      .score-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
      }
      .token-table-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 8px;
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
        background: var(--tag);
        color: var(--contrast);
      }
      .metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
      }
      .label-spam {
        background: var(--danger-bg);
        color: var(--danger-text);
        border: 1px solid var(--danger-border);
      }
      .label-ham {
        background: var(--success-bg);
        color: var(--success-text);
        border: 1px solid var(--success-border);
      }
      .stButton > button {
        background: linear-gradient(90deg, #6f7782, #b1946b);
        color: var(--contrast);
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 10px 20px rgba(8, 10, 14, 0.28);
      }
      .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px rgba(8, 10, 14, 0.36);
      }
      .stTextArea textarea, .stTextInput input {
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--input);
        color: var(--text);
      }
      [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        border: 1px dashed var(--border);
        background: var(--input-soft);
      }
      [data-testid="metric-container"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 12px 24px var(--shadow);
      }
      [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
      }
      [data-testid="stTabs"] {
        margin-top: 10px;
      }
      [data-testid="stTabs"] button[role="tab"] {
        border-radius: 999px;
        padding: 10px 18px;
        margin-right: 10px;
        transition: transform 0.18s ease, background 0.18s ease, color 0.18s ease;
      }
      [data-testid="stTabs"] button[role="tab"]:hover {
        transform: translateY(-2px);
        background: var(--surface-soft);
      }
      [data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(90deg, #5d646e, #9f8764);
        color: var(--contrast);
        box-shadow: 0 14px 26px rgba(15, 23, 42, 0.12);
      }
      [data-testid="stDataFrame"], .stTable {
        background: var(--surface-soft);
        border-radius: 18px;
        border: 1px solid var(--border);
        overflow: hidden;
      }
      [data-testid="metric-container"] *, [data-testid="stDataFrame"] *, .stTable *, [data-testid="stSidebar"] *, [data-baseweb="select"] *, [data-baseweb="radio"] *, [data-baseweb="tag"] *, [data-baseweb="input"] *, [data-baseweb="textarea"] * {
        color: var(--text) !important;
      }
      [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
        background: var(--input) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
      }
      [data-baseweb="tag"] {
        background: var(--chip) !important;
        border: 1px solid var(--border) !important;
      }
      [data-baseweb="radio"] > div {
        background: transparent !important;
      }
      [data-baseweb="radio"] input:checked + div, .st-c5 {
        border-color: var(--accent) !important;
      }
      ::placeholder {
        color: var(--muted);
        opacity: 1;
      }
      .section-gap-lg {
        height: 14px;
      }
      .fade-in {
        animation: fadeUp 0.7s ease-out;
      }
      @media (max-width: 900px) {
        .hero-grid, .overview-grid, .result-score {
          grid-template-columns: 1fr;
        }
        [data-testid="stAppViewContainer"] .main .block-container {
          padding-left: 1.2rem;
          padding-right: 1.2rem;
        }
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
    "<div class='bg-layer'><span class='bg-grid'></span><span class='bg-halo'></span><span class='bg-cloud'></span><span class='bg-beam'></span><span class='bg-ribbon'></span><span class='bg-ring ring-1'></span><span class='bg-ring ring-2'></span><span class='spark spark-1'></span><span class='spark spark-2'></span><span class='spark spark-3'></span><span class='spark spark-4'></span><span class='bg-orb orb-1'></span><span class='bg-orb orb-2'></span><span class='bg-orb orb-3'></span><span class='bg-orb orb-4'></span></div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
      <div class='hero-grid'>
        <div class='hero-copy'>
          <span class='eyebrow'>Information Retrieval Project</span>
          <h1><span class='hero-gradient'>Spam Detection</span><br/>that feels production-ready</h1>
          <p class='hero-lead'>Phân loại email spam/ham bằng TF-IDF kết hợp Machine Learning, hỗ trợ song ngữ Anh - Việt và giải thích quyết định bằng token đóng góp.</p>
          <div class='hero-tags'>
            <span class='hero-tag'>TF-IDF Retrieval View</span>
            <span class='hero-tag'>Linear SVM / NB Benchmarks</span>
            <span class='hero-tag'>Streamlit Demo</span>
          </div>
        </div>
        <div class='hero-side'>
          <div class='signal-card'>
            <div class='signal-title'>What This Demo Shows</div>
            <div class='signal-value'>Parse → Score → Explain</div>
            <p class='signal-copy'>Dán email hoặc tải file để xem dự đoán, điểm số và những token đang kéo mô hình về phía spam hoặc ham.</p>
          </div>
          <div class='signal-card'>
            <div class='signal-title'>Why It Matters</div>
            <div class='signal-value'>Fast, traceable, report-ready</div>
            <p class='signal-copy'>Phù hợp cho demo đồ án vì chạy nhanh, dễ giải thích và bám đúng luồng tiền xử lý, biểu diễn văn bản, đánh giá mô hình.</p>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


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


def discover_available_model_paths(metrics_table: pd.DataFrame | None):
    preferred_order = list(TRAINED_MODEL_PATHS.keys())
    if metrics_table is not None and not metrics_table.empty and "model" in metrics_table.columns:
        ordered_from_metrics = [
            key for key in metrics_table["model"].tolist() if key in TRAINED_MODEL_PATHS
        ]
        preferred_order = list(dict.fromkeys(ordered_from_metrics + preferred_order))

    available = {}
    for model_key in preferred_order:
        model_path = TRAINED_MODEL_PATHS[model_key]
        if model_path.exists():
            available[model_key] = model_path
    return available


def get_default_compare_models(available_model_keys: list[str], primary_model_key: str):
    preferred = [primary_model_key, "lr", "nb", "rf"]
    compare = []
    for model_key in preferred:
        if model_key in available_model_keys and model_key not in compare:
            compare.append(model_key)
    if len(compare) < min(4, len(available_model_keys)):
        for model_key in available_model_keys:
            if model_key not in compare:
                compare.append(model_key)
    return compare


@st.cache_data
def load_classification_report(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


@st.cache_data
def load_error_analysis(path: Path):
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


@st.cache_data
def evaluate_saved_model(model_key: str, model_path_str: str, test_path_str: str):
    model_path = Path(model_path_str)
    test_path = Path(test_path_str)
    if not model_path.exists() or not test_path.exists():
        return None

    model = load_model(model_path)
    if model is None:
        return None

    df = pd.read_csv(test_path)
    if df.empty or "text" not in df.columns or "label" not in df.columns:
        return None

    X = df["text"].astype(str)
    y_true = df["label"].astype(int)
    y_pred = model.predict(X)
    y_score = get_batch_scores(model, X)

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["model"] = model_key
    metrics = {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }

    return {
        "metrics": metrics,
        "class_report": build_classification_report(y_true, y_pred),
        "errors": build_error_analysis(df, y_true, y_pred, y_score),
    }


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


def set_text_input_sample(label: str):
    sample = load_random_sample(label)
    if sample:
        st.session_state["text_input"] = sample


def get_model_info(model):
    if not hasattr(model, "named_steps"):
        return {"model_type": model.__class__.__name__}
    vec = model.named_steps.get("tfidf")
    clf = model.named_steps.get("clf")
    classifier_name = clf.__class__.__name__ if clf else "unknown"
    classifier_map = {
        "LinearSVC": ("Linear SVM (LinearSVC)", "svm"),
        "MultinomialNB": ("Naive Bayes (MultinomialNB)", "nb"),
        "LogisticRegression": ("Logistic Regression", "lr"),
        "RandomForestClassifier": ("Random Forest", "rf"),
        "XGBClassifier": ("XGBoost", "xgb"),
    }
    classifier_display, model_key = classifier_map.get(classifier_name, (classifier_name, None))
    info = {
        "model_type": model.__class__.__name__,
        "classifier": classifier_display,
        "model_key": model_key,
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


def get_best_metrics_row(metrics_table: pd.DataFrame | None):
    if metrics_table is None or metrics_table.empty:
        return None
    row = metrics_table.sort_values("f1_spam", ascending=False).iloc[0]
    return row


def get_active_metrics_row(metrics_table: pd.DataFrame | None, model_info: dict):
    if metrics_table is None or metrics_table.empty:
        return None
    model_key = model_info.get("model_key")
    if model_key is None:
        return None
    rows = metrics_table.loc[metrics_table["model"] == model_key]
    if rows.empty:
        return None
    return rows.iloc[0]


def get_metrics_row_by_key(metrics_table: pd.DataFrame | None, model_key: str | None):
    if metrics_table is None or metrics_table.empty or model_key is None:
        return None
    rows = metrics_table.loc[metrics_table["model"] == model_key]
    if rows.empty:
        return None
    return rows.iloc[0]


def render_overview_cards(dataset_counts, eval_metrics, model_info, active_row):
    cards = [
        {
            "label": "Active Model",
            "value": model_info.get("classifier", "unknown"),
            "note": "Mô hình đang được nạp cho demo hiện tại.",
        },
        {
            "label": "Dataset Size",
            "value": f"{dataset_counts['total']:,}",
            "note": f"Ham {dataset_counts['ham']:,} • Spam {dataset_counts['spam']:,}",
        },
        {
            "label": "Active F1 Spam",
            "value": f"{active_row['f1_spam'] * 100:.2f}%" if active_row is not None else "N/A",
            "note": "Điểm F1 của chính mô hình đang dùng để demo.",
        },
        {
            "label": "Accuracy",
            "value": f"{eval_metrics['accuracy'] * 100:.2f}%" if eval_metrics else "N/A",
            "note": "Kết quả trên tập test lưu trong reports/results.",
        },
    ]
    cols = st.columns(len(cards), gap="large")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class='overview-card'>
                  <div class='overview-label'>{card['label']}</div>
                  <div class='overview-value'>{card['value']}</div>
                  <div class='overview-note'>{card['note']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_section_intro(kicker: str, title: str, note: str):
    st.markdown(
        f"""
        <div class='soft-panel'>
          <div class='section-kicker'>{kicker}</div>
          <div class='section-title'>{title}</div>
          <div class='section-note'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(
    label_name: str | None,
    score: float | None,
    score_label: str,
    threshold: float | None,
):
    if label_name is None:
        st.markdown(
            """
            <div class='result-card'>
              <div class='result-topline'>Prediction Console</div>
              <div class='result-headline'>Ready to classify</div>
              <div class='result-copy'>Chọn một email mẫu, dán nội dung hoặc tải file lên rồi nhấn <b>Run prediction</b>.</div>
              <div class='result-score'>
                <div class='score-chip'>
                  <div class='score-label'>Current mode</div>
                  <div class='score-value'>Awaiting input</div>
                </div>
                <div class='score-chip'>
                  <div class='score-label'>Explainability</div>
                  <div class='score-value'>Token-level</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    low_confidence = (
        score is not None
        and threshold is not None
        and abs(score - threshold) < DECISION_MARGIN
    )
    badge_class = "spam" if label_name == "spam" else "ham"
    headline = "Potential Spam" if label_name == "spam" else "Legitimate Email"
    note = (
        "Mẫu này nghiêng mạnh về nhóm thư rác. Kiểm tra thêm liên kết, lời mời gấp, nội dung trúng thưởng hoặc yêu cầu thao tác bất thường."
        if label_name == "spam"
        else "Mẫu này đang có tín hiệu gần với email hợp lệ hơn. Vẫn nên xem thêm nội dung và nguồn gửi nếu cần."
    )
    if low_confidence:
        note += " Kết quả này đang khá sát ngưỡng quyết định nên độ chắc chắn không cao."
    score_text = f"{score:.4f}" if score is not None else "N/A"
    threshold_text = "default" if threshold is None else f"{threshold:.2f}"
    badge_text = label_name.upper()
    st.markdown(
        f"""
        <div class='result-card'>
          <div class='result-topline'>Prediction Console</div>
          <div class='result-badge {badge_class}'>{badge_text}</div>
          <div class='result-headline'>{headline}</div>
          <div class='result-copy'>{note}</div>
          <div class='result-score'>
            <div class='score-chip'>
              <div class='score-label'>{score_label}</div>
              <div class='score-value'>{score_text}</div>
            </div>
            <div class='score-chip'>
              <div class='score-label'>Threshold</div>
              <div class='score-value'>{threshold_text}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_details_cards(model_info: dict):
    cards = [
        {
            "label": "Pipeline",
            "value": model_info.get("model_type", "unknown"),
            "note": "Luồng xử lý hoàn chỉnh từ vector hóa đến phân loại.",
        },
        {
            "label": "Classifier",
            "value": model_info.get("classifier", "unknown"),
            "note": "Thuật toán hiện đang dùng để demo trên app.",
        },
        {
            "label": "N-gram",
            "value": model_info.get("ngram_range", "N/A"),
            "note": "Khai thác từ đơn và cặp từ để giữ thêm ngữ cảnh.",
        },
        {
            "label": "Feature Filter",
            "value": f"min_df={model_info.get('min_df', 'N/A')} • max_df={model_info.get('max_df', 'N/A')}",
            "note": "Loại bớt token quá hiếm hoặc quá phổ biến.",
        },
        {
            "label": "TF Scaling",
            "value": "Sublinear TF" if model_info.get("sublinear_tf") else "Raw TF",
            "note": "Giảm ảnh hưởng của từ lặp lại quá nhiều trong cùng một email.",
        },
    ]

    st.markdown(
        """
        <div class='soft-panel'>
          <div class='section-kicker'>Model Setup</div>
          <div class='section-title'>Cấu hình mô hình đang được dùng để demo</div>
          <div class='section-note'>Thay vì xem JSON kỹ thuật, phần này tóm tắt những tham số quan trọng nhất của pipeline theo cách dễ trình bày hơn.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = [cards[:3], cards[3:]]
    for row in rows:
        cols = st.columns(len(row), gap="large")
        for col, card in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class='overview-card'>
                      <div class='overview-label'>{card['label']}</div>
                      <div class='overview-value'>{card['value']}</div>
                      <div class='overview-note'>{card['note']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def prepare_metrics_display(metrics_table: pd.DataFrame | None):
    if metrics_table is None or metrics_table.empty:
        return None
    display = metrics_table.copy()
    display["model"] = display["model"].replace(MODEL_LABELS)
    display = display.rename(
        columns={
            "model": "Model",
            "precision_spam": "Precision Spam",
            "recall_spam": "Recall Spam",
            "f1_spam": "F1 Spam",
            "accuracy": "Accuracy",
            "best_params": "Best Params",
            "cv_best_f1": "CV Best F1",
        }
    )
    return display.sort_values("F1 Spam", ascending=False)


def prepare_comparison_chart_data(metrics_table: pd.DataFrame | None):
    if metrics_table is None or metrics_table.empty:
        return None
    df = metrics_table.copy()
    df["model_label"] = df["model"].replace(MODEL_LABELS)
    df["precision_pct"] = df["precision_spam"] * 100
    df["recall_pct"] = df["recall_spam"] * 100
    df["f1_pct"] = df["f1_spam"] * 100
    df["accuracy_pct"] = df["accuracy"] * 100
    return df.sort_values("f1_spam", ascending=False).reset_index(drop=True)


def render_model_benchmark_section(metrics_table: pd.DataFrame | None, primary_model_key: str | None):
    chart_df = prepare_comparison_chart_data(metrics_table)
    if chart_df is None or chart_df.empty:
        st.info("Run training to generate metrics: `python -m src.models.train --data-dir data/bilingual`")
        return

    best = chart_df.iloc[0]
    runner_up = chart_df.iloc[1] if len(chart_df) > 1 else None
    margin = (best["f1_pct"] - runner_up["f1_pct"]) if runner_up is not None else 0.0
    primary_label = MODEL_LABELS.get(primary_model_key, primary_model_key or "N/A")
    primary_row = get_metrics_row_by_key(metrics_table, primary_model_key)

    summary_cols = st.columns(4, gap="large")
    summary_cols[0].metric("Benchmark Leader", best["model_label"], f"F1 spam {best['f1_pct']:.2f}%")
    if runner_up is not None:
        summary_cols[1].metric("Runner-up", runner_up["model_label"], f"F1 spam {runner_up['f1_pct']:.2f}%")
        summary_cols[2].metric("Winning Margin", f"{margin:.2f} điểm", "so với mô hình xếp thứ 2")
    else:
        summary_cols[1].metric("Runner-up", "N/A")
        summary_cols[2].metric("Winning Margin", "N/A")
    if primary_row is not None:
        summary_cols[3].metric(
            "Primary Demo Model",
            primary_label,
            f"F1 spam {float(primary_row['f1_spam']) * 100:.2f}%",
        )
    else:
        summary_cols[3].metric("Primary Demo Model", primary_label)

    nb_row = chart_df[chart_df["model"] == "nb"]
    lr_row = chart_df[chart_df["model"] == "lr"]
    compare_bits = []
    if not nb_row.empty:
        compare_bits.append(f"cao hơn Naive Bayes {best['f1_pct'] - float(nb_row.iloc[0]['f1_pct']):.2f} điểm F1")
    if not lr_row.empty:
        compare_bits.append(f"cao hơn Logistic Regression {best['f1_pct'] - float(lr_row.iloc[0]['f1_pct']):.2f} điểm F1")
    compare_sentence = " và ".join(compare_bits) if compare_bits else "đang đứng đầu bảng xếp hạng"
    st.markdown(
        f"""
        <div class='soft-panel'>
          <div class='section-kicker'>Benchmark View</div>
          <div class='section-title'>{best['model_label']} đang dẫn đầu benchmark trên tập test</div>
          <div class='section-note'>Trên cùng dữ liệu huấn luyện và tập test, mô hình này đạt F1 spam {best['f1_pct']:.2f}% và Accuracy {best['accuracy_pct']:.2f}%, {compare_sentence}. App vẫn có thể dùng {primary_label} làm mô hình demo chính để bạn chạy ví dụ trực tiếp và so sánh hành vi từng mô hình trên cùng một câu.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_options = {
        "F1 spam": "f1_pct",
        "Accuracy": "accuracy_pct",
        "Precision spam": "precision_pct",
        "Recall spam": "recall_pct",
    }
    selected_metric = st.selectbox(
        "Comparison metric",
        list(metric_options.keys()),
        index=0,
        key="comparison_metric",
    )
    metric_col = metric_options[selected_metric]

    bar_chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X(f"{metric_col}:Q", title=f"{selected_metric} (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("model_label:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.model_label == best["model_label"],
                alt.value(theme["accent"]),
                alt.value("#7f7a72"),
            ),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("precision_pct:Q", title="Precision", format=".2f"),
                alt.Tooltip("recall_pct:Q", title="Recall", format=".2f"),
                alt.Tooltip("f1_pct:Q", title="F1", format=".2f"),
                alt.Tooltip("accuracy_pct:Q", title="Accuracy", format=".2f"),
            ],
        )
        .properties(height=320)
    )

    scatter_chart = (
        alt.Chart(chart_df)
        .mark_circle(opacity=0.92, stroke="#d9d1c7", strokeWidth=1.0)
        .encode(
            x=alt.X("precision_pct:Q", title="Precision spam (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("recall_pct:Q", title="Recall spam (%)", scale=alt.Scale(domain=[0, 100])),
            size=alt.Size("accuracy_pct:Q", title="Accuracy (%)", scale=alt.Scale(range=[250, 1200])),
            color=alt.condition(
                alt.datum.model_label == best["model_label"],
                alt.value(theme["accent2"]),
                alt.value("#7f7a72"),
            ),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("precision_pct:Q", title="Precision", format=".2f"),
                alt.Tooltip("recall_pct:Q", title="Recall", format=".2f"),
                alt.Tooltip("f1_pct:Q", title="F1", format=".2f"),
                alt.Tooltip("accuracy_pct:Q", title="Accuracy", format=".2f"),
            ],
        )
        .properties(height=320)
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.caption("Leaderboard theo metric bạn chọn")
        st.altair_chart(bar_chart, use_container_width=True)
    with chart_col2:
        st.caption("Precision vs Recall, kích thước điểm theo Accuracy")
        st.altair_chart(scatter_chart, use_container_width=True)


def build_live_comparison_table(
    input_text: str,
    compare_model_keys: list[str],
    available_model_paths: dict[str, Path],
    metrics_table: pd.DataFrame | None,
    primary_model_key: str,
    primary_threshold: float | None,
    expected_label: str | None,
):
    if not input_text.strip():
        return None

    benchmark_lookup = {}
    if metrics_table is not None and not metrics_table.empty and "model" in metrics_table.columns:
        benchmark_lookup = metrics_table.set_index("model").to_dict(orient="index")

    expected_value = {"ham": 0, "spam": 1}.get(expected_label) if expected_label else None
    rows = []
    for order, model_key in enumerate(compare_model_keys):
        model_path = available_model_paths.get(model_key)
        if model_path is None:
            continue
        model = load_model(model_path)
        if model is None:
            continue

        if model_key == primary_model_key:
            label, score = predict_with_threshold(model, input_text, primary_threshold)
            threshold_note = "custom threshold" if primary_threshold is not None else "model default"
        else:
            label = int(model.predict([input_text])[0])
            score = get_score(model, input_text)
            threshold_note = "model default"

        info = get_model_info(model)
        score_type = "Spam probability" if hasattr(model, "predict_proba") else "Decision score"
        benchmark = benchmark_lookup.get(model_key, {})
        row = {
            "order": order,
            "role": "Primary demo" if model_key == primary_model_key else "Compare",
            "model": MODEL_LABELS.get(model_key, info.get("classifier", model_key)),
            "prediction": "spam" if label == 1 else "ham",
            "score": score,
            "score_type": score_type,
            "threshold_mode": threshold_note,
            "test_accuracy": benchmark.get("accuracy"),
            "test_f1_spam": benchmark.get("f1_spam"),
        }
        if expected_value is not None:
            row["match_expected"] = label == expected_value
        rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values(["order", "test_f1_spam"], ascending=[True, False]).reset_index(drop=True)


def render_live_model_comparison(comparison_df: pd.DataFrame | None, expected_label: str | None):
    st.subheader("Compare Models On This Input")
    if comparison_df is None or comparison_df.empty:
        st.info("Nhập một email hoặc tải file rồi nhấn `Run prediction` để so sánh nhiều mô hình.")
        return

    spam_votes = int((comparison_df["prediction"] == "spam").sum())
    ham_votes = int((comparison_df["prediction"] == "ham").sum())
    summary_cols = st.columns(4, gap="large")
    summary_cols[0].metric("Models compared", len(comparison_df))
    summary_cols[1].metric("Spam votes", spam_votes)
    summary_cols[2].metric("Ham votes", ham_votes)
    if expected_label and "match_expected" in comparison_df.columns:
        correct_count = int(comparison_df["match_expected"].sum())
        summary_cols[3].metric(
            "Matches expected",
            f"{correct_count}/{len(comparison_df)}",
            f"expected = {expected_label}",
        )
    else:
        summary_cols[3].metric("Expected label", "Not set")

    display_df = comparison_df.copy()
    display_df["prediction"] = display_df["prediction"].str.upper()
    display_df["score"] = display_df["score"].apply(
        lambda x: f"{float(x):.4f}" if pd.notna(x) else "-"
    )
    display_df["test_accuracy"] = display_df["test_accuracy"].apply(
        lambda x: f"{float(x) * 100:.2f}%" if pd.notna(x) else "-"
    )
    display_df["test_f1_spam"] = display_df["test_f1_spam"].apply(
        lambda x: f"{float(x) * 100:.2f}%" if pd.notna(x) else "-"
    )
    if "match_expected" in display_df.columns:
        display_df["match_expected"] = display_df["match_expected"].map(
            {True: "Dung", False: "Sai"}
        )

    display_df = display_df.rename(
        columns={
            "role": "Role",
            "model": "Model",
            "prediction": "Prediction",
            "score": "Score",
            "score_type": "Score Type",
            "threshold_mode": "Threshold Mode",
            "test_accuracy": "Benchmark Accuracy",
            "test_f1_spam": "Benchmark F1 Spam",
            "match_expected": "Matches Expected",
        }
    )
    display_df = display_df.drop(columns=["order"], errors="ignore")

    st.caption(
        "Cột benchmark là kết quả thật trên tập test. Nếu bạn đặt `expected label`, app sẽ đánh dấu mô hình nào đoán đúng ngay trên ví dụ đang nhập."
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


metrics_table = load_metrics_table(MODELS_TABLE_PATH)
available_model_paths = discover_available_model_paths(metrics_table)
available_model_keys = list(available_model_paths.keys())

if not available_model_keys:
    st.error("No trained models found in models/. Train first: `python -m src.models.train --data-dir data/bilingual`")
    st.stop()

default_primary_index = (
    available_model_keys.index(DEFAULT_PRIMARY_MODEL)
    if DEFAULT_PRIMARY_MODEL in available_model_keys
    else 0
)

st.sidebar.header("Controls")
primary_model_key = st.sidebar.selectbox(
    "Primary demo model",
    available_model_keys,
    index=default_primary_index,
    format_func=lambda key: MODEL_LABELS.get(key, key),
)
default_compare_models = get_default_compare_models(available_model_keys, primary_model_key)
compare_model_keys = st.sidebar.multiselect(
    "Compare models on your input",
    available_model_keys,
    default=default_compare_models,
    format_func=lambda key: MODEL_LABELS.get(key, key),
)
compare_model_keys = [primary_model_key] + [key for key in compare_model_keys if key != primary_model_key]

model = load_model(available_model_paths[primary_model_key])

if model is None:
    st.error("Model not found. Train first: `python -m src.models.train --data-dir data/bilingual`")
    st.stop()

eval_bundle = evaluate_saved_model(
    primary_model_key,
    str(available_model_paths[primary_model_key]),
    str(TEST_CSV_PATH),
)
eval_metrics = eval_bundle["metrics"] if eval_bundle else load_eval_metrics(METRICS_PATH)
classification_report_df = (
    eval_bundle["class_report"] if eval_bundle else load_classification_report(CLASS_REPORT_PATH)
)
error_analysis_df = eval_bundle["errors"] if eval_bundle else load_error_analysis(ERROR_ANALYSIS_PATH)
dataset_counts = count_dataset_files(DATASET_DIR)
split_counts = load_split_counts(TRAIN_CSV_PATH, TEST_CSV_PATH)
model_info = get_model_info(model)
best_metrics_row = get_best_metrics_row(metrics_table)
active_metrics_row = get_active_metrics_row(metrics_table, model_info)
metrics_display = prepare_metrics_display(metrics_table)
top_k = st.sidebar.slider("Top tokens", min_value=5, max_value=30, value=12, step=1)
show_explain = st.sidebar.checkbox("Show token contributions", value=True)

threshold = None
score_label = "Score"
if hasattr(model, "predict_proba"):
    threshold = st.sidebar.slider("Spam threshold", 0.0, 1.0, 0.5, 0.01)
    score_label = "Spam probability"
elif hasattr(model, "decision_function"):
    threshold = st.sidebar.number_input("Decision threshold", value=0.0, step=0.05)
    score_label = "Decision score"

st.sidebar.caption(f"Primary model: {model_info.get('classifier', 'unknown')}")
st.sidebar.caption(f"Compare set: {len(compare_model_keys)} model(s)")
if best_metrics_row is not None:
    st.sidebar.caption(
        f"Benchmark leader: {MODEL_LABELS.get(best_metrics_row['model'], best_metrics_row['model'])}"
    )

render_overview_cards(dataset_counts, eval_metrics, model_info, active_metrics_row)
st.markdown("<div class='section-gap-lg'></div>", unsafe_allow_html=True)

tabs = st.tabs(["Predict", "Metrics", "Dataset"])

with tabs[0]:
    render_section_intro(
        "Live Inference",
        "Thử email bất kỳ và xem mô hình phản ứng thế nào",
        "Dán nội dung mail, tải file .txt/.eml hoặc dùng mẫu ngẫu nhiên để kiểm tra nhanh. Phần bên phải sẽ hiển thị nhãn dự đoán và điểm số.",
    )
    col_left, col_right = st.columns([2, 1], gap="large")
    predicted_label = None
    predicted_score = None
    comparison_df = None

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
            sample_cols = st.columns(2, gap="large")
            with sample_cols[0]:
                st.button(
                    "Load random spam sample",
                    on_click=set_text_input_sample,
                    args=("spam",),
                )
            with sample_cols[1]:
                st.button(
                    "Load random ham sample",
                    on_click=set_text_input_sample,
                    args=("ham",),
                )
            st.caption("Quick test: dùng hai nút mẫu để demo nhanh một email spam và một email ham có sẵn trong dataset.")
            input_text = st.session_state.get("text_input", "")
        else:
            uploaded = st.file_uploader("Upload a raw email (.txt)", type=["txt", "eml"])
            input_text = ""
            if uploaded is not None:
                subject, body = parse_email_bytes(uploaded.getvalue())
                input_text = f"{subject}\n{body}".strip()
                st.text_area("Parsed preview", value=input_text[:4000], height=200, disabled=True)

        expected_choice = st.selectbox(
            "Expected label for this example",
            ["Unknown / just compare", "ham", "spam"],
            index=0,
            help="Nếu bạn biết câu này nên là ham hay spam, app sẽ đánh dấu mô hình nào đoán đúng ngay trên ví dụ bạn nhập.",
        )
        expected_label = None if expected_choice.startswith("Unknown") else expected_choice
        predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Prediction")
        if predict_clicked:
            if not input_text.strip():
                st.warning("Please provide email content first.")
            else:
                label, score = predict_with_threshold(model, input_text, threshold)
                label_name = "spam" if label == 1 else "ham"
                predicted_label = label_name
                predicted_score = score
                comparison_df = build_live_comparison_table(
                    input_text=input_text,
                    compare_model_keys=compare_model_keys,
                    available_model_paths=available_model_paths,
                    metrics_table=metrics_table,
                    primary_model_key=primary_model_key,
                    primary_threshold=threshold,
                    expected_label=expected_label,
                )

                render_prediction_card(label_name, score, score_label, threshold)
                if score is not None and hasattr(model, "predict_proba"):
                    st.progress(min(max(score, 0.0), 1.0))

                if show_explain:
                    explanation = explain_linear_model(model, input_text, top_k=top_k)
                    if explanation:
                        top_pos, top_neg = explanation
                        st.markdown(
                            "<div class='token-table-note'>Top contributing tokens đang kéo quyết định của mô hình về phía spam hoặc ham.</div>",
                            unsafe_allow_html=True,
                        )
                        pos_col, neg_col = st.columns(2, gap="large")
                        with pos_col:
                            st.caption("Spam-leaning")
                            st.dataframe(
                                [{"token": tok, "weight": f"{val:.4f}"} for tok, val in top_pos],
                                use_container_width=True,
                                hide_index=True,
                            )
                        with neg_col:
                            st.caption("Ham-leaning")
                            st.dataframe(
                                [{"token": tok, "weight": f"{val:.4f}"} for tok, val in top_neg],
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        st.info("Explainability is available for linear models only.")
        else:
            render_prediction_card(None, None, score_label, threshold)

    st.markdown("<div class='section-gap-lg'></div>", unsafe_allow_html=True)
    render_live_model_comparison(comparison_df, expected_label if predict_clicked else None)

with tabs[1]:
    render_section_intro(
        "Evaluation View",
        "Hiệu năng mô hình trên tập test",
        "Tab này gom toàn bộ số đo quan trọng để bạn trình bày lúc demo: precision, recall, F1, accuracy, confusion matrix, báo cáo theo từng lớp và danh sách các mẫu dự đoán sai.",
    )
    st.subheader("Evaluation Metrics")
    if eval_metrics:
        mcol1, mcol2, mcol3, mcol4 = st.columns(4, gap="large")
        mcol1.metric("Precision (spam)", f"{eval_metrics['precision_spam']:.4f}")
        mcol2.metric("Recall (spam)", f"{eval_metrics['recall_spam']:.4f}")
        mcol3.metric("F1 (spam)", f"{eval_metrics['f1_spam']:.4f}")
        mcol4.metric("Accuracy", f"{eval_metrics['accuracy']:.4f}")

        extra_cols = st.columns(4, gap="large")
        extra_cols[0].metric("F1 (ham)", f"{eval_metrics.get('f1_ham', 0.0):.4f}")
        extra_cols[1].metric("Macro F1", f"{eval_metrics.get('macro_f1', 0.0):.4f}")
        extra_cols[2].metric(
            "Balanced Acc",
            f"{eval_metrics.get('balanced_accuracy', eval_metrics['accuracy']):.4f}",
        )
        extra_cols[3].metric(
            "Specificity",
            f"{eval_metrics.get('specificity', 0.0):.4f}",
        )

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
        st.info(
            f"Run evaluation first: `python -m src.models.evaluate --model {available_model_paths[primary_model_key]}`"
        )

    st.subheader("Class-wise Report")
    if classification_report_df is not None:
        class_display = classification_report_df.rename(
            columns={
                "class": "Class",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1",
                "support": "Support",
            }
        ).copy()
        for col in ("Precision", "Recall", "F1"):
            class_display[col] = class_display[col].apply(
                lambda x: f"{float(x):.4f}" if pd.notna(x) else "-"
            )
        st.dataframe(class_display, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Run evaluation to generate class-wise report: "
            f"`python -m src.models.evaluate --model {available_model_paths[primary_model_key]}`"
        )

    st.subheader("Model Comparison")
    if metrics_display is not None:
        render_model_benchmark_section(metrics_table, primary_model_key)
        st.dataframe(metrics_display, use_container_width=True, hide_index=True)
    else:
        st.info("Run training to generate metrics: `python -m src.models.train --data-dir data/bilingual`")

    st.subheader("Misclassified Samples")
    if error_analysis_df is not None:
        error_display = error_analysis_df.copy()
        rename_map = {
            "path": "Path",
            "true_name": "True",
            "predicted_name": "Predicted",
            "score": "Score",
            "confidence": "Confidence",
            "length_chars": "Chars",
            "length_words": "Words",
            "text_preview": "Text Preview",
        }
        error_display = error_display.rename(columns=rename_map)
        for col in ("Score", "Confidence"):
            if col in error_display.columns:
                error_display[col] = error_display[col].apply(
                    lambda x: f"{float(x):.4f}" if pd.notna(x) else "-"
                )
        st.caption(
            "Các mẫu dưới đây là email mà mô hình dự đoán sai trên tập test. Đây là phần rất hữu ích để thảo luận giới hạn của hệ thống khi vấn đáp."
        )
        st.dataframe(error_display.head(20), use_container_width=True, hide_index=True)
    else:
        st.info(
            "Run evaluation to generate error analysis: "
            f"`python -m src.models.evaluate --model {available_model_paths[primary_model_key]}`"
        )

    st.subheader("Model Details")
    render_model_details_cards(model_info)

with tabs[2]:
    render_section_intro(
        "Dataset View",
        "Nhìn nhanh quy mô và phân bố dữ liệu",
        "Tab này giúp giải thích dataset gồm bao nhiêu ham/spam, cách chia train-test và vì sao mô hình có thể học được tín hiệu phân biệt rõ.",
    )
    st.subheader("Dataset Summary")
    if dataset_counts["total"] > 0:
        dcol1, dcol2, dcol3 = st.columns(3, gap="large")
        dcol1.metric("Ham", dataset_counts["ham"])
        dcol2.metric("Spam", dataset_counts["spam"])
        dcol3.metric("Total", dataset_counts["total"])

        df_counts = pd.DataFrame(
            {"label": ["ham", "spam"], "count": [dataset_counts["ham"], dataset_counts["spam"]]}
        )
        st.bar_chart(df_counts, x="label", y="count", use_container_width=True)
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
        st.bar_chart(split_df, x="split", y="count", color="label", use_container_width=True)
