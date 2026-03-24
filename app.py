from __future__ import annotations

from pathlib import Path
import random
import warnings

import joblib
import altair as alt
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

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


MODEL_PATH = Path("models") / "svm_best.joblib"
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
          linear-gradient(rgba(255,255,255,0.20) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.16) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: radial-gradient(circle at center, black 30%, transparent 78%);
        opacity: 0.48;
        animation: drift 24s linear infinite;
      }
      .bg-halo {
        position: absolute;
        top: -8%;
        left: 18%;
        width: 42vw;
        height: 42vw;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.78), transparent 65%);
        filter: blur(40px);
        opacity: 0.42;
        animation: sway 18s ease-in-out infinite;
      }
      .bg-beam {
        position: absolute;
        left: -12%;
        top: 42%;
        width: 72vw;
        height: 16vw;
        border-radius: 999px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.26), transparent);
        filter: blur(24px);
        opacity: 0.34;
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
        background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.72), transparent 68%);
        filter: blur(36px);
        opacity: 0.24;
        animation: sway 24s ease-in-out infinite reverse;
      }
      .bg-ribbon {
        position: absolute;
        right: 12%;
        bottom: 8%;
        width: 34vw;
        height: 34vw;
        border-radius: 50%;
        background: conic-gradient(from 160deg, rgba(255,255,255,0.0), rgba(255,255,255,0.48), transparent 58%);
        filter: blur(26px);
        opacity: 0.30;
        animation: spinSlow 30s linear infinite;
      }
      .bg-ring {
        position: absolute;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.42);
        opacity: 0.38;
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
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 0 18px rgba(255,255,255,0.95);
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
        opacity: 0.60;
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
      .orb-4 {
        top: 18%;
        left: 44%;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0.85), transparent 62%);
        opacity: 0.22;
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
        0%, 100% { box-shadow: 0 18px 38px rgba(15, 23, 42, 0.08); }
        50% { box-shadow: 0 24px 48px rgba(15, 23, 42, 0.13); }
      }
      .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 32px 34px;
        box-shadow: 0 26px 70px rgba(15, 23, 42, 0.12);
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
        background: radial-gradient(circle, rgba(255,255,255,0.85), transparent 70%);
        opacity: 0.8;
      }
      .hero:after {
        content: "";
        position: absolute;
        top: -20%;
        bottom: -20%;
        width: 180px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.28), transparent);
        transform: translateX(-140%) rotate(14deg);
        animation: sheen 9s ease-in-out infinite;
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
        background: rgba(255,255,255,0.72);
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
        background: rgba(17, 24, 39, 0.88);
        color: #fff;
        font-size: 0.84rem;
        font-weight: 600;
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.14);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
      }
      .hero-tag:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 16px 28px rgba(15, 23, 42, 0.2);
      }
      .hero-side {
        position: relative;
        z-index: 1;
        display: grid;
        gap: 18px;
      }
      .signal-card {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 18px 28px rgba(15, 23, 42, 0.08);
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        animation: fadeUp 0.85s ease-out both;
      }
      .signal-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 26px 42px rgba(15, 23, 42, 0.14);
        border-color: rgba(255,255,255,0.85);
      }
      .signal-card:after {
        content: "";
        position: absolute;
        right: -35px;
        top: -35px;
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.8), transparent 68%);
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
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
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
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 20px 20px 18px;
        box-shadow: 0 16px 28px rgba(15, 23, 42, 0.07);
        backdrop-filter: blur(8px);
        min-height: 178px;
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        animation: fadeUp 0.9s ease-out both;
      }
      .overview-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 24px 40px rgba(15, 23, 42, 0.12);
        border-color: rgba(249, 115, 22, 0.35);
      }
      .overview-card:before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(140deg, rgba(255,255,255,0.12), transparent 42%, rgba(255,255,255,0.28));
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
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 22px 24px;
        box-shadow: 0 16px 28px rgba(15, 23, 42, 0.07);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        animation: fadeUp 0.9s ease-out both;
      }
      .soft-panel:hover {
        transform: translateY(-4px);
        box-shadow: 0 22px 38px rgba(15, 23, 42, 0.11);
      }
      .soft-panel:before {
        content: "";
        position: absolute;
        inset: 0 auto auto 0;
        width: 140px;
        height: 4px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
      }
      .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.78));
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 22px 22px 20px;
        box-shadow: 0 20px 36px rgba(15, 23, 42, 0.09);
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
        width: 160px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.36), transparent);
        transform: translateX(-160%) rotate(16deg);
        animation: sheen 8s ease-in-out infinite;
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
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #fda4af;
      }
      .result-badge.ham {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #86efac;
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
        background: rgba(17, 24, 39, 0.06);
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
        background: rgba(255,255,255,0.72);
      }
      [data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
        box-shadow: 0 14px 26px rgba(15, 23, 42, 0.12);
      }
      [data-testid="stDataFrame"], .stTable {
        background: rgba(255,255,255,0.74);
        border-radius: 18px;
        border: 1px solid var(--border);
        overflow: hidden;
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


def render_model_benchmark_section(metrics_table: pd.DataFrame | None):
    chart_df = prepare_comparison_chart_data(metrics_table)
    if chart_df is None or chart_df.empty:
        st.info("Run training to generate metrics: `python -m src.models.train --data-dir data/bilingual`")
        return

    best = chart_df.iloc[0]
    runner_up = chart_df.iloc[1] if len(chart_df) > 1 else None
    margin = (best["f1_pct"] - runner_up["f1_pct"]) if runner_up is not None else 0.0

    summary_cols = st.columns(3, gap="large")
    summary_cols[0].metric("Best Model", best["model_label"], f"F1 spam {best['f1_pct']:.2f}%")
    if runner_up is not None:
        summary_cols[1].metric("Runner-up", runner_up["model_label"], f"F1 spam {runner_up['f1_pct']:.2f}%")
        summary_cols[2].metric("Winning Margin", f"{margin:.2f} điểm", "so với mô hình xếp thứ 2")
    else:
        summary_cols[1].metric("Runner-up", "N/A")
        summary_cols[2].metric("Winning Margin", "N/A")

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
          <div class='section-kicker'>Why This Model</div>
          <div class='section-title'>{best['model_label']} đang là mô hình tốt nhất trên chính dataset của nhóm</div>
          <div class='section-note'>Trên cùng dữ liệu huấn luyện và tập test, mô hình này đạt F1 spam {best['f1_pct']:.2f}% và Accuracy {best['accuracy_pct']:.2f}%, {compare_sentence}. Đây là lý do hợp lý để chọn nó làm mô hình demo.</div>
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
                alt.value("#94a3b8"),
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
        .mark_circle(opacity=0.92, stroke="white", strokeWidth=1.2)
        .encode(
            x=alt.X("precision_pct:Q", title="Precision spam (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("recall_pct:Q", title="Recall spam (%)", scale=alt.Scale(domain=[0, 100])),
            size=alt.Size("accuracy_pct:Q", title="Accuracy (%)", scale=alt.Scale(range=[250, 1200])),
            color=alt.condition(
                alt.datum.model_label == best["model_label"],
                alt.value(theme["accent2"]),
                alt.value("#94a3b8"),
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


model = load_model(MODEL_PATH)

if model is None:
    st.error("Model not found. Train first: `python -m src.models.train --data-dir data/bilingual`")
    st.stop()

eval_metrics = load_eval_metrics(METRICS_PATH)
metrics_table = load_metrics_table(MODELS_TABLE_PATH)
dataset_counts = count_dataset_files(DATASET_DIR)
split_counts = load_split_counts(TRAIN_CSV_PATH, TEST_CSV_PATH)
model_info = get_model_info(model)
best_metrics_row = get_best_metrics_row(metrics_table)
active_metrics_row = get_active_metrics_row(metrics_table, model_info)
metrics_display = prepare_metrics_display(metrics_table)

st.sidebar.header("Controls")
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

st.sidebar.caption(f"Model: {model_info.get('classifier', 'unknown')}")

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

with tabs[1]:
    render_section_intro(
        "Evaluation View",
        "Hiệu năng mô hình trên tập test",
        "Tab này gom toàn bộ số đo quan trọng để bạn trình bày lúc demo: precision, recall, F1, accuracy, confusion matrix và bảng benchmark giữa các mô hình.",
    )
    st.subheader("Evaluation Metrics")
    if eval_metrics:
        mcol1, mcol2, mcol3, mcol4 = st.columns(4, gap="large")
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
    if metrics_display is not None:
        render_model_benchmark_section(metrics_table)
        st.dataframe(metrics_display, use_container_width=True, hide_index=True)
    else:
        st.info("Run training to generate metrics: `python -m src.models.train --data-dir data/bilingual`")

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
