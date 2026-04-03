# So Do He Thong

File nay gom 2 so do Mermaid de dua vao bao cao, slide, hoac mo truc tiep trong IDE neu markdown viewer ho tro Mermaid.

## 1. So do tong quan he thong

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#17181c",
    "primaryColor": "#1f232a",
    "primaryTextColor": "#ece7df",
    "primaryBorderColor": "#c4a878",
    "lineColor": "#8a93a3",
    "secondaryColor": "#252830",
    "tertiaryColor": "#2c313a"
  }
}}%%
flowchart LR
    U[Nguoi dung / Hoi dong]
    UI[Streamlit UI<br/>app.py]

    subgraph INPUT[Du lieu dau vao]
        T[Paste text]
        F[Upload .txt / .eml]
        S[Random sample<br/>data/bilingual]
    end

    P[parse_email_bytes<br/>src/utils/email_parse.py]
    M[Model da train<br/>models/*.joblib]
    X[TF-IDF + Classifier<br/>SVM / LR / NB / RF / XGB]
    R[Ket qua du doan<br/>spam / ham]
    E[Token explanation<br/>top spam-leaning / ham-leaning]
    C[So sanh model<br/>bang benchmark + live comparison]
    G[So lieu danh gia<br/>reports/results/*.csv]

    U --> UI
    T --> UI
    F --> P --> UI
    S --> UI
    UI --> M --> X
    X --> R
    X --> E
    G --> UI
    UI --> C
    X --> C
```

## 2. So do pipeline train va evaluate

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#17181c",
    "primaryColor": "#1f232a",
    "primaryTextColor": "#ece7df",
    "primaryBorderColor": "#c4a878",
    "lineColor": "#8a93a3",
    "secondaryColor": "#252830",
    "tertiaryColor": "#2c313a"
  }
}}%%
flowchart TD
    A[data/bilingual<br/>ham + spam]
    B[Doc va lam sach text<br/>src/utils + text_clean.py]
    C[Train / Test Split<br/>data/processed/train.csv<br/>data/processed/test.csv]
    D[TF-IDF Vectorizer<br/>src/features/vectorize.py]

    subgraph TRAIN[Huong huan luyen]
        NB[Naive Bayes]
        SVM[Linear SVM]
        LR[Logistic Regression]
        RF[Random Forest]
        XGB[XGBoost]
    end

    H[Chon model / luu model<br/>models/*.joblib]
    I[Evaluate tren test set<br/>src/models/evaluate.py]
    J[metrics.csv]
    K[eval_metrics.csv]
    L[classification_report.csv]
    N[misclassified_examples.csv]
    O[Streamlit app.py]

    A --> B --> C
    C --> D
    D --> NB
    D --> SVM
    D --> LR
    D --> RF
    D --> XGB
    NB --> H
    SVM --> H
    LR --> H
    RF --> H
    XGB --> H
    H --> I
    C --> I
    I --> J
    I --> K
    I --> L
    I --> N
    H --> O
    J --> O
    K --> O
    L --> O
    N --> O
```

## 3. Cach trinh bay nhanh khi van dap

- So do 1 de giai thich app dang chay nhu the nao khi nguoi dung nhap email.
- So do 2 de giai thich quy trinh huan luyen, danh gia, va sinh ra cac file model/metric.
- Neu can noi ngan gon: `Input -> Parse/Clean -> TF-IDF -> Model -> Prediction -> Evaluation/Visualization`.
