
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)


# =========================================================
# تنظیمات صفحه
# =========================================================

st.set_page_config(
    page_title="سامانه هوشمند غربالگری دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

KAGGLE_FILE = DATA_DIR / "diabetes_prediction_dataset.csv"
CDC_FILE = DATA_DIR / "cdc_diabetes_health_indicators.csv"


# =========================================================
# طراحی و رنگ‌بندی
# =========================================================

st.markdown("""
<style>

/* ---------- رنگ‌های اصلی ---------- */
:root {
    --navy: #173B57;
    --navy-dark: #102D43;
    --teal: #079B91;
    --teal-dark: #087F78;
    --page: #F2F6F9;
    --white: #FFFFFF;
    --text: #263746;
    --muted: #64748B;
    --border: #DCE5EC;
    --green: #16865B;
    --red: #C43D4B;
    --orange: #C47B20;
}

/* ---------- پس‌زمینه ---------- */
.stApp {
    background: var(--page) !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--page) !important;
}

[data-testid="stMain"] {
    background: var(--page) !important;
}

.block-container {
    padding-top: 1.7rem !important;
    padding-bottom: 3rem !important;
    max-width: 1450px !important;
}

/* ---------- متن‌های عمومی ---------- */
html, body, [class*="st-"] {
    font-family: Tahoma, Arial, sans-serif !important;
}

[data-testid="stMarkdownContainer"] {
    color: var(--text) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: var(--text) !important;
    line-height: 1.9 !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: var(--navy) !important;
    font-weight: 800 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--navy) !important;
    font-weight: 700 !important;
}

/* ---------- هدر اصلی ---------- */
.hero {
    background: linear-gradient(120deg, #173B57 0%, #205D75 65%, #079B91 100%);
    border-radius: 22px;
    padding: 30px 32px;
    margin-bottom: 25px;
    color: #FFFFFF !important;
    box-shadow: 0 8px 25px rgba(23, 59, 87, 0.13);
    direction: rtl;
    text-align: right;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 27px;
    font-weight: 900;
    margin: 0 0 10px 0;
}

.hero p {
    color: #E6F4F7 !important;
    font-size: 14px;
    margin: 0;
    line-height: 2;
}

.hero-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    border-radius: 17px;
    background: rgba(255,255,255,0.17);
    border: 1px solid rgba(255,255,255,0.25);
    color: #FFFFFF !important;
    font-size: 29px;
    margin-bottom: 14px;
}

/* ---------- کارت‌ها ---------- */
.panel {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 3px 12px rgba(23, 59, 87, 0.045);
    direction: rtl;
    text-align: right;
}

.panel-title {
    color: var(--navy) !important;
    font-size: 17px;
    font-weight: 900;
    margin-bottom: 7px;
}

.panel-description {
    color: var(--muted) !important;
    font-size: 13px;
    line-height: 1.9;
}

/* ---------- کارت‌های آماری ---------- */
.stat-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 17px 18px;
    min-height: 112px;
    box-shadow: 0 3px 12px rgba(23, 59, 87, 0.04);
    direction: rtl;
    text-align: right;
}

.stat-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 37px;
    height: 37px;
    border-radius: 11px;
    background: #E8F5F4;
    color: #087F78 !important;
    font-size: 19px;
    margin-bottom: 9px;
}

.stat-label {
    color: #64748B !important;
    font-size: 12px;
    margin-bottom: 5px;
}

.stat-value {
    color: #173B57 !important;
    font-size: 23px;
    font-weight: 900;
}

/* ---------- برچسب‌ها ---------- */
.section-heading {
    color: #173B57 !important;
    font-size: 19px;
    font-weight: 900;
    margin: 24px 0 13px 0;
    direction: rtl;
    text-align: right;
}

.info-chip {
    display: inline-block;
    background: #E6F4F3;
    color: #087F78 !important;
    border: 1px solid #C8E8E5;
    border-radius: 9px;
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 800;
}

/* ---------- دکمه‌ها ---------- */
.stButton > button,
.stFormSubmitButton > button {
    background: #079B91 !important;
    color: #FFFFFF !important;
    border: 1px solid #079B91 !important;
    border-radius: 11px !important;
    min-height: 43px;
    font-weight: 800 !important;
    transition: 0.2s ease;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #087F78 !important;
    border-color: #087F78 !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 13px rgba(7,155,145,0.2);
}

.stButton > button:disabled {
    background: #A8C8C5 !important;
    color: #FFFFFF !important;
}

/* ---------- فیلدهای ورودی ---------- */
input, textarea {
    color: #263746 !important;
    background: #FFFFFF !important;
    border-color: #C9D6E0 !important;
    border-radius: 9px !important;
}

input::placeholder,
textarea::placeholder {
    color: #8493A1 !important;
    opacity: 1 !important;
}

[data-baseweb="input"] {
    background: #FFFFFF !important;
    border-color: #C9D6E0 !important;
}

[data-baseweb="input"] input {
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
}

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #C9D6E0 !important;
    color: #263746 !important;
}

[data-baseweb="select"] span {
    color: #263746 !important;
}

[data-baseweb="select"] svg {
    fill: #173B57 !important;
    color: #173B57 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background: #FFFFFF !important;
    color: #263746 !important;
}

[role="option"] {
    background: #FFFFFF !important;
    color: #263746 !important;
}

[role="option"]:hover {
    background: #E8F5F4 !important;
    color: #173B57 !important;
}

/* ---------- رادیو و چک‌باکس ---------- */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #263746 !important;
}

[data-testid="stRadio"] input,
[data-testid="stCheckbox"] input {
    accent-color: #079B91 !important;
}

[data-testid="stRadio"] svg,
[data-testid="stCheckbox"] svg {
    color: #079B91 !important;
    fill: #079B91 !important;
}

/* ---------- نوار کناری ---------- */
[data-testid="stSidebar"] {
    background: #173B57 !important;
    border-left: 1px solid #244B65 !important;
}

[data-testid="stSidebar"] > div {
    background: #173B57 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label {
    color: #F4F8FB !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #C5D7E3 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] svg {
    color: #48D5C7 !important;
    fill: #48D5C7 !important;
}

/* ---------- آیکون‌ها ---------- */
svg {
    color: inherit;
}

[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons {
    color: #079B91 !important;
    fill: #079B91 !important;
}

[data-testid="stSidebar"] [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .material-symbols-rounded,
[data-testid="stSidebar"] .material-icons {
    color: #65E0D4 !important;
    fill: #65E0D4 !important;
}

[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] button svg {
    color: #079B91 !important;
    fill: #079B91 !important;
}

/* ---------- تب‌ها ---------- */
button[data-baseweb="tab"] {
    color: #526579 !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #087F78 !important;
    border-bottom-color: #079B91 !important;
}

button[data-baseweb="tab"] svg {
    color: #079B91 !important;
    fill: #079B91 !important;
}

/* ---------- پیام‌ها ---------- */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #263746 !important;
}

[data-testid="stAlert"] svg {
    color: #087F78 !important;
    fill: #087F78 !important;
}

/* ---------- بازشونده ---------- */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE5EC !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #173B57 !important;
    font-weight: 700 !important;
}

/* ---------- متریک‌های Streamlit ---------- */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE5EC !important;
    border-radius: 13px !important;
    padding: 13px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #64748B !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #173B57 !important;
}

[data-testid="stMetricDelta"] {
    color: #087F78 !important;
}

/* ---------- جدول ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE5EC !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ---------- جداکننده ---------- */
hr {
    border-color: #DCE5EC !important;
}

/* ---------- فوتر ---------- */
.footer {
    border-top: 1px solid #DCE5EC;
    margin-top: 35px;
    padding-top: 18px;
    text-align: center;
    color: #718096 !important;
    font-size: 12px;
    direction: rtl;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# توابع عمومی
# =========================================================

def show_header():
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🩺</div>
        <h1>سامانه هوشمند غربالگری و پیش‌بینی دیابت</h1>
        <p>
            تحلیل داده‌های سلامت با استفاده از الگوریتم‌های یادگیری ماشین
            برای ارزیابی ریسک ابتلا به دیابت
        </p>
    </div>
    """, unsafe_allow_html=True)


def section_title(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )
    if description:
        st.caption(description)


def stat_card(icon, label, value):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">{icon}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def make_styler(df):
    return (
        df.style
        .set_properties(**{
            "background-color": "#FFFFFF",
            "color": "#263746",
            "border-color": "#DCE5EC",
            "text-align": "center"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#E8F1F6"),
                    ("color", "#173B57"),
                    ("font-weight", "bold"),
                    ("text-align", "center")
                ]
            }
        ])
    )


# =========================================================
# بارگذاری داده‌ها
# =========================================================

@st.cache_data(show_spinner=False)
def load_kaggle_data():
    if not KAGGLE_FILE.exists():
        raise FileNotFoundError(
            "فایل diabetes_prediction_dataset.csv در پوشه dataset پیدا نشد."
        )

    df = pd.read_csv(KAGGLE_FILE)
    df.columns = df.columns.str.strip()

    if "diabetes" not in df.columns:
        raise ValueError("ستون هدف diabetes در دیتاست پیدا نشد.")

    return df


@st.cache_data(show_spinner=False)
def load_cdc_data():
    if CDC_FILE.exists():
        df = pd.read_csv(CDC_FILE)
        df.columns = df.columns.str.strip()
        return df

    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=891)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()

        df = pd.concat(
            [features.reset_index(drop=True),
             targets.reset_index(drop=True)],
            axis=1
        )

        df.columns = df.columns.str.strip()
        return df

    except Exception as e:
        raise FileNotFoundError(
            "دیتاست CDC در پوشه dataset موجود نیست و دریافت خودکار آن نیز موفق نبود. "
            "فایل cdc_diabetes_health_indicators.csv را در پوشه dataset قرار دهید."
        ) from e


# =========================================================
# آماده‌سازی دیتاست
# =========================================================

KAGGLE_BASIC_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi"
]

KAGGLE_FULL_FEATURES = KAGGLE_BASIC_FEATURES + [
    "HbA1c_level",
    "blood_glucose_level"
]

CDC_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]


def prepare_dataset(dataset_name, mode):
    if dataset_name == "دیتاست Kaggle":
        df = load_kaggle_data()

        if mode == "غربالگری بدون آزمایش":
            features = KAGGLE_BASIC_FEATURES
        else:
            features = KAGGLE_FULL_FEATURES

        missing = [c for c in features + ["diabetes"] if c not in df.columns]
        if missing:
            raise ValueError(f"ستون‌های زیر در دیتاست وجود ندارند: {missing}")

        X = df[features].copy()
        y = pd.to_numeric(df["diabetes"], errors="coerce")

        valid = y.notna()
        X = X.loc[valid].copy()
        y = y.loc[valid].astype(int)

        categorical = [
            c for c in ["gender", "smoking_history"]
            if c in X.columns
        ]
        numeric = [c for c in features if c not in categorical]

        return X, y, categorical, numeric, (
            "برچسب مثبت این دیتاست مطابق ستون diabetes است."
        )

    else:
        df = load_cdc_data()

        if "Diabetes_012" in df.columns:
            target_col = "Diabetes_012"
            raw_y = pd.to_numeric(df[target_col], errors="coerce")
            y = (raw_y > 0).astype(int)
        elif "Diabetes_binary" in df.columns:
            target_col = "Diabetes_binary"
            y = pd.to_numeric(df[target_col], errors="coerce")
        elif "diabetes" in df.columns:
            target_col = "diabetes"
            y = pd.to_numeric(df[target_col], errors="coerce")
        else:
            raise ValueError(
                "ستون هدف دیابت در دیتاست CDC پیدا نشد."
            )

        features = [c for c in CDC_FEATURES if c in df.columns]

        if not features:
            raise ValueError("هیچ‌کدام از ستون‌های موردنیاز CDC پیدا نشدند.")

        X = df[features].copy()

        valid = y.notna()
        X = X.loc[valid].copy()
        y = y.loc[valid].astype(int)

        categorical = [
            c for c in ["Sex", "Age", "Education", "Income"]
            if c in X.columns
        ]
        numeric = [c for c in features if c not in categorical]

        return X, y, categorical, numeric, (
            "در این دیتاست، افراد دارای پیش‌دیابت یا دیابت در گروه مثبت قرار می‌گیرند."
        )


# =========================================================
# ساخت پیش‌پردازش و الگوریتم‌ها
# =========================================================

def create_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def build_preprocessor(categorical, numeric):
    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder())
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def build_models(preprocessor):
    models = {
        "رگرسیون لجستیک": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        ),

        "درخت تصمیم": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42
        ),

        "جنگل تصادفی": RandomForestClassifier(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "درخت‌های بسیار تصادفی": ExtraTreesClassifier(
            n_estimators=180,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "گرادیان بوستینگ": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    except Exception:
        pass

    pipelines = {}

    for name, model in models.items():
        pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

    return pipelines


# =========================================================
# آموزش و ارزیابی
# =========================================================

def train_all_models(X, y, categorical, numeric):
    if y.nunique() < 2:
        raise ValueError("برای آموزش، دیتاست باید حداقل دو کلاس داشته باشد.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(categorical, numeric)
    pipelines = build_models(preprocessor)

    trained = {}
    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(pipelines)

    for index, (name, pipeline) in enumerate(pipelines.items()):
        status.info(f"در حال آموزش الگوریتم: {name}")

        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        predictions = pipeline.predict(X_test)

        try:
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, probabilities)
            pr_auc = average_precision_score(y_test, probabilities)
        except Exception:
            roc = np.nan
            pr_auc = np.nan

        results.append({
            "الگوریتم": name,
            "دقت": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc,
            "PR-AUC": pr_auc,
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "زمان آموزش (ثانیه)": elapsed
        })

        trained[name] = pipeline
        progress.progress((index + 1) / total)

    status.success("آموزش همه الگوریتم‌ها به پایان رسید.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    return trained, results_df, X_test, y_test


# =========================================================
# مقداردهی اولیه حافظه برنامه
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "active_key" not in st.session_state:
    st.session_state.active_key = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =========================================================
# نوار کناری
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 5px 20px 5px;">
        <div style="
            width:70px;height:70px;margin:auto;
            background:#24536E;border:1px solid #4B758D;
            border-radius:20px;display:flex;
            align-items:center;justify-content:center;
            font-size:34px;color:#FFFFFF;">
            🩺
        </div>
        <h3 style="color:#FFFFFF !important;margin-top:13px;">
            پنل تحلیل دیابت
        </h3>
        <p style="color:#C5D7E3 !important;font-size:12px;">
            سامانه پژوهشی یادگیری ماشین
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "بخش موردنظر را انتخاب کنید:",
        [
            "داشبورد اصلی",
            "آموزش و ارزیابی مدل‌ها",
            "پیش‌بینی بیمار",
            "درباره پروژه"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown(
        "<p style='color:#FFFFFF;font-weight:800;'>تنظیمات دیتاست</p>",
        unsafe_allow_html=True
    )

    dataset_name = st.selectbox(
        "منبع داده",
        ["دیتاست Kaggle", "دیتاست CDC"]
    )

    if dataset_name == "دیتاست Kaggle":
        mode = st.selectbox(
            "نوع تحلیل",
            [
                "غربالگری بدون آزمایش",
                "همه ویژگی‌ها"
            ]
        )
    else:
        mode = "همه ویژگی‌ها"
        st.caption("در دیتاست CDC از ویژگی‌های موجود استفاده می‌شود.")

    current_key = f"{dataset_name}_{mode}"

    if st.session_state.active_key != current_key:
        st.session_state.trained_models = None
        st.session_state.results_df = None
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.session_state.last_prediction = None
        st.session_state.active_key = current_key

    st.markdown("---")
    st.markdown(
        "<p style='color:#C5D7E3;font-size:11px;text-align:center;'>"
        "نسخه پژوهشی | Machine Learning"
        "</p>",
        unsafe_allow_html=True
    )


# =========================================================
# هدر
# =========================================================

show_header()


# =========================================================
# داشبورد اصلی
# =========================================================

if page == "داشبورد اصلی":

    section_title(
        "نمای کلی سامانه",
        "در این بخش وضعیت دیتاست و عملکرد کلی سامانه نمایش داده می‌شود."
    )

    try:
        X, y, categorical, numeric, target_note = prepare_dataset(
            dataset_name,
            mode
        )

        total_records = len(X)
        total_features = len(X.columns)
        positive = int((y == 1).sum())
        negative = int((y == 0).sum())

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            stat_card("📊", "تعداد کل رکوردها", f"{total_records:,}")

        with c2:
            stat_card("🧩", "تعداد ویژگی‌ها", str(total_features))

        with c3:
            stat_card("🟢", "کلاس مثبت", f"{positive:,}")

        with c4:
            stat_card("🔵", "کلاس منفی", f"{negative:,}")

        section_title("وضعیت دیتاست")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("""
            <div class="panel">
                <div class="panel-title">📁 اطلاعات منبع داده</div>
                <p class="panel-description">
                    منبع انتخاب‌شده:
                </p>
            """, unsafe_allow_html=True)

            st.markdown(
                f"<div class='info-chip'>{dataset_name}</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<p style='color:#263746;margin-top:15px;'>"
                f"نوع تحلیل: <b>{mode}</b></p>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<p style='color:#64748B;font-size:12px;'>{target_note}</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            distribution = pd.DataFrame({
                "وضعیت": ["بدون دیابت", "گروه مثبت"],
                "تعداد": [negative, positive]
            })

            st.markdown(
                "<div class='panel-title'>📈 توزیع کلاس‌ها</div>",
                unsafe_allow_html=True
            )

            st.bar_chart(
                distribution.set_index("وضعیت"),
                color="#079B91"
            )

        section_title("نمونه‌ای از داده‌ها")
        st.dataframe(
            make_styler(X.head(10)),
            use_container_width=True
        )

    except Exception as e:
        st.error(str(e))
        st.info(
            "ابتدا فایل دیتاست را در پوشه dataset پروژه قرار دهید."
        )


# =========================================================
# آموزش و ارزیابی مدل‌ها
# =========================================================

elif page == "آموزش و ارزیابی مدل‌ها":

    section_title(
        "آموزش الگوریتم‌های یادگیری ماشین",
        "الگوریتم‌ها با داده‌های آموزشی آموزش داده شده و روی داده‌های آزمون ارزیابی می‌شوند."
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-title">⚙️ تنظیمات آموزش</div>
        <p class="panel-description">
            داده‌ها به دو بخش آموزش و آزمون تقسیم می‌شوند.
            ۸۰ درصد داده‌ها برای آموزش و ۲۰ درصد برای آزمون استفاده می‌شود.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        X, y, categorical, numeric, target_note = prepare_dataset(
            dataset_name,
            mode
        )

        st.info(
            f"تعداد رکوردها: {len(X):,} | "
            f"تعداد ویژگی‌ها: {len(X.columns)}"
        )

        st.caption(target_note)

        if st.button(
            "🚀 شروع آموزش و ارزیابی الگوریتم‌ها",
            use_container_width=True
        ):
            try:
                st.session_state.last_prediction = None

                trained, results, X_test, y_test = train_all_models(
                    X,
                    y,
                    categorical,
                    numeric
                )

                st.session_state.trained_models = trained
                st.session_state.results_df = results
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                st.success("مدل‌ها با موفقیت آموزش داده شدند.")

            except Exception as e:
                st.error(f"خطا در فرایند آموزش: {e}")

        if st.session_state.results_df is not None:

            results = st.session_state.results_df

            section_title("مقایسه عملکرد الگوریتم‌ها")

            display_columns = [
                "الگوریتم",
                "دقت",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
                "Balanced Accuracy",
                "زمان آموزش (ثانیه)"
            ]

            formatted = results[display_columns].copy()

            for col in [
                "دقت",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
                "Balanced Accuracy"
            ]:
                formatted[col] = formatted[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "ناموجود"
                )

            formatted["زمان آموزش (ثانیه)"] = (
                formatted["زمان آموزش (ثانیه)"].apply(
                    lambda x: f"{x:.2f}"
                )
            )

            st.dataframe(
                make_styler(formatted),
                use_container_width=True,
                hide_index=True
            )

            best_model = results.iloc[0]

            st.markdown(f"""
            <div class="panel">
                <div class="panel-title">🏆 بهترین الگوریتم بر اساس F1</div>
                <h2 style="color:#079B91 !important;">
                    {best_model['الگوریتم']}
                </h2>
                <p style="color:#263746 !important;">
                    امتیاز F1:
                    <b>{best_model['F1']:.4f}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            section_title("نمودار مقایسه الگوریتم‌ها")

            chart_df = results.set_index("الگوریتم")[
                ["دقت", "Precision", "Recall", "F1"]
            ]

            st.bar_chart(
                chart_df,
                use_container_width=True
            )

            section_title("ماتریس درهم‌ریختگی")

            selected_for_matrix = st.selectbox(
                "الگوریتم موردنظر برای مشاهده ماتریس:",
                list(st.session_state.trained_models.keys())
            )

            model = st.session_state.trained_models[
                selected_for_matrix
            ]

            predictions = model.predict(st.session_state.X_test)

            cm = confusion_matrix(
                st.session_state.y_test,
                predictions,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(figsize=(5, 3.8))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")

            image = ax.imshow(
                cm,
                interpolation="nearest",
                cmap="GnBu"
            )

            ax.set_title(
                "Confusion Matrix",
                color="#173B57",
                fontsize=13,
                fontweight="bold"
            )

            ax.set_xlabel("Predicted Label", color="#263746")
            ax.set_ylabel("Actual Label", color="#263746")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(
                ["Negative", "Positive"],
                color="#263746"
            )
            ax.set_yticklabels(
                ["Negative", "Positive"],
                color="#263746"
            )

            threshold = cm.max() / 2 if cm.max() > 0 else 0

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="#102D43" if cm[i, j] < threshold else "#FFFFFF",
                        fontsize=13,
                        fontweight="bold"
                    )

            fig.colorbar(image, ax=ax)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.error(str(e))


# =========================================================
# پیش‌بینی بیمار
# =========================================================

elif page == "پیش‌بینی بیمار":

    section_title(
        "ارزیابی اطلاعات بیمار",
        "اطلاعات زیر را وارد کنید تا مدل انتخاب‌شده نتیجه غربالگری را ارائه دهد."
    )

    if st.session_state.trained_models is None:
        st.warning(
            "ابتدا از بخش «آموزش و ارزیابی مدل‌ها» الگوریتم‌ها را آموزش دهید."
        )

    else:

        model_names = list(
            st.session_state.trained_models.keys()
        )

        selected_algorithm = st.selectbox(
            "🤖 الگوریتم مورد استفاده برای پیش‌بینی",
            model_names
        )

        st.markdown("""
        <div class="panel">
            <div class="panel-title">👤 اطلاعات فردی و پزشکی</div>
            <p class="panel-description">
                اطلاعات را تا حد امکان دقیق وارد کنید.
                نتیجه صرفاً خروجی مدل یادگیری ماشین است.
            </p>
        </div>
        """, unsafe_allow_html=True)

        patient_data = {}

        if dataset_name == "دیتاست Kaggle":

            col1, col2 = st.columns(2)

            with col1:
                patient_data["gender"] = st.selectbox(
                    "جنسیت",
                    ["Male", "Female"]
                )

                patient_data["age"] = st.number_input(
                    "سن",
                    min_value=1,
                    max_value=120,
                    value=35
                )

                patient_data["hypertension"] = st.selectbox(
                    "سابقه فشار خون بالا",
                    [0, 1],
                    format_func=lambda x: "دارد" if x == 1 else "ندارد"
                )

                patient_data["heart_disease"] = st.selectbox(
                    "سابقه بیماری قلبی",
                    [0, 1],
                    format_func=lambda x: "دارد" if x == 1 else "ندارد"
                )

            with col2:
                patient_data["smoking_history"] = st.selectbox(
                    "سابقه مصرف دخانیات",
                    [
                        "never",
                        "No Info",
                        "current",
                        "former",
                        "ever",
                        "not current"
                    ]
                )

                patient_data["bmi"] = st.number_input(
                    "شاخص توده بدنی (BMI)",
                    min_value=10.0,
                    max_value=80.0,
                    value=25.0,
                    step=0.1
                )

                if mode == "همه ویژگی‌ها":
                    patient_data["HbA1c_level"] = st.number_input(
                        "سطح HbA1c",
                        min_value=3.0,
                        max_value=20.0,
                        value=5.5,
                        step=0.1
                    )

                    patient_data["blood_glucose_level"] = st.number_input(
                        "قند خون",
                        min_value=50,
                        max_value=500,
                        value=100
                    )

        else:

            col1, col2 = st.columns(2)

            with col1:
                patient_data["HighBP"] = st.selectbox(
                    "فشار خون بالا",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["HighChol"] = st.selectbox(
                    "کلسترول بالا",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["CholCheck"] = st.selectbox(
                    "آزمایش کلسترول در ۵ سال گذشته",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["BMI"] = st.number_input(
                    "شاخص توده بدنی",
                    min_value=10,
                    max_value=80,
                    value=25
                )

                patient_data["Smoker"] = st.selectbox(
                    "مصرف سیگار",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["Stroke"] = st.selectbox(
                    "سابقه سکته مغزی",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["HeartDiseaseorAttack"] = st.selectbox(
                    "سابقه بیماری قلبی یا حمله قلبی",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["PhysActivity"] = st.selectbox(
                    "فعالیت بدنی منظم",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["Fruits"] = st.selectbox(
                    "مصرف روزانه میوه",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["Veggies"] = st.selectbox(
                    "مصرف روزانه سبزیجات",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["HvyAlcoholConsump"] = st.selectbox(
                    "مصرف زیاد نوشیدنی الکلی",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

            with col2:
                patient_data["AnyHealthcare"] = st.selectbox(
                    "دسترسی به خدمات درمانی",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["NoDocbcCost"] = st.selectbox(
                    "عدم مراجعه به پزشک به دلیل هزینه",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["GenHlth"] = st.selectbox(
                    "وضعیت عمومی سلامت",
                    [1, 2, 3, 4, 5],
                    format_func=lambda x: {
                        1: "عالی",
                        2: "خیلی خوب",
                        3: "خوب",
                        4: "متوسط",
                        5: "ضعیف"
                    }[x]
                )

                patient_data["MentHlth"] = st.number_input(
                    "تعداد روزهای سلامت روان نامناسب در ماه",
                    min_value=0,
                    max_value=30,
                    value=0
                )

                patient_data["PhysHlth"] = st.number_input(
                    "تعداد روزهای سلامت جسمی نامناسب در ماه",
                    min_value=0,
                    max_value=30,
                    value=0
                )

                patient_data["DiffWalk"] = st.selectbox(
                    "مشکل در راه رفتن یا بالا رفتن از پله",
                    [0, 1],
                    format_func=lambda x: "بله" if x else "خیر"
                )

                patient_data["Sex"] = st.selectbox(
                    "جنسیت",
                    [0, 1],
                    format_func=lambda x: "زن" if x == 0 else "مرد"
                )

                patient_data["Age"] = st.selectbox(
                    "گروه سنی",
                    list(range(1, 14)),
                    format_func=lambda x: f"گروه {x}"
                )

                patient_data["Education"] = st.selectbox(
                    "سطح تحصیلات",
                    list(range(1, 7))
                )

                patient_data["Income"] = st.selectbox(
                    "سطح درآمد",
                    list(range(1, 9))
                )

        st.markdown("---")

        if st.button(
            "🔍 انجام پیش‌بینی با الگوریتم انتخاب‌شده",
            use_container_width=True
        ):

            try:
                model = st.session_state.trained_models[
                    selected_algorithm
                ]

                input_df = pd.DataFrame([patient_data])

                expected_columns = st.session_state.X_test.columns
                input_df = input_df.reindex(
                    columns=expected_columns
                )

                prediction = int(model.predict(input_df)[0])

                score = None

                if hasattr(model, "predict_proba"):
                    try:
                        score = float(
                            model.predict_proba(input_df)[0][1]
                        )
                    except Exception:
                        score = None

                st.session_state.last_prediction = {
                    "algorithm": selected_algorithm,
                    "prediction": prediction,
                    "score": score
                }

            except Exception as e:
                st.error(f"خطا در پیش‌بینی: {e}")

        result = st.session_state.last_prediction

        if result is not None:

            if result["prediction"] == 1:

                st.markdown("""
                <div style="
                    background:#FFF0F0;
                    border:1px solid #F2C6CA;
                    border-radius:16px;
                    padding:22px;
                    text-align:center;
                    margin-top:20px;
                    direction:rtl;">
                    <div style="font-size:36px;color:#C43D4B;">⚠️</div>
                    <h2 style="color:#B83243 !important;">
                        نتیجه مدل: گروه مثبت
                    </h2>
                    <p style="color:#71343B !important;">
                        مدل، فرد را در گروه مثبت از نظر برچسب دیابت این دیتاست قرار داده است.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div style="
                    background:#EAF8F1;
                    border:1px solid #BDE5D0;
                    border-radius:16px;
                    padding:22px;
                    text-align:center;
                    margin-top:20px;
                    direction:rtl;">
                    <div style="font-size:36px;color:#16865B;">✓</div>
                    <h2 style="color:#14764F !important;">
                        نتیجه مدل: گروه منفی
                    </h2>
                    <p style="color:#285B46 !important;">
                        مدل، فرد را در گروه منفی از نظر برچسب دیابت این دیتاست قرار داده است.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"<p style='color:#263746;text-align:center;margin-top:15px;'>"
                f"الگوریتم استفاده‌شده: <b>{result['algorithm']}</b></p>",
                unsafe_allow_html=True
            )

            if result["score"] is not None:
                st.metric(
                    "امتیاز خروجی مدل برای کلاس مثبت",
                    f"{result['score'] * 100:.2f}%"
                )

                st.caption(
                    "این عدد امتیاز خروجی مدل است و لزوماً احتمال بالینیِ کالیبره‌شده نیست."
                )

            st.warning(
                "این نتیجه صرفاً خروجی یک مدل پژوهشی و غربالگری است؛ "
                "تشخیص پزشکی محسوب نمی‌شود. برای تشخیص قطعی باید آزمایش‌ها "
                "و نظر پزشک بررسی شوند."
            )


# =========================================================
# درباره پروژه
# =========================================================

elif page == "درباره پروژه":

    section_title("معرفی سامانه")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">🩺 سامانه هوشمند غربالگری دیابت</div>
        <p class="panel-description">
            این پروژه با هدف بررسی کاربرد الگوریتم‌های یادگیری ماشین
            در تحلیل داده‌های سلامت و شناسایی الگوهای مرتبط با دیابت
            طراحی شده است.
        </p>
        <p style="color:#263746 !important;">
            در این سامانه، داده‌های پزشکی پس از پیش‌پردازش در اختیار
            چند الگوریتم یادگیری ماشین قرار می‌گیرند. عملکرد الگوریتم‌ها
            با معیارهای مختلف ارزیابی و با یکدیگر مقایسه می‌شود.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_title("الگوریتم‌های مورد استفاده")

    algorithms = [
        ("رگرسیون لجستیک", "مدل پایه برای دسته‌بندی داده‌ها"),
        ("درخت تصمیم", "مدل قابل تفسیر برای تصمیم‌گیری"),
        ("جنگل تصادفی", "ترکیب چندین درخت تصمیم"),
        ("درخت‌های بسیار تصادفی", "روش ensemble مبتنی بر درخت"),
        ("گرادیان بوستینگ", "مدل تقویتی برای یادگیری الگوها"),
        ("XGBoost", "الگوریتم تقویتی پیشرفته، در صورت نصب بودن کتابخانه")
    ]

    for name, description in algorithms:
        st.markdown(f"""
        <div class="panel" style="padding:14px 18px;">
            <div class="panel-title" style="font-size:15px;">
                🧠 {name}
            </div>
            <div class="panel-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)

    section_title("معیارهای ارزیابی")

    st.markdown("""
    - Accuracy: میزان پیش‌بینی‌های صحیح
    - Precision: دقت پیش‌بینی کلاس مثبت
    - Recall: توانایی شناسایی نمونه‌های مثبت
    - F1-score: میانگین هماهنگ Precision و Recall
    - ROC-AUC: توانایی تفکیک دو کلاس
    - PR-AUC: ارزیابی عملکرد کلاس مثبت
    - Balanced Accuracy: میانگین عملکرد روی دو کلاس
    """)

    st.warning(
        "عملکرد مناسب روی یک دیتاست به‌تنهایی نشان‌دهنده اعتبار بالینی مدل نیست. "
        "برای استفاده پزشکی به اعتبارسنجی مستقل و بررسی تخصصی نیاز است."
    )


# =========================================================
# فوتر
# =========================================================

st.markdown("""
<div class="footer">
    سامانه پژوهشی غربالگری دیابت | طراحی‌شده با Python و Streamlit
    <br>
    Machine Learning Based Diabetes Screening
</div>
""", unsafe_allow_html=True)
