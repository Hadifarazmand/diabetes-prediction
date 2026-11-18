
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier


# =========================================================
# تنظیمات صفحه
# =========================================================

st.set_page_config(
    page_title="سامانه هوشمند تحلیل دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
DATA_FILE = DATA_DIR / "cdc_diabetes_health_indicators.csv"

TARGET = "Diabetes_012"

FEATURES = [
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

CLASS_NAMES = {
    0: "دیابت ندارد",
    1: "در محدوده پیش‌دیابت",
    2: "دیابت دارد"
}


# =========================================================
# طراحی ظاهری
# =========================================================

st.markdown("""
<style>

/* رنگ‌های اصلی */
:root {
    --navy: #173B57;
    --navy-dark: #102D43;
    --teal: #087F78;
    --teal-light: #E7F5F3;
    --page: #F3F6F9;
    --white: #FFFFFF;
    --text: #263746;
    --muted: #596B7B;
    --border: #D9E3EA;
}

/* پس‌زمینه */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F3F6F9 !important;
    color: #263746 !important;
}

.block-container {
    max-width: 1400px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* فونت */
html, body, [class*="st-"] {
    font-family: Tahoma, Arial, sans-serif !important;
}

/* متن‌های اصلی */
[data-testid="stMarkdownContainer"] {
    color: #263746 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong {
    color: #263746 !important;
    line-height: 1.9 !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #173B57 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #596B7B !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #173B57 !important;
    font-weight: 700 !important;
}

/* هدر */
.hero {
    background: linear-gradient(120deg, #173B57, #205D75, #087F78);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 24px;
    direction: rtl;
    text-align: right;
    box-shadow: 0 8px 24px rgba(23,59,87,0.13);
}

.hero-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    background: #FFFFFF;
    color: #087F78 !important;
    border-radius: 16px;
    font-size: 30px;
    margin-bottom: 12px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 26px;
    font-weight: 900;
    margin: 0 0 10px 0;
}

.hero p {
    color: #F0F8FA !important;
    font-size: 14px;
    margin: 0;
}

/* عنوان بخش */
.section-title {
    color: #173B57 !important;
    font-size: 19px;
    font-weight: 900;
    margin: 23px 0 12px;
    text-align: right;
    direction: rtl;
}

/* کارت سفید */
.panel {
    background: #FFFFFF !important;
    border: 1px solid #D9E3EA;
    border-radius: 16px;
    padding: 19px;
    margin-bottom: 15px;
    box-shadow: 0 3px 12px rgba(23,59,87,0.04);
    direction: rtl;
    text-align: right;
}

.panel-title {
    color: #173B57 !important;
    font-size: 16px;
    font-weight: 900;
    margin-bottom: 8px;
}

.panel-text {
    color: #526577 !important;
    font-size: 13px;
    line-height: 2;
}

/* کارت نتیجه */
.result-card {
    border-radius: 17px;
    padding: 24px;
    margin: 18px 0;
    text-align: center;
    direction: rtl;
}

.result-card h2 {
    font-size: 23px;
    font-weight: 900;
    margin: 9px 0;
}

.result-card p {
    font-size: 14px;
    line-height: 2;
    margin: 0;
}

.result-card .result-icon {
    font-size: 38px;
    display: block;
    margin-bottom: 8px;
}

/* سه وضعیت نتیجه */
.result-negative {
    background: #EAF7EF !important;
    border: 2px solid #A9D8BA !important;
}

.result-negative h2,
.result-negative p {
    color: #145C36 !important;
}

.result-negative .result-icon {
    color: #16834D !important;
}

.result-borderline {
    background: #FFF7DF !important;
    border: 2px solid #E9D18A !important;
}

.result-borderline h2,
.result-borderline p {
    color: #704A0B !important;
}

.result-borderline .result-icon {
    color: #A66A08 !important;
}

.result-positive {
    background: #FCEBEC !important;
    border: 2px solid #E9B5BA !important;
}

.result-positive h2,
.result-positive p {
    color: #8F2430 !important;
}

.result-positive .result-icon {
    color: #BD3444 !important;
}

/* کارت‌های آمار */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #D9E3EA;
    border-radius: 15px;
    padding: 16px;
    min-height: 112px;
    direction: rtl;
    text-align: right;
}

.stat-icon {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 37px;
    height: 37px;
    background: #E5F3F2;
    color: #087F78 !important;
    border-radius: 11px;
    font-size: 19px;
    margin-bottom: 8px;
}

.stat-label {
    color: #596B7B !important;
    font-size: 12px;
}

.stat-value {
    color: #173B57 !important;
    font-size: 22px;
    font-weight: 900;
    margin-top: 5px;
}

/* دکمه‌ها */
.stButton > button,
.stFormSubmitButton > button {
    background: #087F78 !important;
    color: #FFFFFF !important;
    border: 1px solid #087F78 !important;
    border-radius: 10px !important;
    min-height: 44px;
    font-weight: 800 !important;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #065F59 !important;
    border-color: #065F59 !important;
}

/* ورودی‌ها */
input, textarea {
    background: #FFFFFF !important;
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
    border-color: #C9D6E0 !important;
}

input::placeholder,
textarea::placeholder {
    color: #8292A0 !important;
}

[data-baseweb="input"] {
    background: #FFFFFF !important;
}

[data-baseweb="input"] input {
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
}

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #263746 !important;
    border-color: #C9D6E0 !important;
}

[data-baseweb="select"] span {
    color: #263746 !important;
}

[data-baseweb="select"] svg {
    fill: #173B57 !important;
    color: #173B57 !important;
}

/* فهرست کشویی */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background: #FFFFFF !important;
    color: #263746 !important;
}

[role="option"]:hover {
    background: #E7F5F3 !important;
    color: #173B57 !important;
}

/* رادیو و چک‌باکس */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #263746 !important;
}

[data-testid="stRadio"] svg,
[data-testid="stCheckbox"] svg {
    color: #087F78 !important;
    fill: #087F78 !important;
}

/* منوی کناری */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #173B57 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #D0DFE8 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #263746 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] svg {
    color: #65E0D4 !important;
    fill: #65E0D4 !important;
}

/* آیکون‌های رابط */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons {
    color: #087F78 !important;
    fill: #087F78 !important;
}

/* تب‌ها */
button[data-baseweb="tab"] {
    color: #526577 !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #087F78 !important;
    border-bottom-color: #087F78 !important;
}

/* پیام‌های هشدار */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}

[data-testid="stAlert"] p {
    color: #263746 !important;
}

/* جدول */
[data-testid="stDataFrame"] {
    border: 1px solid #D9E3EA !important;
    border-radius: 10px !important;
}

/* بازشونده */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #D9E3EA !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #173B57 !important;
}

/* جداکننده */
hr {
    border-color: #D9E3EA !important;
}

/* فوتر */
.footer {
    text-align: center;
    color: #596B7B !important;
    border-top: 1px solid #D9E3EA;
    padding-top: 18px;
    margin-top: 35px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# توابع طراحی
# =========================================================

def header():
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🩺</div>
        <h1>سامانه هوشمند تحلیل و غربالگری دیابت</h1>
        <p>
            بررسی اطلاعات سلامت با الگوریتم‌های یادگیری ماشین
        </p>
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def stat(icon, label, value):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">{icon}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def show_result(prediction):
    if prediction == 0:
        st.markdown("""
        <div class="result-card result-negative">
            <span class="result-icon">✓</span>
            <h2>دیابت ندارد</h2>
            <p>
                بر اساس اطلاعات واردشده، مدل فرد را در گروه بدون دیابت قرار داده است.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown("""
        <div class="result-card result-borderline">
            <span class="result-icon">⚠</span>
            <h2>در محدوده پیش‌دیابت</h2>
            <p>
                بر اساس اطلاعات واردشده، مدل فرد را در گروه پیش‌دیابت قرار داده است.
                بررسی آزمایشگاهی و مشورت با پزشک توصیه می‌شود.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-card result-positive">
            <span class="result-icon">!</span>
            <h2>دیابت دارد</h2>
            <p>
                بر اساس اطلاعات واردشده، مدل فرد را در گروه دیابت قرار داده است.
                برای بررسی دقیق‌تر باید به پزشک مراجعه شود.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# بارگذاری دیتاست
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)

    else:
        try:
            from ucimlrepo import fetch_ucirepo

            dataset = fetch_ucirepo(id=891)

            X = dataset.data.features.copy()
            y = dataset.data.targets.copy()

            df = pd.concat(
                [
                    X.reset_index(drop=True),
                    y.reset_index(drop=True)
                ],
                axis=1
            )

        except Exception as e:
            raise FileNotFoundError(
                "فایل دیتاست CDC پیدا نشد. "
                "فایل cdc_diabetes_health_indicators.csv را در پوشه dataset قرار دهید."
            ) from e

    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            "ستون Diabetes_012 پیدا نشد. "
            "برای تحلیل سه‌حالته، دیتاست باید برچسب‌های 0 و 1 و 2 را داشته باشد."
        )

    missing = [c for c in FEATURES if c not in df.columns]

    if missing:
        raise ValueError(
            f"ستون‌های موردنیاز در دیتاست وجود ندارند: {missing}"
        )

    df = df[FEATURES + [TARGET]].copy()

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    df = df[df[TARGET].isin([0, 1, 2])]

    return df


# =========================================================
# تعریف الگوریتم‌ها
# =========================================================

def create_models():

    return {
        "رگرسیون لجستیک": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42
        ),

        "درخت تصمیم": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

        "جنگل تصادفی": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        "درخت‌های بسیار تصادفی": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "گرادیان تقویتی": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=20,
            random_state=42
        )
    }


def create_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])


# =========================================================
# آموزش یک الگوریتم
# =========================================================

def train_one_model(model_name, X_train, y_train):

    models = create_models()

    if model_name not in models:
        raise ValueError("الگوریتم انتخاب‌شده معتبر نیست.")

    pipeline = create_pipeline(models[model_name])

    pipeline.fit(X_train, y_train)

    return pipeline


# =========================================================
# ارزیابی همه الگوریتم‌ها
# =========================================================

def train_and_evaluate_all(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = create_models()

    trained_models = {}
    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(models)

    for index, (name, model) in enumerate(models.items()):

        status.info(f"در حال آموزش {name} ...")

        start_time = time.time()

        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)

        elapsed = time.time() - start_time

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        error_rate = 1 - accuracy

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        balanced = balanced_accuracy_score(
            y_test,
            predictions
        )

        results.append({
            "الگوریتم": name,
            "درصد درستی": accuracy * 100,
            "درصد خطا": error_rate * 100,
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1": f1 * 100,
            "Balanced Accuracy": balanced * 100,
            "زمان آموزش": elapsed
        })

        trained_models[name] = pipeline

        progress.progress((index + 1) / total)

    status.success("آموزش و ارزیابی الگوریتم‌ها کامل شد.")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    return (
        trained_models,
        results_df,
        X_test,
        y_test
    )


# =========================================================
# وضعیت حافظه
# =========================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "patient_result" not in st.session_state:
    st.session_state.patient_result = None

if "patient_model_name" not in st.session_state:
    st.session_state.patient_model_name = None

if "data_signature" not in st.session_state:
    st.session_state.data_signature = None


# =========================================================
# منوی کناری
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:15px 5px 20px;">
        <div style="
            width:68px;height:68px;margin:auto;
            display:flex;align-items:center;justify-content:center;
            background:#FFFFFF;color:#087F78;
            border-radius:18px;font-size:32px;">
            🩺
        </div>
        <h3 style="color:#FFFFFF !important;margin-top:13px;">
            پنل هوشمند دیابت
        </h3>
        <p style="color:#D0DFE8 !important;font-size:12px;">
            سامانه تحلیل داده‌های سلامت
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "انتخاب بخش",
        [
            "تحلیل اطلاعات بیمار",
            "تحلیل و مقایسه الگوریتم‌ها",
            "درباره پروژه"
        ]
    )

    st.markdown("---")

    st.markdown("""
    <p style="color:#FFFFFF !important;font-size:12px;text-align:center;">
        Diabetes Prediction System
    </p>
    """, unsafe_allow_html=True)


# =========================================================
# هدر
# =========================================================

header()


# =========================================================
# آماده‌سازی داده‌ها
# =========================================================

try:
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    signature = (
        len(df),
        int(df[TARGET].sum()),
        str(df[TARGET].value_counts().sort_index().to_dict())
    )

    if st.session_state.data_signature != signature:
        st.session_state.trained_models = {}
        st.session_state.results_df = None
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.session_state.patient_result = None
        st.session_state.data_signature = signature

except Exception as e:
    st.error(str(e))
    st.stop()


# =========================================================
# بخش اول: تحلیل اطلاعات بیمار
# =========================================================

if page == "تحلیل اطلاعات بیمار":

    section(
        "اطلاعات بیمار و انتخاب الگوریتم"
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-title">👤 فرم اطلاعات سلامت</div>
        <p class="panel-text">
            اطلاعات فرد را وارد کنید، الگوریتم موردنظر را انتخاب کنید
            و دکمه تحلیل اطلاعات را بزنید.
        </p>
    </div>
    """, unsafe_allow_html=True)

    algorithm_names = list(create_models().keys())

    selected_algorithm = st.selectbox(
        "🤖 الگوریتم موردنظر برای تحلیل",
        algorithm_names,
        key="patient_algorithm"
    )

    section("مشخصات و سوابق سلامت")

    patient = {}

    col1, col2 = st.columns(2)

    with col1:

        patient["Sex"] = st.selectbox(
            "جنسیت",
            [0, 1],
            format_func=lambda x: "زن" if x == 0 else "مرد"
        )

        patient["Age"] = st.selectbox(
            "گروه سنی",
            list(range(1, 14)),
            format_func=lambda x: f"گروه سنی {x}"
        )

        patient["BMI"] = st.number_input(
            "شاخص توده بدنی (BMI)",
            min_value=10,
            max_value=80,
            value=25
        )

        patient["HighBP"] = st.selectbox(
            "آیا فشار خون بالا دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["HighChol"] = st.selectbox(
            "آیا کلسترول خون بالا دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["CholCheck"] = st.selectbox(
            "آیا در ۵ سال گذشته آزمایش کلسترول داده‌اید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["Smoker"] = st.selectbox(
            "آیا حداقل ۱۰۰ نخ سیگار در طول عمر مصرف کرده‌اید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["Stroke"] = st.selectbox(
            "آیا سابقه سکته مغزی دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["HeartDiseaseorAttack"] = st.selectbox(
            "آیا سابقه بیماری قلبی یا حمله قلبی دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["PhysActivity"] = st.selectbox(
            "آیا فعالیت بدنی منظم دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["Fruits"] = st.selectbox(
            "آیا روزانه میوه مصرف می‌کنید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

    with col2:

        patient["Veggies"] = st.selectbox(
            "آیا روزانه سبزیجات مصرف می‌کنید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["HvyAlcoholConsump"] = st.selectbox(
            "آیا مصرف زیاد نوشیدنی الکلی دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["AnyHealthcare"] = st.selectbox(
            "آیا به خدمات درمانی دسترسی دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["NoDocbcCost"] = st.selectbox(
            "آیا به دلیل هزینه از مراجعه به پزشک خودداری کرده‌اید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["GenHlth"] = st.selectbox(
            "وضعیت عمومی سلامت شما",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "عالی",
                2: "خیلی خوب",
                3: "خوب",
                4: "متوسط",
                5: "ضعیف"
            }[x]
        )

        patient["MentHlth"] = st.number_input(
            "تعداد روزهای سلامت روان نامناسب در ماه",
            min_value=0,
            max_value=30,
            value=0
        )

        patient["PhysHlth"] = st.number_input(
            "تعداد روزهای سلامت جسمی نامناسب در ماه",
            min_value=0,
            max_value=30,
            value=0
        )

        patient["DiffWalk"] = st.selectbox(
            "آیا در راه رفتن یا بالا رفتن از پله مشکل دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x else "خیر"
        )

        patient["Education"] = st.selectbox(
            "سطح تحصیلات",
            list(range(1, 7)),
            format_func=lambda x: {
                1: "کمتر از کلاس اول",
                2: "کلاس اول تا هشتم",
                3: "کلاس نهم تا یازدهم",
                4: "دیپلم",
                5: "دانشگاه ناقص",
                6: "دانشگاه کامل"
            }[x]
        )

        patient["Income"] = st.selectbox(
            "سطح درآمد",
            list(range(1, 9)),
            format_func=lambda x: f"سطح {x}"
        )

    st.markdown("---")

    if st.button(
        "🔍 تحلیل اطلاعات بیمار",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "در حال بررسی اطلاعات و اجرای الگوریتم..."
            ):

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )

                if selected_algorithm not in st.session_state.trained_models:

                    model = train_one_model(
                        selected_algorithm,
                        X_train,
                        y_train
                    )

                    st.session_state.trained_models[
                        selected_algorithm
                    ] = model

                else:
                    model = st.session_state.trained_models[
                        selected_algorithm
                    ]

                input_df = pd.DataFrame(
                    [patient],
                    columns=FEATURES
                )

                prediction = int(model.predict(input_df)[0])

                st.session_state.patient_result = prediction
                st.session_state.patient_model_name = selected_algorithm

        except Exception as e:
            st.error(f"خطا در تحلیل اطلاعات: {e}")

    if st.session_state.patient_result is not None:

        section("نتیجه تحلیل بیمار")

        show_result(
            st.session_state.patient_result
        )

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">اطلاعات تحلیل</div>
                <p class="panel-text">
                    الگوریتم استفاده‌شده:
                    <b>{st.session_state.patient_model_name}</b>
                </p>
                <p class="panel-text">
                    نتیجه بر اساس مدل آموزش‌دیده و اطلاعات واردشده محاسبه شده است.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "این خروجی یک پیش‌بینی پژوهشی بر اساس داده‌های مدل است و تشخیص پزشکی قطعی نیست. "
            "برای تشخیص دیابت باید آزمایش‌های معتبر مانند قند خون و HbA1c "
            "و نظر پزشک بررسی شوند."
        )


# =========================================================
# بخش دوم: مقایسه الگوریتم‌ها
# =========================================================

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    section(
        "آموزش و مقایسه عملکرد الگوریتم‌ها"
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-title">📊 مقایسه علمی الگوریتم‌ها</div>
        <p class="panel-text">
            الگوریتم‌ها روی یک مجموعه داده آموزشی و آزمون یکسان ارزیابی می‌شوند.
            سپس درصد درستی، درصد خطا و سایر معیارها برای هر الگوریتم نمایش داده می‌شود.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "🚀 آموزش و ارزیابی همه الگوریتم‌ها",
        use_container_width=True
    ):

        try:

            (
                trained_models,
                results_df,
                X_test,
                y_test
            ) = train_and_evaluate_all(X, y)

            st.session_state.trained_models = trained_models
            st.session_state.results_df = results_df
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.success(
                "همه الگوریتم‌ها با موفقیت ارزیابی شدند."
            )

        except Exception as e:
            st.error(f"خطا در آموزش الگوریتم‌ها: {e}")

    if st.session_state.results_df is not None:

        results = st.session_state.results_df

        section("انتخاب الگوریتم برای مشاهده جزئیات")

        selected = st.selectbox(
            "الگوریتم موردنظر را انتخاب کنید",
            results["الگوریتم"].tolist(),
            key="comparison_algorithm"
        )

        selected_row = results[
            results["الگوریتم"] == selected
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            stat(
                "✓",
                "درصد درستی",
                f"{selected_row['درصد درستی']:.2f}%"
            )

        with c2:
            stat(
                "×",
                "درصد خطا",
                f"{selected_row['درصد خطا']:.2f}%"
            )

        with c3:
            stat(
                "★",
                "امتیاز F1",
                f"{selected_row['F1']:.2f}%"
            )

        section("سایر معیارهای الگوریتم انتخاب‌شده")

        detail_df = pd.DataFrame({
            "معیار": [
                "Precision",
                "Recall",
                "F1",
                "Balanced Accuracy",
                "زمان آموزش"
            ],
            "مقدار": [
                f"{selected_row['Precision']:.2f}%",
                f"{selected_row['Recall']:.2f}%",
                f"{selected_row['F1']:.2f}%",
                f"{selected_row['Balanced Accuracy']:.2f}%",
                f"{selected_row['زمان آموزش']:.2f} ثانیه"
            ]
        })

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )

        section("جدول مقایسه تمام الگوریتم‌ها")

        display_df = results.copy()

        for col in [
            "درصد درستی",
            "درصد خطا",
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy"
        ]:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}%"
            )

        display_df["زمان آموزش"] = display_df[
            "زمان آموزش"
        ].apply(lambda x: f"{x:.2f} ثانیه")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        section("نمودار مقایسه درصد درستی")

        chart = results.set_index("الگوریتم")[
            ["درصد درستی", "درصد خطا"]
        ]

        st.bar_chart(
            chart,
            use_container_width=True
        )

        section("ماتریس درهم‌ریختگی الگوریتم انتخاب‌شده")

        model = st.session_state.trained_models[selected]

        predictions = model.predict(st.session_state.X_test)

        cm = confusion_matrix(
            st.session_state.y_test,
            predictions,
            labels=[0, 1, 2]
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        im = ax.imshow(
            cm,
            cmap="GnBu",
            interpolation="nearest"
        )

        ax.set_title(
            "Confusion Matrix",
            color="#173B57",
            fontweight="bold"
        )

        ax.set_xlabel(
            "Predicted Class",
            color="#263746"
        )

        ax.set_ylabel(
            "Actual Class",
            color="#263746"
        )

        labels = [
            "No Diabetes",
            "Prediabetes",
            "Diabetes"
        ]

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])

        ax.set_xticklabels(labels, rotation=15, color="#263746")
        ax.set_yticklabels(labels, color="#263746")

        threshold = cm.max() / 2 if cm.max() > 0 else 0

        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="#102D43" if cm[i, j] < threshold else "#FFFFFF",
                    fontweight="bold",
                    fontsize=11
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "درصد درستی و خطا از مجموعه آزمون محاسبه شده‌اند؛ "
            "این اعداد با وارد کردن یک بیمار جدید تغییر نمی‌کنند. "
            "ورودی بیمار فقط نتیجه پیش‌بینی همان فرد را تغییر می‌دهد."
        )

    else:

        st.info(
            "برای مشاهده درصد درستی و خطای الگوریتم‌ها، "
            "ابتدا دکمه آموزش و ارزیابی همه الگوریتم‌ها را بزنید."
        )


# =========================================================
# بخش سوم: درباره پروژه
# =========================================================

elif page == "درباره پروژه":

    section("معرفی سامانه")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">سامانه هوشمند تحلیل و غربالگری دیابت</div>
        <p class="panel-text">
            این سامانه با استفاده از الگوریتم‌های یادگیری ماشین،
            اطلاعات مرتبط با سلامت افراد را بررسی کرده و آن‌ها را
            در یکی از سه گروه بدون دیابت، پیش‌دیابت و دیابت دسته‌بندی می‌کند.
        </p>
        <p class="panel-text">
            همچنین امکان آموزش، ارزیابی و مقایسه چند الگوریتم
            یادگیری ماشین برای بررسی عملکرد آن‌ها فراهم شده است.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section("الگوریتم‌های مورد استفاده")

    for name in create_models().keys():
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">🧠 {name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    section("معیارهای ارزیابی")

    st.markdown("""
    - **درصد درستی:** سهم پیش‌بینی‌های صحیح از کل پیش‌بینی‌ها.
    - **درصد خطا:** سهم پیش‌بینی‌های نادرست از کل پیش‌بینی‌ها.
    - **Precision:** میزان صحت پیش‌بینی کلاس‌ها.
    - **Recall:** میزان توانایی شناسایی نمونه‌های هر کلاس.
    - **F1:** معیار ترکیبی Precision و Recall.
    - **Balanced Accuracy:** میانگین عملکرد مدل روی کلاس‌ها.
    """)

    st.warning(
        "این سامانه ابزار پژوهشی و غربالگری است و جایگزین آزمایش پزشکی یا تشخیص پزشک نیست."
    )


# =========================================================
# فوتر
# =========================================================

st.markdown("""
<div class="footer">
    سامانه هوشمند تحلیل و غربالگری دیابت
    <br>
    Diabetes Machine Learning Research System
</div>
""", unsafe_allow_html=True)
