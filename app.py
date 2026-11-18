
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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =====================================================
# تنظیمات اولیه
# =====================================================

st.set_page_config(
    page_title="سامانه هوشمند تحلیل دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "diabetes_prediction_dataset.csv"

TARGET = "diabetes"

ALL_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

SCREENING_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi"
]

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_history"
]

ALGORITHMS = [
    "رگرسیون لجستیک",
    "درخت تصمیم",
    "جنگل تصادفی",
    "درخت‌های بسیار تصادفی",
    "گرادیان تقویتی",
    "XGBoost"
]


# =====================================================
# طراحی رابط کاربری
# =====================================================

st.markdown("""
<style>

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

.stApp {
    background: var(--page);
    color: var(--text);
    direction: rtl;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

html, body, [class*="st-"] {
    font-family: Tahoma, Arial, sans-serif;
}

[data-testid="stMarkdownContainer"] {
    direction: rtl;
    text-align: right;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--text);
    line-height: 1.9;
}

h1, h2, h3 {
    color: var(--navy) !important;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {
    color: var(--navy) !important;
    font-weight: 700;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
}

/* هدر اصلی */

.hero {
    background: linear-gradient(120deg, #173B57, #205D75, #087F78);
    border-radius: 22px;
    padding: 30px;
    margin-bottom: 25px;
    direction: rtl;
    text-align: right;
    box-shadow: 0 8px 25px rgba(23,59,87,.13);
}

.hero-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    background: #FFFFFF;
    color: #087F78;
    border-radius: 16px;
    font-size: 30px;
    margin-bottom: 12px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 26px;
    font-weight: 900;
    margin: 0 0 10px;
}

.hero p {
    color: #F0F8FA !important;
    font-size: 14px;
    margin: 0;
}

/* عنوان بخش‌ها */

.section-title {
    color: #173B57;
    font-size: 19px;
    font-weight: 900;
    margin: 25px 0 14px;
    text-align: right;
    direction: rtl;
}

/* پنل‌ها */

.panel {
    background: #FFFFFF;
    border: 1px solid #D9E3EA;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 3px 12px rgba(23,59,87,.04);
    direction: rtl;
    text-align: right;
}

.panel-title {
    color: #173B57;
    font-size: 16px;
    font-weight: 900;
    margin-bottom: 8px;
}

.panel-text {
    color: #526577;
    font-size: 13px;
    line-height: 2;
}

/* کارت‌های آماری */

.stat-card {
    background: #FFFFFF;
    border: 1px solid #D9E3EA;
    border-radius: 15px;
    padding: 17px;
    min-height: 115px;
    direction: rtl;
    text-align: right;
}

.stat-icon {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 38px;
    height: 38px;
    background: #E5F3F2;
    color: #087F78;
    border-radius: 11px;
    font-size: 19px;
    margin-bottom: 9px;
}

.stat-label {
    color: #596B7B;
    font-size: 12px;
}

.stat-value {
    color: #173B57;
    font-size: 23px;
    font-weight: 900;
    margin-top: 5px;
}

/* نتیجه بیمار */

.result-card {
    border-radius: 18px;
    padding: 25px;
    margin: 18px 0;
    text-align: center;
    direction: rtl;
}

.result-card h2 {
    font-size: 23px;
    font-weight: 900;
    margin: 10px 0;
}

.result-card p {
    font-size: 14px;
    line-height: 2;
    margin: 0;
}

.result-icon {
    font-size: 40px;
    display: block;
    margin-bottom: 8px;
}

.result-negative {
    background: #EAF7EF;
    border: 2px solid #A9D8BA;
}

.result-negative h2,
.result-negative p {
    color: #145C36 !important;
}

.result-positive {
    background: #FCEBEC;
    border: 2px solid #E9B5BA;
}

.result-positive h2,
.result-positive p {
    color: #8F2430 !important;
}

/* دکمه‌ها */

.stButton > button,
.stFormSubmitButton > button {
    background: #087F78 !important;
    color: #FFFFFF !important;
    border: 1px solid #087F78 !important;
    border-radius: 11px !important;
    min-height: 45px;
    font-weight: 800 !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #065F59 !important;
    border-color: #065F59 !important;
}

.stButton > button p {
    color: #FFFFFF !important;
}

/* ورودی‌ها */

input, textarea {
    background: #FFFFFF !important;
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
    border-color: #C9D6E0 !important;
}

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #263746 !important;
    border-color: #C9D6E0 !important;
}

[data-baseweb="select"] span {
    color: #263746 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background: #FFFFFF !important;
    color: #263746 !important;
}

[role="option"]:hover {
    background: #E7F5F3 !important;
}

/* سایدبار */

[data-testid="stSidebar"] {
    background: #173B57 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #D0DFE8 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #263746 !important;
}

/* تب‌ها و جدول‌ها */

button[data-baseweb="tab"] {
    color: #526577 !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #087F78 !important;
    border-bottom-color: #087F78 !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #D9E3EA;
    border-radius: 12px;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #D9E3EA;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #173B57 !important;
}

hr {
    border-color: #D9E3EA;
}

.footer {
    text-align: center;
    color: #596B7B;
    border-top: 1px solid #D9E3EA;
    padding-top: 20px;
    margin-top: 35px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# توابع ظاهری
# =====================================================

def header():
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🩺</div>
        <h1>سامانه هوشمند تحلیل و غربالگری دیابت</h1>
        <p>تحلیل اطلاعات سلامت با استفاده از الگوریتم‌های یادگیری ماشین</p>
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


def show_result(prediction, probability):
    if prediction == 0:
        st.markdown("""
        <div class="result-card result-negative">
            <span class="result-icon">✓</span>
            <h2>پیش‌بینی مدل: بدون دیابت</h2>
            <p>
                مدل بر اساس اطلاعات واردشده، نمونه را در گروه بدون دیابت
                قرار داده است. این نتیجه تشخیص قطعی پزشکی نیست.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card result-positive">
            <span class="result-icon">!</span>
            <h2>پیش‌بینی مدل: احتمال دیابت</h2>
            <p>
                مدل بر اساس اطلاعات واردشده، نمونه را در گروه دارای دیابت
                قرار داده است. برای بررسی وضعیت واقعی، آزمایش و نظر پزشک لازم است.
            </p>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        stat("📊", "احتمال پیش‌بینی کلاس دیابت", f"{probability:.2f}%")

    with c2:
        stat("✓", "احتمال پیش‌بینی کلاس بدون دیابت", f"{100-probability:.2f}%")


# =====================================================
# بارگذاری دیتاست Kaggle
# =====================================================

@st.cache_data(show_spinner=False)
def load_data(file_path, file_mtime):

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            "فایل دیتاست پیدا نشد. فایل "
            "diabetes_prediction_dataset.csv را داخل پوشه dataset قرار دهید."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = ALL_FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"ستون‌های زیر در فایل دیتاست وجود ندارند: {missing}"
        )

    df = df[required].copy()

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gender"] = df["gender"].astype("string").str.strip()
    df["smoking_history"] = (
        df["smoking_history"].astype("string").str.strip()
    )

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]

    df[TARGET] = df[TARGET].astype(int)

    df = df.drop_duplicates().reset_index(drop=True)

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "دیتاست باید هر دو کلاس 0 و 1 را داشته باشد."
        )

    if df.empty:
        raise ValueError("دیتاست پس از پاک‌سازی خالی شده است.")

    return df


# =====================================================
# پیش‌پردازش و ساخت مدل‌ها
# =====================================================

def create_preprocessor(features):

    categorical = [
        col for col in CATEGORICAL_FEATURES if col in features
    ]

    numeric = [
        col for col in features if col not in categorical
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical)
    ])


def create_models(y_train=None):

    models = {
        "رگرسیون لجستیک": LogisticRegression(
            max_iter=1500,
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
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        "درخت‌های بسیار تصادفی": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "گرادیان تقویتی": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.06,
            max_depth=3,
            random_state=42
        )
    }

    if XGBOOST_AVAILABLE:
        scale_pos_weight = 1.0

        if y_train is not None:
            negatives = int((y_train == 0).sum())
            positives = int((y_train == 1).sum())

            if positives > 0:
                scale_pos_weight = negatives / positives

        models["XGBoost"] = XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )

    return models


def create_pipeline(model, features):
    return Pipeline([
        ("preprocessor", create_preprocessor(features)),
        ("model", model)
    ])


def train_one_model(model_name, X_train, y_train, features):

    models = create_models(y_train)

    if model_name not in models:
        raise ValueError("الگوریتم انتخاب‌شده در دسترس نیست.")

    pipeline = create_pipeline(models[model_name], features)
    pipeline.fit(X_train[features], y_train)

    return pipeline


# =====================================================
# محاسبه معیارهای ارزیابی
# =====================================================

def calculate_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    return {
        "درصد درستی": accuracy * 100,
        "درصد خطا": (1 - accuracy) * 100,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ) * 100,
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ) * 100,
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ) * 100,
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ) * 100,
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ) * 100,
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ) * 100
    }


# =====================================================
# آموزش و ارزیابی همه الگوریتم‌ها
# =====================================================

def train_and_evaluate_all(X, y, features):

    X_train, X_test, y_train, y_test = train_test_split(
        X[features],
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = create_models(y_train)

    trained_models = {}
    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(models)

    for index, (name, algorithm) in enumerate(models.items()):

        status.info(f"در حال آموزش الگوریتم {name} ...")

        start_time = time.time()

        pipeline = create_pipeline(algorithm, features)
        pipeline.fit(X_train, y_train)

        elapsed = time.time() - start_time

        metrics = calculate_metrics(
            pipeline,
            X_test,
            y_test
        )

        metrics["الگوریتم"] = name
        metrics["زمان آموزش"] = elapsed

        results.append(metrics)
        trained_models[name] = pipeline

        progress.progress((index + 1) / total)

    status.success("آموزش و ارزیابی الگوریتم‌ها کامل شد.")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "F1",
        ascending=False
    ).reset_index(drop=True)

    return trained_models, results_df, X_test, y_test


# =====================================================
# مدیریت حافظه نشست
# =====================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "results" not in st.session_state:
    st.session_state.results = {}

if "patient_result" not in st.session_state:
    st.session_state.patient_result = None

if "patient_model_name" not in st.session_state:
    st.session_state.patient_model_name = None

if "patient_mode" not in st.session_state:
    st.session_state.patient_mode = None

if "patient_probability" not in st.session_state:
    st.session_state.patient_probability = None

if "data_signature" not in st.session_state:
    st.session_state.data_signature = None


# =====================================================
# منوی کناری
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:18px 5px 22px;">
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
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("""
    <p style="color:#FFFFFF !important;font-size:12px;text-align:center;">
        Diabetes Prediction System
    </p>
    """, unsafe_allow_html=True)


# =====================================================
# نمایش هدر و بارگذاری داده
# =====================================================

header()

try:

    file_mtime = DATA_FILE.stat().st_mtime if DATA_FILE.exists() else 0

    df = load_data(str(DATA_FILE), file_mtime)

    signature = (
        len(df),
        file_mtime,
        str(df[TARGET].value_counts().sort_index().to_dict())
    )

    if st.session_state.data_signature != signature:

        st.session_state.trained_models = {}
        st.session_state.results = {}
        st.session_state.patient_result = None
        st.session_state.patient_probability = None
        st.session_state.data_signature = signature

except Exception as e:
    st.error(str(e))
    st.stop()


# =====================================================
# اطلاعات کلی دیتاست
# =====================================================

with st.expander("مشاهده اطلاعات دیتاست", expanded=False):

    c1, c2, c3 = st.columns(3)

    with c1:
        stat("📁", "تعداد رکوردها", f"{len(df):,}")

    with c2:
        stat("📊", "تعداد ویژگی‌ها", len(ALL_FEATURES))

    with c3:
        stat("🎯", "تعداد کلاس‌ها", "2")

    class_counts = df[TARGET].value_counts().sort_index()

    class_df = pd.DataFrame({
        "وضعیت": ["بدون دیابت", "دارای دیابت"],
        "تعداد": [
            int(class_counts.get(0, 0)),
            int(class_counts.get(1, 0))
        ]
    })

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# بخش اول: تحلیل اطلاعات بیمار
# =====================================================

if page == "تحلیل اطلاعات بیمار":

    section("تحلیل و بررسی اطلاعات بیمار")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">👤 فرم اطلاعات سلامت</div>
        <p class="panel-text">
            ابتدا نوع تحلیل و الگوریتم موردنظر را انتخاب کنید.
            سپس اطلاعات فرد را وارد کرده و نتیجه پیش‌بینی را مشاهده کنید.
        </p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "نوع تحلیل",
        [
            "غربالگری اولیه (بدون آزمایش)",
            "تحلیل کامل (با اطلاعات آزمایشگاهی)"
        ],
        horizontal=True
    )

    if mode == "غربالگری اولیه (بدون آزمایش)":
        features = SCREENING_FEATURES
        mode_key = "screening"
    else:
        features = ALL_FEATURES
        mode_key = "full"

    algorithm_names = list(create_models().keys())

    selected_algorithm = st.selectbox(
        "🤖 انتخاب الگوریتم یادگیری ماشین",
        algorithm_names,
        key="patient_algorithm"
    )

    section("مشخصات فردی")

    col1, col2 = st.columns(2)

    with col1:

        gender_options = sorted(
            df["gender"].dropna().unique().tolist()
        )

        if not gender_options:
            gender_options = ["Female", "Male"]

        gender = st.selectbox(
            "جنسیت",
            gender_options,
            format_func=lambda x: {
                "Female": "زن",
                "Male": "مرد",
                "Other": "سایر"
            }.get(x, str(x))
        )

        age = st.number_input(
            "سن",
            min_value=1,
            max_value=100,
            value=35,
            step=1
        )

        bmi = st.number_input(
            "شاخص توده بدنی (BMI)",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=0.5
        )

        hypertension = st.selectbox(
            "آیا فشار خون بالا دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x == 1 else "خیر"
        )

    with col2:

        heart_disease = st.selectbox(
            "آیا سابقه بیماری قلبی دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x == 1 else "خیر"
        )

        smoking_options = sorted(
            df["smoking_history"].dropna().unique().tolist()
        )

        if not smoking_options:
            smoking_options = ["never", "No Info"]

        smoking = st.selectbox(
            "سابقه مصرف دخانیات",
            smoking_options,
            format_func=lambda x: {
                "never": "هرگز",
                "No Info": "اطلاعاتی ندارم",
                "current": "در حال حاضر",
                "former": "در گذشته",
                "not current": "در حال حاضر مصرف نمی‌کنم",
                "ever": "در طول زندگی"
            }.get(x, str(x))
        )

    patient = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking,
        "bmi": bmi
    }

    if mode_key == "full":

        section("اطلاعات آزمایشگاهی")

        c1, c2 = st.columns(2)

        with c1:
            patient["HbA1c_level"] = st.number_input(
                "میزان HbA1c",
                min_value=3.0,
                max_value=20.0,
                value=5.5,
                step=0.1
            )

        with c2:
            patient["blood_glucose_level"] = st.number_input(
                "قند خون (mg/dL)",
                min_value=50,
                max_value=400,
                value=100,
                step=1
            )

    else:
        st.info(
            "در این حالت، اطلاعات HbA1c و قند خون وارد نمی‌شوند "
            "و مدل فقط با شش ویژگی اولیه آموزش داده می‌شود."
        )

    st.markdown("---")

    if st.button(
        "🔍 تحلیل اطلاعات بیمار",
        use_container_width=True
    ):

        try:

            with st.spinner("در حال آموزش مدل و تحلیل اطلاعات بیمار..."):

                X = df[features]
                y = df[TARGET]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )

                model_key = f"{mode_key}_{selected_algorithm}"

                if model_key not in st.session_state.trained_models:

                    model = train_one_model(
                        selected_algorithm,
                        X_train,
                        y_train,
                        features
                    )

                    st.session_state.trained_models[model_key] = model

                else:
                    model = st.session_state.trained_models[model_key]

                input_df = pd.DataFrame(
                    [patient],
                    columns=features
                )

                prediction = int(model.predict(input_df)[0])

                probability = float(
                    model.predict_proba(input_df)[0][1]
                ) * 100

                st.session_state.patient_result = prediction
                st.session_state.patient_probability = probability
                st.session_state.patient_model_name = selected_algorithm
                st.session_state.patient_mode = mode

        except Exception as e:
            st.error(f"خطا در تحلیل اطلاعات: {e}")

    if st.session_state.patient_result is not None:

        section("نتیجه تحلیل بیمار")

        show_result(
            st.session_state.patient_result,
            st.session_state.patient_probability
        )

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">جزئیات تحلیل</div>
            <p class="panel-text">
                الگوریتم استفاده‌شده:
                <b>{st.session_state.patient_model_name}</b>
            </p>
            <p class="panel-text">
                نوع تحلیل:
                <b>{st.session_state.patient_mode}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "این سامانه یک ابزار پژوهشی مبتنی بر یادگیری ماشین است. "
            "درصد نمایش‌داده‌شده احتمال پیش‌بینی مدل است، نه احتمال قطعی ابتلای فرد. "
            "این نتیجه جایگزین آزمایش قند خون، HbA1c یا تشخیص پزشک نیست."
        )


# =====================================================
# بخش دوم: مقایسه الگوریتم‌ها
# =====================================================

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    section("آموزش و مقایسه الگوریتم‌های یادگیری ماشین")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">📊 ارزیابی عملکرد مدل‌ها</div>
        <p class="panel-text">
            الگوریتم‌ها روی داده‌های آموزشی و آزمون یکسان ارزیابی می‌شوند.
            معیارهای مختلف عملکرد برای بررسی و مقایسه آن‌ها نمایش داده می‌شود.
        </p>
    </div>
    """, unsafe_allow_html=True)

    evaluation_mode = st.radio(
        "انتخاب نوع داده برای ارزیابی",
        [
            "غربالگری اولیه (بدون آزمایش)",
            "تحلیل کامل (با اطلاعات آزمایشگاهی)"
        ],
        horizontal=True,
        key="evaluation_mode"
    )

    if evaluation_mode == "غربالگری اولیه (بدون آزمایش)":
        features = SCREENING_FEATURES
        mode_key = "screening"
    else:
        features = ALL_FEATURES
        mode_key = "full"

    if mode_key not in st.session_state.results:
        st.session_state.results[mode_key] = None

    if st.button(
        "🚀 آموزش و ارزیابی همه الگوریتم‌ها",
        use_container_width=True
    ):

        try:

            trained, results, X_test, y_test = train_and_evaluate_all(
                df[features],
                df[TARGET],
                features
            )

            st.session_state.trained_models.update({
                f"{mode_key}_{name}": model
                for name, model in trained.items()
            })

            st.session_state.results[mode_key] = {
                "results_df": results,
                "X_test": X_test,
                "y_test": y_test,
                "features": features
            }

            st.success("آموزش و ارزیابی الگوریتم‌ها با موفقیت انجام شد.")

        except Exception as e:
            st.error(f"خطا در آموزش الگوریتم‌ها: {e}")

    evaluation = st.session_state.results.get(mode_key)

    if evaluation is not None:

        results = evaluation["results_df"]

        section("انتخاب الگوریتم برای مشاهده جزئیات")

        selected = st.selectbox(
            "الگوریتم موردنظر را انتخاب کنید",
            results["الگوریتم"].tolist(),
            key=f"comparison_{mode_key}"
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

        section("معیارهای ارزیابی الگوریتم انتخاب‌شده")

        metric_names = [
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy",
            "ROC-AUC",
            "PR-AUC"
        ]

        detail_df = pd.DataFrame({
            "معیار": metric_names,
            "مقدار": [
                f"{selected_row[name]:.2f}%"
                for name in metric_names
            ]
        })

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )

        section("جدول مقایسه تمام الگوریتم‌ها")

        display_df = results.copy()

        percentage_columns = [
            "درصد درستی",
            "درصد خطا",
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy",
            "ROC-AUC",
            "PR-AUC"
        ]

        for col in percentage_columns:
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.2f}%"
            )

        display_df["زمان آموزش"] = display_df[
            "زمان آموزش"
        ].map(lambda x: f"{x:.2f} ثانیه")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        section("نمودار مقایسه عملکرد الگوریتم‌ها")

        chart = results.set_index("الگوریتم")[
            ["درصد درستی", "F1", "ROC-AUC"]
        ]

        st.bar_chart(chart, use_container_width=True)

        section("ماتریس درهم‌ریختگی الگوریتم انتخاب‌شده")

        model_key = f"{mode_key}_{selected}"
        model = st.session_state.trained_models[model_key]

        X_test = evaluation["X_test"]
        y_test = evaluation["y_test"]

        predictions = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
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

        ax.set_xlabel("Predicted Class", color="#263746")
        ax.set_ylabel("Actual Class", color="#263746")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["No Diabetes", "Diabetes"],
            color="#263746"
        )

        ax.set_yticklabels(
            ["No Diabetes", "Diabetes"],
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
                    fontweight="bold",
                    fontsize=12
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "تمام معیارها از مجموعه آزمون مستقل محاسبه شده‌اند. "
            "برای هر دو حالت غربالگری و تحلیل کامل، مدل‌ها جداگانه آموزش داده می‌شوند."
        )

    else:

        st.info(
            "برای مشاهده نتایج، ابتدا نوع تحلیل را انتخاب کنید "
            "و سپس دکمه آموزش و ارزیابی الگوریتم‌ها را بزنید."
        )


# =====================================================
# بخش سوم: درباره پروژه
# =====================================================

elif page == "درباره پروژه":

    section("معرفی سامانه")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">سامانه هوشمند تحلیل و غربالگری دیابت</div>
        <p class="panel-text">
            این پروژه با هدف بررسی کاربرد الگوریتم‌های یادگیری ماشین
            در تحلیل داده‌های مرتبط با دیابت طراحی شده است.
        </p>
        <p class="panel-text">
            سامانه با استفاده از اطلاعات فردی، سوابق سلامت و در حالت تحلیل کامل،
            اطلاعات آزمایشگاهی، یک پیش‌بینی دودسته‌ای درباره وضعیت دیابت ارائه می‌کند.
        </p>
        <p class="panel-text">
            علاوه بر پیش‌بینی، عملکرد چند الگوریتم مختلف با معیارهای آماری
            ارزیابی و مقایسه می‌شود.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section("دیتاست مورد استفاده")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">Diabetes Prediction Dataset</div>
        <p class="panel-text">
            منبع داده: Kaggle
        </p>
        <p class="panel-text">
            ویژگی‌ها شامل سن، جنسیت، فشار خون، بیماری قلبی،
            سابقه مصرف دخانیات، BMI، سطح HbA1c و قند خون هستند.
        </p>
        <p class="panel-text">
            متغیر هدف diabetes دارای دو کلاس صفر و یک است.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section("الگوریتم‌های استفاده‌شده")

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
    - **Accuracy:** درصد پیش‌بینی‌های صحیح.
    - **Error Rate:** درصد پیش‌بینی‌های اشتباه.
    - **Precision:** میزان صحت پیش‌بینی کلاس مثبت.
    - **Recall:** توانایی شناسایی نمونه‌های کلاس مثبت.
    - **F1-score:** میانگین هماهنگ Precision و Recall.
    - **Balanced Accuracy:** میانگین عملکرد مدل در دو کلاس.
    - **ROC-AUC:** توانایی تفکیک دو کلاس.
    - **PR-AUC:** ارزیابی عملکرد مدل با توجه به Precision و Recall.
    """)

    st.warning(
        "این سامانه برای اهداف پژوهشی و آموزشی طراحی شده است "
        "و جایگزین تشخیص یا مشاوره پزشکی نیست."
    )


# =====================================================
# فوتر
# =====================================================

st.markdown("""
<div class="footer">
    سامانه هوشمند تحلیل و غربالگری دیابت
    <br>
    Diabetes Machine Learning Research System
</div>
""", unsafe_allow_html=True)
