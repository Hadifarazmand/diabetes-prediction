
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
    confusion_matrix,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)


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

MODEL_NAMES = [
    "رگرسیون لجستیک",
    "درخت تصمیم",
    "جنگل تصادفی",
    "درخت‌های بسیار تصادفی",
    "گرادیان تقویتی"
]


# =========================================================
# طراحی ظاهری و واکنش‌گرایی
# =========================================================

st.markdown("""
<style>

:root {
    --navy: #173B57;
    --navy-dark: #102D43;
    --teal: #087F78;
    --teal-dark: #065F59;
    --page: #F3F6F9;
    --white: #FFFFFF;
    --text: #263746;
    --muted: #596B7B;
    --border: #D9E3EA;
}

/* پس‌زمینه کلی */

html, body, .stApp {
    background: #F3F6F9 !important;
    color: #263746 !important;
    font-family: Tahoma, Arial, sans-serif !important;
}

.block-container {
    max-width: 1250px !important;
    padding: 1.4rem 1.35rem 3rem !important;
}

/* راست‌چین کردن متن‌ها */

[data-testid="stMarkdownContainer"] {
    direction: rtl;
    text-align: right;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #263746 !important;
    line-height: 1.95 !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #173B57 !important;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #173B57 !important;
    direction: rtl;
    text-align: right;
    font-weight: 700 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #596B7B !important;
}

/* هدر اصلی */

.hero {
    background: linear-gradient(
        120deg,
        #173B57,
        #205D75,
        #087F78
    );
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 22px;
    direction: rtl;
    text-align: right;
    box-shadow: 0 8px 24px rgba(23,59,87,.13);
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
    font-size: 25px;
    font-weight: 900;
    line-height: 1.7;
    margin: 0 0 8px;
}

.hero p {
    color: #F0F8FA !important;
    font-size: 14px;
    line-height: 2;
    margin: 0;
}

/* عنوان بخش‌ها */

.section-title {
    color: #173B57 !important;
    font-size: 19px;
    font-weight: 900;
    margin: 24px 0 13px;
    text-align: right;
    direction: rtl;
}

/* کارت‌ها */

.panel {
    background: #FFFFFF !important;
    border: 1px solid #D9E3EA;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 3px 12px rgba(23,59,87,.04);
    direction: rtl;
    text-align: right;
}

.panel-title {
    color: #173B57 !important;
    font-size: 16px;
    font-weight: 900;
    margin-bottom: 7px;
}

.panel-text {
    color: #526577 !important;
    font-size: 13px;
    line-height: 2;
}

/* کارت‌های آماری */

.stat-card {
    background: #FFFFFF;
    border: 1px solid #D9E3EA;
    border-radius: 15px;
    padding: 15px;
    min-height: 105px;
    direction: rtl;
    text-align: right;
}

.stat-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: #E5F3F2;
    color: #087F78 !important;
    border-radius: 11px;
    font-size: 19px;
    margin-bottom: 7px;
}

.stat-label {
    color: #596B7B !important;
    font-size: 12px;
}

.stat-value {
    color: #173B57 !important;
    font-size: 21px;
    font-weight: 900;
    margin-top: 5px;
}

/* کارت نتیجه */

.result-card {
    border-radius: 17px;
    padding: 23px;
    margin: 17px 0;
    text-align: center;
    direction: rtl;
}

.result-card h2 {
    font-size: 23px;
    font-weight: 900;
    margin: 8px 0;
}

.result-card p {
    font-size: 14px;
    line-height: 2;
    margin: 0;
}

.result-icon {
    font-size: 38px;
    display: block;
}

/* نتیجه منفی */

.result-negative {
    background: #EAF7EF !important;
    border: 2px solid #A9D8BA;
}

.result-negative h2,
.result-negative p {
    color: #145C36 !important;
}

/* نتیجه مثبت */

.result-positive {
    background: #FCEBEC !important;
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
    font-family: Tahoma, Arial, sans-serif !important;
    font-weight: 800 !important;
}

.stButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover {
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

/* نوار کناری */

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

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #D0DFE8 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #263746 !important;
}

/* پیام‌ها */

[data-testid="stAlert"] {
    border-radius: 12px !important;
}

[data-testid="stAlert"] p {
    color: #263746 !important;
}

/* جدول */

[data-testid="stDataFrame"] {
    border: 1px solid #D9E3EA;
    border-radius: 10px;
}

/* بازشونده */

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #D9E3EA;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #173B57 !important;
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

/* تنظیمات موبایل */

@media (max-width: 768px) {

    .block-container {
        padding: .8rem .65rem 2rem !important;
    }

    .hero {
        padding: 20px 16px;
        border-radius: 16px;
    }

    .hero h1 {
        font-size: 20px;
    }

    .hero p {
        font-size: 13px;
    }

    .section-title {
        font-size: 17px;
    }

    .panel {
        padding: 14px;
    }

    .stat-card {
        padding: 11px;
        min-height: 96px;
    }

    .stat-value {
        font-size: 18px;
    }

    .result-card {
        padding: 18px 12px;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.7rem !important;
    }

    [data-testid="stWidgetLabel"] p {
        font-size: 13px !important;
    }

    .stButton > button {
        width: 100%;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# توابع رابط کاربری
# =========================================================

def header():

    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🩺</div>

        <h1>
            سامانه هوشمند تحلیل و غربالگری دیابت
        </h1>

        <p>
            تحلیل اطلاعات سلامت با استفاده از الگوریتم‌های یادگیری ماشین
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


def show_result(prediction, probability=None):

    if int(prediction) == 0:

        st.markdown("""
        <div class="result-card result-negative">

            <span class="result-icon">✓</span>

            <h2>دیابت پیش‌بینی نشد</h2>

            <p>
                بر اساس اطلاعات واردشده، مدل فرد را
                در گروه بدون دیابت قرار داده است.
            </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card result-positive">

            <span class="result-icon">!</span>

            <h2>احتمال دیابت وجود دارد</h2>

            <p>
                مدل فرد را در گروه دارای دیابت قرار داده است.
                بررسی آزمایشگاهی و مراجعه به پزشک توصیه می‌شود.
            </p>

        </div>
        """, unsafe_allow_html=True)

    if probability is not None:

        st.metric(
            "احتمال پیش‌بینی‌شده توسط مدل برای گروه دیابت",
            f"{probability:.1f}%"
        )


# =========================================================
# بارگذاری و اعتبارسنجی دیتاست
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            "فایل دیتاست پیدا نشد. مسیر صحیح باید این باشد: "
            "dataset/diabetes_prediction_dataset.csv"
        )

    df = pd.read_csv(DATA_FILE)

    df.columns = df.columns.astype(str).str.strip()

    missing = [
        col for col in ALL_FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "ستون‌های زیر در فایل دیتاست پیدا نشدند: "
            + ", ".join(missing)
        )

    df = df[ALL_FEATURES + [TARGET]].copy()

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(subset=[TARGET])

    df[TARGET] = df[TARGET].astype(int)

    df = df[df[TARGET].isin([0, 1])]

    if df.empty or df[TARGET].nunique() != 2:

        raise ValueError(
            "ستون diabetes باید هر دو برچسب 0 و 1 را داشته باشد."
        )

    numeric_columns = [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=["age"])

    if df[TARGET].value_counts().min() < 2:

        raise ValueError(
            "تعداد نمونه‌های یکی از کلاس‌ها برای آموزش کافی نیست."
        )

    return df


# =========================================================
# تعریف الگوریتم‌ها
# =========================================================

def create_model(model_name):

    if model_name == "رگرسیون لجستیک":

        return LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        )

    if model_name == "درخت تصمیم":

        return DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )

    if model_name == "جنگل تصادفی":

        return RandomForestClassifier(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        )

    if model_name == "درخت‌های بسیار تصادفی":

        return ExtraTreesClassifier(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

    if model_name == "گرادیان تقویتی":

        return HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=20,
            random_state=42
        )

    raise ValueError("الگوریتم انتخاب‌شده معتبر نیست.")


# =========================================================
# ساخت Pipeline
# =========================================================

def create_pipeline(model_name, features):

    categorical = [
        col for col in features
        if col in ["gender", "smoking_history"]
    ]

    numeric = [
        col for col in features
        if col not in categorical
    ]

    numeric_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipe,
            numeric
        ),
        (
            "categorical",
            categorical_pipe,
            categorical
        )
    ])

    return Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            create_model(model_name)
        )
    ])


# =========================================================
# تقسیم داده‌ها
# =========================================================

def split_data(df, features):

    X = df[features].copy()

    y = df[TARGET].copy()

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# =========================================================
# ارزیابی الگوریتم
# =========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    balanced = balanced_accuracy_score(
        y_test,
        predictions
    )

    auc = np.nan

    if hasattr(model, "predict_proba"):

        try:

            auc = roc_auc_score(
                y_test,
                model.predict_proba(X_test)[:, 1]
            )

        except Exception:

            pass

    return {
        "درصد درستی": accuracy * 100,
        "درصد خطا": (1 - accuracy) * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "Balanced Accuracy": balanced * 100,
        "ROC-AUC": auc * 100 if not np.isnan(auc) else np.nan
    }


# =========================================================
# خلاصه دیتاست
# =========================================================

def get_dataset_summary(df):

    return {
        "rows": len(df),
        "positive": int((df[TARGET] == 1).sum()),
        "negative": int((df[TARGET] == 0).sum())
    }


# =========================================================
# مدیریت وضعیت نشست
# =========================================================

defaults = {
    "trained_models": {},
    "results": {},
    "patient_result": None,
    "patient_model_name": None,
    "patient_mode": None,
    "patient_probability": None,
    "data_signature": None
}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# منوی کناری
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="
        text-align:center;
        padding:15px 5px 18px;
        direction:rtl;
    ">

        <div style="
            width:66px;
            height:66px;
            margin:auto;
            display:flex;
            align-items:center;
            justify-content:center;
            background:#FFFFFF;
            color:#087F78;
            border-radius:18px;
            font-size:31px;
        ">
            🩺
        </div>

        <h3 style="
            color:#FFFFFF !important;
            margin-top:12px;
        ">
            پنل هوشمند دیابت
        </h3>

        <p style="
            color:#D0DFE8 !important;
            font-size:12px;
        ">
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

    st.caption(
        "Diabetes Prediction Research System"
    )


# =========================================================
# نمایش هدر
# =========================================================

header()


# =========================================================
# آماده‌سازی دیتاست
# =========================================================

try:

    df = load_data()

except Exception as exc:

    st.error(str(exc))

    st.stop()


summary = get_dataset_summary(df)

signature = (
    len(df),
    summary["positive"],
    summary["negative"]
)

if st.session_state.data_signature != signature:

    st.session_state.trained_models = {}

    st.session_state.results = {}

    st.session_state.patient_result = None

    st.session_state.patient_probability = None

    st.session_state.data_signature = signature


# =========================================================
# بخش اول: تحلیل اطلاعات بیمار
# =========================================================

if page == "تحلیل اطلاعات بیمار":

    section(
        "تحلیل و بررسی اطلاعات بیمار"
    )

    st.markdown("""
    <div class="panel">

        <div class="panel-title">
            👤 فرم اطلاعات سلامت
        </div>

        <div class="panel-text">
            ابتدا نوع تحلیل و الگوریتم را انتخاب کنید.
            سپس اطلاعات فرد را وارد کنید تا مدل بر اساس
            داده‌های دیتاست، یک پیش‌بینی پژوهشی ارائه دهد.
        </div>

    </div>
    """, unsafe_allow_html=True)

    # انتخاب نوع تحلیل

    mode = st.radio(
        "نوع تحلیل",
        [
            "غربالگری اولیه (بدون آزمایش)",
            "تحلیل کامل (با اطلاعات آزمایشگاهی)"
        ],
        horizontal=False,
        key="patient_mode_select"
    )

    features = (
        SCREENING_FEATURES
        if mode.startswith("غربالگری")
        else ALL_FEATURES
    )

    # انتخاب الگوریتم

    selected_algorithm = st.selectbox(
        "🤖 انتخاب الگوریتم یادگیری ماشین",
        MODEL_NAMES,
        key="patient_algorithm"
    )

    section(
        "مشخصات فردی و سوابق سلامت"
    )

    patient = {}

    c1, c2 = st.columns(2)

    with c1:

        patient["gender"] = st.selectbox(
            "جنسیت",
            ["Female", "Male", "Other"],
            format_func=lambda x: {
                "Female": "زن",
                "Male": "مرد",
                "Other": "سایر"
            }[x]
        )

        patient["age"] = st.number_input(
            "سن",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            step=1.0
        )

        patient["hypertension"] = st.selectbox(
            "آیا فشار خون بالا دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x == 1 else "خیر"
        )

        patient["heart_disease"] = st.selectbox(
            "آیا سابقه بیماری قلبی دارید؟",
            [0, 1],
            format_func=lambda x: "بله" if x == 1 else "خیر"
        )

    with c2:

        patient["smoking_history"] = st.selectbox(
            "سابقه مصرف دخانیات",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current"
            ],
            format_func=lambda x: {
                "never": "هرگز",
                "No Info": "اطلاعاتی ندارم",
                "current": "در حال حاضر",
                "former": "در گذشته",
                "ever": "در طول عمر",
                "not current": "در حال حاضر مصرف نمی‌کنم"
            }[x]
        )

        patient["bmi"] = st.number_input(
            "شاخص توده بدنی (BMI)",
            min_value=10.0,
            max_value=80.0,
            value=25.0,
            step=0.1
        )

    # اطلاعات آزمایشگاهی

    if mode.startswith("تحلیل کامل"):

        section(
            "اطلاعات آزمایشگاهی"
        )

        a, b = st.columns(2)

        with a:

            patient["HbA1c_level"] = st.number_input(
                "مقدار HbA1c",
                min_value=3.0,
                max_value=20.0,
                value=5.5,
                step=0.1,
                help="مقدار آزمایش هموگلوبین A1c"
            )

        with b:

            patient["blood_glucose_level"] = st.number_input(
                "قند خون (mg/dL)",
                min_value=50,
                max_value=500,
                value=100,
                step=1
            )

    else:

        st.info(
            "در حالت غربالگری اولیه، اطلاعات HbA1c و قند خون "
            "وارد مدل نمی‌شوند؛ مدل فقط با شش ویژگی اولیه آموزش "
            "داده و اجرا می‌شود."
        )

    st.markdown("---")

    # اجرای پیش‌بینی

    if st.button(
        "🔍 تحلیل اطلاعات بیمار",
        use_container_width=True
    ):

        try:

            X_train, X_test, y_train, y_test = split_data(
                df,
                features
            )

            cache_key = f"{mode}_{selected_algorithm}"

            model = st.session_state.trained_models.get(
                cache_key
            )

            with st.spinner(
                "در حال آموزش الگوریتم و تحلیل اطلاعات..."
            ):

                if model is None:

                    model = create_pipeline(
                        selected_algorithm,
                        features
                    )

                    model.fit(
                        X_train,
                        y_train
                    )

                    st.session_state.trained_models[
                        cache_key
                    ] = model

                input_df = pd.DataFrame(
                    [
                        {
                            col: patient[col]
                            for col in features
                        }
                    ],
                    columns=features
                )

                prediction = int(
                    model.predict(input_df)[0]
                )

                probability = None

                if hasattr(model, "predict_proba"):

                    probability = float(
                        model.predict_proba(input_df)[0][1] * 100
                    )

            st.session_state.patient_result = prediction

            st.session_state.patient_probability = probability

            st.session_state.patient_model_name = selected_algorithm

            st.session_state.patient_mode = mode

        except Exception as exc:

            st.error(
                f"خطا در تحلیل اطلاعات: {exc}"
            )

    # نمایش نتیجه

    if st.session_state.patient_result is not None:

        section(
            "نتیجه تحلیل بیمار"
        )

        show_result(
            st.session_state.patient_result,
            st.session_state.patient_probability
        )

        st.markdown(f"""
        <div class="panel">

            <div class="panel-title">
                جزئیات تحلیل
            </div>

            <div class="panel-text">
                الگوریتم:
                <b>{st.session_state.patient_model_name}</b>
            </div>

            <div class="panel-text">
                نوع تحلیل:
                <b>{st.session_state.patient_mode}</b>
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "این خروجی صرفاً پیش‌بینی یک مدل یادگیری ماشین "
            "بر اساس دیتاست است؛ تشخیص پزشکی نیست و جایگزین "
            "آزمایش قند خون، HbA1c یا نظر پزشک نمی‌شود."
        )


# =========================================================
# بخش دوم: تحلیل و مقایسه الگوریتم‌ها
# =========================================================

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    section(
        "آموزش و مقایسه عملکرد الگوریتم‌ها"
    )

    st.markdown("""
    <div class="panel">

        <div class="panel-title">
            📊 ارزیابی الگوریتم‌ها
        </div>

        <div class="panel-text">
            الگوریتم‌ها روی داده‌های آموزشی و آزمون یکسان
            ارزیابی می‌شوند. نتایج شامل درصد درستی، درصد خطا،
            Precision، Recall، F1 و ROC-AUC است.
        </div>

    </div>
    """, unsafe_allow_html=True)

    comparison_mode = st.radio(
        "مجموعه ویژگی مورد ارزیابی",
        [
            "غربالگری اولیه (بدون آزمایش)",
            "تحلیل کامل (با اطلاعات آزمایشگاهی)"
        ],
        key="comparison_mode"
    )

    features = (
        SCREENING_FEATURES
        if comparison_mode.startswith("غربالگری")
        else ALL_FEATURES
    )

    mode_key = comparison_mode

    # آموزش و ارزیابی همه مدل‌ها

    if st.button(
        "🚀 آموزش و ارزیابی همه الگوریتم‌ها",
        use_container_width=True
    ):

        try:

            X_train, X_test, y_train, y_test = split_data(
                df,
                features
            )

            rows = []

            fitted = {}

            progress = st.progress(0)

            status = st.empty()

            for idx, model_name in enumerate(MODEL_NAMES):

                status.info(
                    f"در حال آموزش {model_name} ..."
                )

                start = time.time()

                model = create_pipeline(
                    model_name,
                    features
                )

                model.fit(
                    X_train,
                    y_train
                )

                elapsed = time.time() - start

                metrics = evaluate_model(
                    model,
                    X_test,
                    y_test
                )

                rows.append({
                    "الگوریتم": model_name,
                    **metrics,
                    "زمان آموزش": elapsed
                })

                fitted[model_name] = model

                progress.progress(
                    (idx + 1) / len(MODEL_NAMES)
                )

            result_df = pd.DataFrame(rows).sort_values(
                "F1",
                ascending=False
            ).reset_index(drop=True)

            st.session_state.results[mode_key] = {
                "table": result_df,
                "models": fitted,
                "X_test": X_test,
                "y_test": y_test
            }

            status.success(
                "آموزش و ارزیابی الگوریتم‌ها با موفقیت انجام شد."
            )

        except Exception as exc:

            st.error(
                f"خطا در آموزش الگوریتم‌ها: {exc}"
            )

    # بازیابی نتایج

    result_pack = st.session_state.results.get(
        mode_key
    )

    if result_pack is None:

        st.info(
            "برای مشاهده نتایج، ابتدا دکمه آموزش و ارزیابی "
            "همه الگوریتم‌ها را بزنید."
        )

    else:

        results = result_pack["table"]

        section(
            "انتخاب الگوریتم برای مشاهده جزئیات"
        )

        selected = st.selectbox(
            "الگوریتم موردنظر",
            results["الگوریتم"].tolist(),
            key=f"details_{mode_key}"
        )

        row = results[
            results["الگوریتم"] == selected
        ].iloc[0]

        # کارت‌های آماری

        m1, m2, m3 = st.columns(3)

        with m1:

            stat(
                "✓",
                "درصد درستی",
                f"{row['درصد درستی']:.2f}%"
            )

        with m2:

            stat(
                "×",
                "درصد خطا",
                f"{row['درصد خطا']:.2f}%"
            )

        with m3:

            stat(
                "★",
                "امتیاز F1",
                f"{row['F1']:.2f}%"
            )

        # معیارهای تفصیلی

        section(
            "معیارهای ارزیابی الگوریتم انتخاب‌شده"
        )

        detail_rows = [
            (
                "Precision",
                f"{row['Precision']:.2f}%"
            ),
            (
                "Recall",
                f"{row['Recall']:.2f}%"
            ),
            (
                "F1",
                f"{row['F1']:.2f}%"
            ),
            (
                "Balanced Accuracy",
                f"{row['Balanced Accuracy']:.2f}%"
            ),
            (
                "ROC-AUC",
                "—" if pd.isna(row["ROC-AUC"])
                else f"{row['ROC-AUC']:.2f}%"
            ),
            (
                "زمان آموزش",
                f"{row['زمان آموزش']:.2f} ثانیه"
            )
        ]

        st.dataframe(
            pd.DataFrame(
                detail_rows,
                columns=["معیار", "مقدار"]
            ),
            use_container_width=True,
            hide_index=True
        )

        # جدول مقایسه

        section(
            "جدول مقایسه تمام الگوریتم‌ها"
        )

        display = results.copy()

        percentage_columns = [
            "درصد درستی",
            "درصد خطا",
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy",
            "ROC-AUC"
        ]

        for col in percentage_columns:

            display[col] = display[col].apply(
                lambda value:
                    "—" if pd.isna(value)
                    else f"{value:.2f}%"
            )

        display["زمان آموزش"] = display[
            "زمان آموزش"
        ].apply(
            lambda value: f"{value:.2f} ثانیه"
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        # نمودار مقایسه

        section(
            "نمودار مقایسه الگوریتم‌ها"
        )

        chart_data = results.set_index(
            "الگوریتم"
        )[
            ["درصد درستی", "درصد خطا"]
        ]

        st.bar_chart(
            chart_data,
            use_container_width=True
        )

        # ماتریس درهم‌ریختگی

        section(
            "ماتریس درهم‌ریختگی"
        )

        selected_model = result_pack[
            "models"
        ][selected]

        predictions = selected_model.predict(
            result_pack["X_test"]
        )

        cm = confusion_matrix(
            result_pack["y_test"],
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(
            figsize=(6, 4.5)
        )

        fig.patch.set_facecolor("#FFFFFF")

        ax.set_facecolor("#FFFFFF")

        im = ax.imshow(
            cm,
            cmap="GnBu"
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

        threshold = cm.max() / 2 if cm.max() else 0

        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color=(
                        "#102D43"
                        if cm[i, j] < threshold
                        else "#FFFFFF"
                    ),
                    fontweight="bold"
                )

        fig.colorbar(
            im,
            ax=ax
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.info(
            "معیارها از مجموعه آزمون جداشده محاسبه شده‌اند. "
            "این نتایج عملکرد مدل روی داده‌های آزمون را نشان می‌دهند "
            "و تضمین‌کننده عملکرد مشابه برای هر بیمار جدید نیستند."
        )


# =========================================================
# بخش سوم: درباره پروژه
# =========================================================

elif page == "درباره پروژه":

    section(
        "معرفی سامانه"
    )

    st.markdown("""
    <div class="panel">

        <div class="panel-title">
            سامانه هوشمند تحلیل و غربالگری دیابت
        </div>

        <div class="panel-text">
            این سامانه با استفاده از الگوریتم‌های یادگیری ماشین
            و دیتاست عمومی پیش‌بینی دیابت، اطلاعات سلامت افراد
            را تحلیل می‌کند. کاربر می‌تواند تحلیل اولیه بدون
            اطلاعات آزمایشگاهی یا تحلیل کامل با مقادیر HbA1c
            و قند خون را انتخاب کند.
        </div>

    </div>
    """, unsafe_allow_html=True)

    # اطلاعات دیتاست

    section(
        "اطلاعات دیتاست"
    )

    a, b, c = st.columns(3)

    with a:

        stat(
            "▦",
            "تعداد رکوردها",
            f"{summary['rows']:,}"
        )

    with b:

        stat(
            "●",
            "نمونه‌های دارای دیابت",
            f"{summary['positive']:,}"
        )

    with c:

        stat(
            "○",
            "نمونه‌های بدون دیابت",
            f"{summary['negative']:,}"
        )

    # الگوریتم‌ها

    section(
        "الگوریتم‌های استفاده‌شده"
    )

    for name in MODEL_NAMES:

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">
                    🧠 {name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # معیارها

    section(
        "معیارهای ارزیابی"
    )

    st.markdown("""
    - **Accuracy:** نسبت پیش‌بینی‌های درست به کل پیش‌بینی‌ها.
    - **Error Rate:** نسبت پیش‌بینی‌های نادرست به کل پیش‌بینی‌ها.
    - **Precision:** دقت پیش‌بینی نمونه‌های مثبت.
    - **Recall:** توانایی شناسایی نمونه‌های مثبت.
    - **F1-score:** میانگین هماهنگ Precision و Recall.
    - **Balanced Accuracy:** میانگین حساسیت در کلاس‌ها.
    - **ROC-AUC:** توانایی تفکیک دو کلاس در آستانه‌های مختلف.
    """)

    st.warning(
        "دیتاست انتخاب‌شده دودسته‌ای است؛ بنابراین این سامانه "
        "فقط دو خروجی «دیابت پیش‌بینی نشد» و "
        "«احتمال دیابت وجود دارد» ارائه می‌کند. "
        "این سامانه ابزار پژوهشی است و جایگزین تشخیص پزشکی نیست."
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
