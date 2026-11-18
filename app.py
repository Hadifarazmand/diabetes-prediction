
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import time
import textwrap

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
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)


# =====================================================
# تنظیمات اصلی
# =====================================================

st.set_page_config(
    page_title="سامانه هوشمند غربالگری دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "cdc_diabetes_health_indicators.csv"

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


# =====================================================
# طراحی ظاهری
# =====================================================

st.markdown(
    textwrap.dedent("""
    <style>
    :root {
        --navy: #173B57;
        --navy-dark: #102D43;
        --teal: #087F78;
        --teal-dark: #06645F;
        --page: #F2F6FA;
        --white: #FFFFFF;
        --text: #263746;
        --muted: #607386;
        --border: #DCE5ED;
    }

    .stApp {
        background: var(--page);
        color: var(--text);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    html, body, [class*="st-"] {
        font-family: Tahoma, Arial, sans-serif;
    }

    h1, h2, h3, h4 {
        color: var(--navy) !important;
        font-weight: 800 !important;
    }

    p, li, label {
        color: var(--text);
        line-height: 1.9;
    }

    [data-testid="stMarkdownContainer"] p {
        color: var(--text);
        line-height: 1.9;
    }

    [data-testid="stWidgetLabel"] p {
        color: var(--navy) !important;
        font-weight: 700 !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    /* هدر اصلی */
    .hero {
        background: linear-gradient(125deg, #173B57 0%, #205D75 55%, #087F78 100%);
        padding: 30px;
        border-radius: 22px;
        margin-bottom: 25px;
        direction: rtl;
        text-align: right;
        box-shadow: 0 10px 28px rgba(23,59,87,0.13);
    }

    .hero-icon {
        background: white;
        color: #087F78;
        width: 58px;
        height: 58px;
        border-radius: 17px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        margin-bottom: 18px;
    }

    .hero h1 {
        color: white !important;
        font-size: 25px;
        margin: 0 0 12px 0;
    }

    .hero p {
        color: #EAF5F7 !important;
        font-size: 14px;
        margin: 0;
    }

    /* عنوان بخش‌ها */
    .section-heading {
        color: #173B57;
        font-size: 19px;
        font-weight: 900;
        margin: 28px 0 15px;
        text-align: right;
        direction: rtl;
    }

    /* کارت اطلاعات */
    .info-card {
        background: white;
        border: 1px solid #DCE5ED;
        border-radius: 17px;
        padding: 20px;
        margin-bottom: 18px;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 14px rgba(23,59,87,0.035);
    }

    .info-card h4 {
        color: #173B57 !important;
        margin: 0 0 8px 0;
        font-size: 16px;
    }

    .info-card p {
        color: #526577 !important;
        font-size: 13px;
        margin: 0;
    }

    /* کارت آمار */
    .metric-card {
        background: white;
        border: 1px solid #DCE5ED;
        border-radius: 15px;
        padding: 17px;
        min-height: 115px;
        direction: rtl;
        text-align: right;
    }

    .metric-label {
        color: #607386 !important;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #173B57 !important;
        font-size: 23px;
        font-weight: 900;
    }

    /* نتیجه بیمار */
    .result-card {
        padding: 25px;
        border-radius: 18px;
        margin: 20px 0;
        text-align: center;
        direction: rtl;
    }

    .result-card h2 {
        font-size: 23px;
        margin: 8px 0;
    }

    .result-card p {
        font-size: 14px;
        line-height: 2;
    }

    .result-negative {
        background: #E8F6ED;
        border: 2px solid #A8D8B8;
    }

    .result-negative h2,
    .result-negative p {
        color: #145C36 !important;
    }

    .result-borderline {
        background: #FFF5D9;
        border: 2px solid #E8D18A;
    }

    .result-borderline h2,
    .result-borderline p {
        color: #704A0B !important;
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
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        min-height: 46px;
        font-weight: 800 !important;
        transition: 0.2s;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #06645F !important;
        color: white !important;
        border: none !important;
    }

    .stButton > button p {
        color: white !important;
    }

    /* ورودی‌ها */
    [data-baseweb="input"] {
        background: white !important;
        border-radius: 10px !important;
    }

    [data-baseweb="input"] input {
        color: #263746 !important;
        -webkit-text-fill-color: #263746 !important;
    }

    [data-baseweb="select"] > div {
        background: white !important;
        border-color: #CBD8E2 !important;
        border-radius: 10px !important;
    }

    [data-baseweb="select"] span {
        color: #263746 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    [role="option"] {
        background: white !important;
        color: #263746 !important;
    }

    [role="option"]:hover {
        background: #E7F5F3 !important;
    }

    /* منوی کناری */
    [data-testid="stSidebar"] {
        background: #173B57 !important;
    }

    [data-testid="stSidebar"] > div {
        background: #173B57 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 12px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 9px;
    }

    /* جدول */
    [data-testid="stDataFrame"] {
        border: 1px solid #DCE5ED;
        border-radius: 12px;
        overflow: hidden;
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

    /* هشدارها */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* فوتر */
    .footer {
        border-top: 1px solid #DCE5ED;
        margin-top: 35px;
        padding-top: 18px;
        text-align: center;
        color: #607386 !important;
        font-size: 12px;
    }

    /* نمایش موبایل */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-top: 1rem !important;
        }

        .hero {
            padding: 22px;
            border-radius: 17px;
        }

        .hero h1 {
            font-size: 20px;
            line-height: 1.8;
        }

        .section-heading {
            font-size: 17px;
        }

        .metric-card {
            padding: 13px;
            min-height: 100px;
        }

        .metric-value {
            font-size: 19px;
        }

        .info-card {
            padding: 16px;
        }
    }
    </style>
    """),
    unsafe_allow_html=True
)


# =====================================================
# توابع رابط کاربری
# =====================================================

def header():
    st.markdown(
        textwrap.dedent("""
        <div class="hero">
            <div class="hero-icon">🩺</div>
            <h1>سامانه هوشمند تحلیل و غربالگری دیابت</h1>
            <p>
                بررسی اطلاعات سلامت با استفاده از الگوریتم‌های یادگیری ماشین
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def info_card(title, description):
    st.markdown(
        f"""
        <div class="info-card">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_result(prediction):
    if prediction == 0:
        css_class = "result-negative"
        icon = "✓"
        description = "مدل، اطلاعات واردشده را در گروه بدون دیابت دسته‌بندی کرده است."

    elif prediction == 1:
        css_class = "result-borderline"
        icon = "⚠"
        description = "مدل، اطلاعات واردشده را در گروه پیش‌دیابت دسته‌بندی کرده است."

    else:
        css_class = "result-positive"
        icon = "!"
        description = "مدل، اطلاعات واردشده را در گروه دیابت دسته‌بندی کرده است."

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div style="font-size:38px;font-weight:900;">{icon}</div>
            <h2>{CLASS_NAMES[prediction]}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# بارگذاری دیتاست
# =====================================================

@st.cache_data(show_spinner=False)
def load_data():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        raise FileNotFoundError(
            "فایل دیتاست پیدا نشد. فایل "
            "cdc_diabetes_health_indicators.csv را داخل پوشه dataset قرار دهید."
        )

    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            "ستون Diabetes_012 در دیتاست وجود ندارد. "
            "این برنامه برای خروجی سه‌کلاسه به این ستون نیاز دارد."
        )

    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        raise ValueError(
            "ستون‌های زیر در دیتاست وجود ندارند: " + ", ".join(missing)
        )

    df = df[FEATURES + [TARGET]].copy()

    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1, 2])]

    if df.empty:
        raise ValueError("داده معتبر برای آموزش مدل پیدا نشد.")

    if set(df[TARGET].unique()) != {0, 1, 2}:
        raise ValueError(
            "دیتاست باید هر سه کلاس ۰، ۱ و ۲ را داشته باشد."
        )

    return df


# =====================================================
# الگوریتم‌ها
# =====================================================

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


def get_data_split(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# =====================================================
# آموزش و ارزیابی
# =====================================================

def train_one_model(model_name, X_train, y_train):
    models = create_models()

    if model_name not in models:
        raise ValueError("الگوریتم انتخاب‌شده معتبر نیست.")

    pipeline = create_pipeline(models[model_name])
    pipeline.fit(X_train, y_train)

    return pipeline


def train_and_evaluate_all(X, y):
    X_train, X_test, y_train, y_test = get_data_split(X, y)

    models = create_models()
    trained_models = {}
    results = []

    progress = st.progress(0)
    status = st.empty()

    for index, (name, model) in enumerate(models.items()):
        status.info(f"در حال آموزش {name} ...")

        start_time = time.time()

        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)

        elapsed = time.time() - start_time

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "الگوریتم": name,
            "درصد درستی": accuracy * 100,
            "درصد خطا": (1 - accuracy) * 100,
            "Precision": precision_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ) * 100,
            "Recall": recall_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ) * 100,
            "F1": f1_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ) * 100,
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, predictions
            ) * 100,
            "زمان آموزش": elapsed
        })

        trained_models[name] = pipeline
        progress.progress((index + 1) / len(models))

    status.success("آموزش و ارزیابی الگوریتم‌ها کامل شد.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    return trained_models, results_df, X_test, y_test


# =====================================================
# حافظه برنامه
# =====================================================

defaults = {
    "trained_models": {},
    "results_df": None,
    "X_test": None,
    "y_test": None,
    "patient_result": None,
    "patient_model_name": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# منوی کناری
# =====================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:12px 5px 20px;">
            <div style="
                background:white;
                color:#087F78;
                width:58px;
                height:58px;
                border-radius:16px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:28px;
                margin:auto;">
                🩺
            </div>
            <h3 style="color:white !important;margin-top:12px;">
                پنل تحلیل دیابت
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "بخش موردنظر",
        [
            "تحلیل اطلاعات بیمار",
            "تحلیل و مقایسه الگوریتم‌ها",
            "درباره پروژه"
        ],
        label_visibility="visible"
    )


# =====================================================
# هدر
# =====================================================

header()


# =====================================================
# دریافت داده
# =====================================================

try:
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

except Exception as error:
    st.error(str(error))
    st.stop()


# =====================================================
# صفحه اول: تحلیل بیمار
# =====================================================

if page == "تحلیل اطلاعات بیمار":

    section("تحلیل و بررسی اطلاعات بیمار")

    info_card(
        "فرم اطلاعات سلامت",
        "ابتدا الگوریتم موردنظر را انتخاب کنید؛ سپس اطلاعات سلامت فرد را وارد کرده و نتیجه پیش‌بینی را مشاهده کنید."
    )

    algorithm_names = list(create_models().keys())

    selected_algorithm = st.selectbox(
        "انتخاب الگوریتم یادگیری ماشین",
        algorithm_names,
        key="patient_algorithm"
    )

    section("مشخصات فردی")

    col1, col2 = st.columns(2)

    patient = {}

    with col1:
        patient["Sex"] = st.selectbox(
            "جنسیت",
            [0, 1],
            format_func=lambda x: "زن" if x == 0 else "مرد"
        )

        patient["Age"] = st.selectbox(
            "سن",
            list(range(1, 14)),
            format_func=lambda x: f"گروه سنی {x}"
        )

        patient["BMI"] = st.number_input(
            "شاخص توده بدنی (BMI)",
            min_value=10.0,
            max_value=80.0,
            value=25.0,
            step=0.5
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

        patient["GenHlth"] = st.selectbox(
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

        patient["MentHlth"] = st.number_input(
            "روزهای سلامت روان نامناسب در ماه",
            0, 30, 0
        )

        patient["PhysHlth"] = st.number_input(
            "روزهای سلامت جسمی نامناسب در ماه",
            0, 30, 0
        )

        patient["BMI"] = float(patient["BMI"])

    with col2:
        binary_questions = {
            "HighBP": "آیا فشار خون بالا دارید؟",
            "HighChol": "آیا کلسترول خون بالا دارید؟",
            "CholCheck": "آیا آزمایش کلسترول داده‌اید؟",
            "Smoker": "آیا سابقه مصرف حداقل ۱۰۰ نخ سیگار دارید؟",
            "Stroke": "آیا سابقه سکته مغزی دارید؟",
            "HeartDiseaseorAttack": "آیا سابقه بیماری قلبی دارید؟",
            "PhysActivity": "آیا فعالیت بدنی منظم دارید؟",
            "Fruits": "آیا روزانه میوه مصرف می‌کنید؟",
            "Veggies": "آیا روزانه سبزیجات مصرف می‌کنید؟",
            "HvyAlcoholConsump": "آیا مصرف زیاد نوشیدنی الکلی دارید؟",
            "AnyHealthcare": "آیا به خدمات درمانی دسترسی دارید؟",
            "NoDocbcCost": "آیا به دلیل هزینه از مراجعه به پزشک خودداری کرده‌اید؟",
            "DiffWalk": "آیا در راه رفتن مشکل دارید؟"
        }

        for key, question in binary_questions.items():
            patient[key] = st.selectbox(
                question,
                [0, 1],
                format_func=lambda x: "بله" if x == 1 else "خیر",
                key=f"patient_{key}"
            )

    st.markdown("---")

    if st.button(
        "🔍 تحلیل اطلاعات بیمار",
        use_container_width=True
    ):
        try:
            with st.spinner("در حال تحلیل اطلاعات..."):
                X_train, X_test, y_train, y_test = get_data_split(X, y)

                model = train_one_model(
                    selected_algorithm,
                    X_train,
                    y_train
                )

                input_df = pd.DataFrame(
                    [patient],
                    columns=FEATURES
                )

                prediction = int(model.predict(input_df)[0])

                st.session_state.patient_result = prediction
                st.session_state.patient_model_name = selected_algorithm

        except Exception as error:
            st.error(f"خطا در تحلیل اطلاعات: {error}")

    if st.session_state.patient_result is not None:
        section("نتیجه تحلیل")

        show_result(st.session_state.patient_result)

        info_card(
            "الگوریتم استفاده‌شده",
            st.session_state.patient_model_name
        )

        st.warning(
            "این نتیجه صرفاً خروجی یک مدل پژوهشی و غربالگری است و تشخیص پزشکی قطعی محسوب نمی‌شود. "
            "تشخیص دیابت باید با آزمایش‌های معتبر و نظر پزشک انجام شود."
        )


# =====================================================
# صفحه دوم: مقایسه الگوریتم‌ها
# =====================================================

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    section("آموزش و مقایسه عملکرد الگوریتم‌ها")

    info_card(
        "ارزیابی الگوریتم‌های یادگیری ماشین",
        "تمام الگوریتم‌ها با داده‌های آموزشی و آزمون یکسان ارزیابی می‌شوند. معیارهای عملکرد هر الگوریتم در ادامه نمایش داده می‌شود."
    )

    if st.button(
        "🚀 آموزش و ارزیابی همه الگوریتم‌ها",
        use_container_width=True
    ):
        try:
            trained, results, X_test, y_test = train_and_evaluate_all(X, y)

            st.session_state.trained_models = trained
            st.session_state.results_df = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.success("ارزیابی همه الگوریتم‌ها با موفقیت انجام شد.")

        except Exception as error:
            st.error(f"خطا در آموزش الگوریتم‌ها: {error}")

    if st.session_state.results_df is not None:

        results = st.session_state.results_df

        section("انتخاب الگوریتم برای مشاهده جزئیات")

        selected = st.selectbox(
            "الگوریتم موردنظر",
            results["الگوریتم"].tolist(),
            key="comparison_algorithm"
        )

        selected_row = results[
            results["الگوریتم"] == selected
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "درصد درستی",
                f"{selected_row['درصد درستی']:.2f}%"
            )

        with c2:
            metric_card(
                "درصد خطا",
                f"{selected_row['درصد خطا']:.2f}%"
            )

        with c3:
            metric_card(
                "امتیاز F1",
                f"{selected_row['F1']:.2f}%"
            )

        section("سایر معیارهای ارزیابی")

        detail = pd.DataFrame({
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
            detail,
            use_container_width=True,
            hide_index=True
        )

        section("جدول مقایسه الگوریتم‌ها")

        display_df = results.copy()

        percentage_columns = [
            "درصد درستی",
            "درصد خطا",
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy"
        ]

        for col in percentage_columns:
            display_df[col] = display_df[col].map(
                lambda value: f"{value:.2f}%"
            )

        display_df["زمان آموزش"] = display_df["زمان آموزش"].map(
            lambda value: f"{value:.2f} ثانیه"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        section("نمودار مقایسه عملکرد")

        chart_df = results.set_index("الگوریتم")[
            ["درصد درستی", "درصد خطا"]
        ]

        st.bar_chart(chart_df, use_container_width=True)

        section("ماتریس درهم‌ریختگی")

        model = st.session_state.trained_models[selected]

        predictions = model.predict(st.session_state.X_test)

        cm = confusion_matrix(
            st.session_state.y_test,
            predictions,
            labels=[0, 1, 2]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        image = ax.imshow(
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

        labels = [
            "No Diabetes",
            "Prediabetes",
            "Diabetes"
        ]

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])

        ax.set_xticklabels(labels, rotation=12, color="#263746")
        ax.set_yticklabels(labels, color="#263746")

        threshold = cm.max() / 2 if cm.max() else 0

        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="#102D43" if cm[i, j] < threshold else "#FFFFFF",
                    fontweight="bold"
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "معیارهای نمایش‌داده‌شده از مجموعه آزمون محاسبه شده‌اند. "
            "این اعداد با وارد کردن اطلاعات یک بیمار جدید تغییر نمی‌کنند."
        )

    else:
        st.info(
            "برای مشاهده نتایج، ابتدا دکمه آموزش و ارزیابی همه الگوریتم‌ها را انتخاب کنید."
        )


# =====================================================
# صفحه سوم: درباره پروژه
# =====================================================

elif page == "درباره پروژه":

    section("معرفی سامانه")

    info_card(
        "سامانه هوشمند تحلیل و غربالگری دیابت",
        "این سامانه با استفاده از الگوریتم‌های یادگیری ماشین، اطلاعات مرتبط با سلامت افراد را بررسی می‌کند و نمونه‌ها را در سه گروه بدون دیابت، پیش‌دیابت و دیابت دسته‌بندی می‌کند."
    )

    section("الگوریتم‌های استفاده‌شده")

    for name in create_models().keys():
        info_card(
            "🧠 " + name,
            "یکی از روش‌های یادگیری ماشین مورد استفاده در این پژوهش."
        )

    section("معیارهای ارزیابی")

    info_card(
        "Accuracy و Error",
        "درصد پیش‌بینی‌های صحیح و نادرست مدل."
    )

    info_card(
        "Precision و Recall",
        "معیارهایی برای بررسی صحت پیش‌بینی و توانایی شناسایی نمونه‌های هر کلاس."
    )

    info_card(
        "F1 و Balanced Accuracy",
        "معیارهایی برای ارزیابی متعادل عملکرد مدل، به‌ویژه در داده‌های دارای عدم توازن کلاسی."
    )

    st.warning(
        "این سامانه یک پروژه پژوهشی است و جایگزین آزمایش پزشکی یا تشخیص پزشک نیست."
    )


# =====================================================
# فوتر
# =====================================================

st.markdown(
    """
    <div class="footer">
        سامانه هوشمند تحلیل و غربالگری دیابت
        <br>
        Diabetes Machine Learning Research System
    </div>
    """,
    unsafe_allow_html=True
)
