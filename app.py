
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
    roc_auc_score,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)


# =========================================================
# تنظیمات اصلی
# =========================================================

st.set_page_config(
    page_title="سامانه هوشمند غربالگری دیابت",
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
    "گرادیان تقویتی"
]

MODE_SCREENING = "غربالگری اولیه (بدون آزمایش)"
MODE_FULL = "تحلیل کامل (با اطلاعات آزمایشگاهی)"


# =========================================================
# طراحی ظاهری
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800;900&display=swap');

:root {
    --navy: #173B57;
    --navy-dark: #102B40;
    --teal: #087F78;
    --teal-dark: #06645F;
    --background: #F3F7FA;
    --white: #FFFFFF;
    --text: #263746;
    --muted: #526779;
    --border: #DCE5EC;
}

html, body, [class*="st-"] {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background: var(--background);
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* متن‌ها */

h1, h2, h3, h4 {
    color: var(--navy) !important;
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
    font-weight: 800 !important;
}

p, li, label {
    color: var(--text);
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

[data-testid="stMarkdownContainer"] p {
    line-height: 2;
    color: var(--text);
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

[data-testid="stWidgetLabel"] p {
    color: var(--navy) !important;
    font-weight: 700 !important;
}

/* هدر اصلی */

.hero {
    background: linear-gradient(125deg, #173B57 0%, #205D75 55%, #087F78 100%);
    padding: 30px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 9px 25px rgba(23,59,87,0.14);
    direction: rtl;
    text-align: right;
}

.hero-icon {
    width: 60px;
    height: 60px;
    background: white;
    color: #087F78;
    border-radius: 17px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 31px;
    margin-bottom: 15px;
}

.hero h1 {
    color: white !important;
    font-size: 26px;
    margin: 0 0 10px 0;
}

.hero p {
    color: #E8F4F6 !important;
    font-size: 14px;
    margin: 0;
}

/* کارت‌ها */

.panel {
    background: white;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 17px;
    box-shadow: 0 3px 13px rgba(23,59,87,0.045);
    direction: rtl;
    text-align: right;
}

.panel-title {
    color: var(--navy);
    font-weight: 800;
    font-size: 16px;
    margin-bottom: 9px;
}

.panel-text {
    color: var(--muted);
    font-size: 13px;
    line-height: 2;
}

/* عنوان بخش */

.section-heading {
    color: var(--navy);
    font-size: 19px;
    font-weight: 900;
    margin: 28px 0 15px;
    direction: rtl;
    text-align: right;
}

/* آمار */

.stat-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 17px;
    text-align: right;
    direction: rtl;
    min-height: 115px;
}

.stat-label {
    color: #526779;
    font-size: 12px;
}

.stat-value {
    color: var(--navy);
    font-size: 23px;
    font-weight: 900;
    margin-top: 9px;
}

/* نتیجه */

.result-card {
    padding: 24px;
    border-radius: 17px;
    margin: 18px 0;
    text-align: center;
    direction: rtl;
}

.result-card h2 {
    font-size: 23px;
    margin: 10px 0;
}

.result-card p {
    font-size: 14px;
    line-height: 2;
}

.result-negative {
    background: #E9F7EF;
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

/* ورودی‌ها */

input, textarea {
    color: #263746 !important;
    background: white !important;
}

[data-baseweb="input"] {
    background: white !important;
}

[data-baseweb="input"] input {
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
}

[data-baseweb="select"] > div {
    background: white !important;
    border-color: #C9D6E0 !important;
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

/* دکمه‌ها */

.stButton > button,
.stFormSubmitButton > button {
    background: #087F78 !important;
    color: white !important;
    border: 1px solid #087F78 !important;
    border-radius: 11px !important;
    min-height: 45px;
    font-weight: 800 !important;
    transition: 0.2s;
}

.stButton > button p,
.stFormSubmitButton > button p {
    color: white !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #06645F !important;
    border-color: #06645F !important;
}

/* منوی کناری */

[data-testid="stSidebar"] {
    background: #173B57 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #D7E5ED !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] p {
    color: white !important;
}

/* تب‌ها */

button[data-baseweb="tab"] {
    color: #526779 !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #087F78 !important;
}

/* جدول */

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* هشدارها */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* فوتر */

.footer {
    border-top: 1px solid #DCE5EC;
    text-align: center;
    padding: 22px 5px;
    margin-top: 35px;
    color: #526779;
    font-size: 12px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem;
    }

    .hero {
        padding: 22px;
    }

    .hero h1 {
        font-size: 21px;
    }

    .panel {
        padding: 15px;
    }

    .section-heading {
        font-size: 17px;
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
        <h1>سامانه هوشمند تحلیل و غربالگری دیابت</h1>
        <p>
            بررسی اطلاعات سلامت با استفاده از الگوریتم‌های یادگیری ماشین
        </p>
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


def stat(label, value):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def show_result(prediction, probability=None):

    if prediction == 0:
        st.markdown("""
        <div class="result-card result-negative">
            <h2>✓ نتیجه: دیابت پیش‌بینی نشد</h2>
            <p>
                مدل بر اساس اطلاعات واردشده، فرد را در گروه بدون دیابت قرار داده است.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-card result-positive">
            <h2>! نتیجه: احتمال دیابت</h2>
            <p>
                مدل بر اساس اطلاعات واردشده، فرد را در گروه دیابت قرار داده است.
                برای بررسی دقیق‌تر، انجام آزمایش و مراجعه به پزشک ضروری است.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if probability is not None:
        st.metric(
            "احتمال پیش‌بینی‌شده توسط مدل برای کلاس دیابت",
            f"{probability:.2f}%"
        )


# =========================================================
# بارگذاری و آماده‌سازی دیتاست
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "فایل دیتاست پیدا نشد.\n"
            "مسیر مورد انتظار:\n"
            "dataset/diabetes_prediction_dataset.csv"
        )

    df = pd.read_csv(DATA_FILE)

    df.columns = df.columns.str.strip()

    required_columns = ALL_FEATURES + [TARGET]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"ستون‌های زیر در دیتاست پیدا نشدند: {missing}"
        )

    df = df[required_columns].copy()

    # پاک‌سازی مقادیر متنی
    for column in CATEGORICAL_FEATURES:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # تبدیل ستون‌های عددی
    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=[TARGET])

    df[TARGET] = df[TARGET].astype(int)

    # دیتاست اصلی فقط دو کلاس معتبر دارد
    df = df[df[TARGET].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("داده معتبر برای آموزش مدل پیدا نشد.")

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "دیتاست باید هر دو کلاس 0 و 1 را داشته باشد."
        )

    df["gender"] = df["gender"].fillna("Other")
    df["smoking_history"] = df["smoking_history"].fillna("No Info")

    return df


# =========================================================
# انتخاب ویژگی‌ها بر اساس نوع تحلیل
# =========================================================

def get_features(mode):

    if mode == MODE_SCREENING:
        return SCREENING_FEATURES

    return ALL_FEATURES


def get_numeric_features(features):
    return [
        column for column in features
        if column in NUMERIC_FEATURES
    ]


def get_categorical_features(features):
    return [
        column for column in features
        if column in CATEGORICAL_FEATURES
    ]


# =========================================================
# ساخت الگوریتم‌ها
# =========================================================

def create_models():

    return {
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
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        "درخت‌های بسیار تصادفی": ExtraTreesClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "گرادیان تقویتی": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=20,
            random_state=42
        )
    }


# =========================================================
# پیش‌پردازش داده‌ها
# =========================================================

def create_pipeline(model, features):

    numeric_features = get_numeric_features(features)
    categorical_features = get_categorical_features(features)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
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
# آموزش یک الگوریتم
# =========================================================

def train_one_model(model_name, df, features):

    X_train, X_test, y_train, y_test = split_data(
        df,
        features
    )

    models = create_models()

    if model_name not in models:
        raise ValueError("الگوریتم انتخاب‌شده معتبر نیست.")

    pipeline = create_pipeline(
        models[model_name],
        features
    )

    pipeline.fit(X_train, y_train)

    return pipeline


# =========================================================
# محاسبه معیارهای ارزیابی
# =========================================================

def calculate_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    balanced = balanced_accuracy_score(
        y_test,
        predictions
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    return {
        "درصد درستی": accuracy * 100,
        "درصد خطا": (1 - accuracy) * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "Balanced Accuracy": balanced * 100,
        "ROC-AUC": roc_auc * 100,
        "PR-AUC": pr_auc * 100
    }


# =========================================================
# آموزش و ارزیابی تمام الگوریتم‌ها
# =========================================================

def train_and_evaluate_all(df, features):

    X_train, X_test, y_train, y_test = split_data(
        df,
        features
    )

    models = create_models()

    trained_models = {}
    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(models)

    for index, (name, model) in enumerate(models.items()):

        status.info(f"در حال آموزش الگوریتم {name} ...")

        start_time = time.time()

        pipeline = create_pipeline(
            model,
            features
        )

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

    status.success("آموزش و ارزیابی الگوریتم‌ها با موفقیت انجام شد.")

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
# وضعیت برنامه
# =========================================================

if "model_store" not in st.session_state:
    st.session_state.model_store = {}

if "patient_result" not in st.session_state:
    st.session_state.patient_result = None

if "patient_probability" not in st.session_state:
    st.session_state.patient_probability = None

if "patient_algorithm" not in st.session_state:
    st.session_state.patient_algorithm = None

if "patient_mode" not in st.session_state:
    st.session_state.patient_mode = None


# =========================================================
# منوی کناری
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:18px 5px;">
        <div style="
            width:65px;
            height:65px;
            background:white;
            color:#087F78;
            border-radius:18px;
            display:flex;
            align-items:center;
            justify-content:center;
            margin:auto;
            font-size:32px;">
            🩺
        </div>

        <h3 style="color:white !important;margin-top:14px;">
            پنل هوشمند دیابت
        </h3>

        <p style="color:#D7E5ED !important;font-size:12px;">
            سامانه تحلیل داده‌های سلامت
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "بخش موردنظر را انتخاب کنید:",
        [
            "تحلیل اطلاعات بیمار",
            "آموزش و مقایسه الگوریتم‌ها",
            "درباره پروژه"
        ],
        key="main_page"
    )

    st.markdown("---")

    st.caption("Diabetes Machine Learning System")


# =========================================================
# بارگذاری داده
# =========================================================

header()

try:

    df = load_data()

except Exception as error:

    st.error(str(error))

    st.info(
        "فایل باید در مسیر dataset/diabetes_prediction_dataset.csv قرار داشته باشد."
    )

    st.stop()


# =========================================================
# اطلاعات کلی دیتاست
# =========================================================

if page == "درباره پروژه":

    section("معرفی دیتاست")

    c1, c2, c3 = st.columns(3)

    with c1:
        stat("تعداد کل نمونه‌ها", f"{len(df):,}")

    with c2:
        stat("تعداد ویژگی‌ها", str(len(ALL_FEATURES)))

    with c3:
        stat("تعداد کلاس‌ها", "2")

    section("توزیع نمونه‌ها")

    distribution = df[TARGET].value_counts().sort_index()

    distribution_df = pd.DataFrame({
        "وضعیت": ["بدون دیابت", "دیابت"],
        "تعداد": [
            int(distribution.get(0, 0)),
            int(distribution.get(1, 0))
        ]
    })

    st.dataframe(
        distribution_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# بخش اول: تحلیل اطلاعات بیمار
# =========================================================

if page == "تحلیل اطلاعات بیمار":

    section("تحلیل و بررسی اطلاعات بیمار")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">👤 فرم اطلاعات سلامت</div>
        <div class="panel-text">
            ابتدا نوع تحلیل و الگوریتم موردنظر را انتخاب کنید.
            سپس اطلاعات فرد را وارد کرده و نتیجه پیش‌بینی را مشاهده کنید.
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "نوع تحلیل",
        [
            MODE_SCREENING,
            MODE_FULL
        ],
        horizontal=False,
        key="patient_mode_select"
    )

    if mode == MODE_SCREENING:

        st.info(
            "در این حالت از اطلاعات اولیه فرد استفاده می‌شود و مقادیر HbA1c و قند خون وارد مدل نمی‌شوند."
        )

    else:

        st.info(
            "در این حالت تمام ویژگی‌ها، شامل HbA1c و قند خون، برای تحلیل استفاده می‌شوند."
        )

    selected_features = get_features(mode)

    section("انتخاب الگوریتم")

    selected_algorithm = st.selectbox(
        "الگوریتم یادگیری ماشین",
        ALGORITHMS,
        key="patient_algorithm_select"
    )

    section("مشخصات فردی")

    with st.form("patient_form"):

        col1, col2 = st.columns(2)

        patient = {}

        with col1:

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

            patient["bmi"] = st.number_input(
                "شاخص توده بدنی (BMI)",
                min_value=10.0,
                max_value=100.0,
                value=25.0,
                step=0.5
            )

            patient["hypertension"] = st.selectbox(
                "آیا فشار خون بالا دارید؟",
                [0, 1],
                format_func=lambda x: "بله" if x == 1 else "خیر"
            )

        with col2:

            patient["heart_disease"] = st.selectbox(
                "آیا سابقه بیماری قلبی دارید؟",
                [0, 1],
                format_func=lambda x: "بله" if x == 1 else "خیر"
            )

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
                    "current": "مصرف فعلی",
                    "former": "مصرف در گذشته",
                    "ever": "سابقه مصرف",
                    "not current": "در حال حاضر مصرف نمی‌کنم"
                }[x]
            )

        if mode == MODE_FULL:

            section("اطلاعات آزمایشگاهی")

            col3, col4 = st.columns(2)

            with col3:

                patient["HbA1c_level"] = st.number_input(
                    "مقدار HbA1c",
                    min_value=3.0,
                    max_value=20.0,
                    value=5.5,
                    step=0.1
                )

            with col4:

                patient["blood_glucose_level"] = st.number_input(
                    "قند خون (mg/dL)",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=1
                )

        submitted = st.form_submit_button(
            "🔍 تحلیل اطلاعات بیمار",
            use_container_width=True
        )

    if submitted:

        try:

            with st.spinner("در حال آموزش مدل و تحلیل اطلاعات..."):

                cache_key = mode + "_" + selected_algorithm

                if cache_key not in st.session_state.model_store:

                    model = train_one_model(
                        selected_algorithm,
                        df,
                        selected_features
                    )

                    st.session_state.model_store[cache_key] = model

                else:

                    model = st.session_state.model_store[cache_key]

                input_df = pd.DataFrame(
                    [patient],
                    columns=selected_features
                )

                prediction = int(model.predict(input_df)[0])

                probability = float(
                    model.predict_proba(input_df)[0][1] * 100
                )

                st.session_state.patient_result = prediction
                st.session_state.patient_probability = probability
                st.session_state.patient_algorithm = selected_algorithm
                st.session_state.patient_mode = mode

        except Exception as error:

            st.error(f"خطا در تحلیل اطلاعات: {error}")

    if st.session_state.patient_result is not None:

        section("نتیجه تحلیل")

        show_result(
            st.session_state.patient_result,
            st.session_state.patient_probability
        )

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">اطلاعات مدل استفاده‌شده</div>
            <div class="panel-text">
                الگوریتم: {st.session_state.patient_algorithm}
                <br>
                نوع تحلیل: {st.session_state.patient_mode}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "این نتیجه صرفاً یک پیش‌بینی پژوهشی بر اساس داده‌های آموزشی است "
            "و تشخیص پزشکی محسوب نمی‌شود. برای تشخیص دیابت باید آزمایش‌های معتبر "
            "و نظر پزشک بررسی شوند."
        )


# =========================================================
# بخش دوم: مقایسه الگوریتم‌ها
# =========================================================

elif page == "آموزش و مقایسه الگوریتم‌ها":

    section("آموزش و مقایسه عملکرد الگوریتم‌ها")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">📊 ارزیابی الگوریتم‌های یادگیری ماشین</div>
        <div class="panel-text">
            تمام الگوریتم‌ها با داده‌های آموزشی و آزمون یکسان ارزیابی می‌شوند.
            معیارهای ارزیابی برای مقایسه عملکرد آن‌ها نمایش داده خواهد شد.
        </div>
    </div>
    """, unsafe_allow_html=True)

    comparison_mode = st.radio(
        "نوع تحلیل برای مقایسه",
        [
            MODE_SCREENING,
            MODE_FULL
        ],
        key="comparison_mode"
    )

    comparison_features = get_features(comparison_mode)

    comparison_key = comparison_mode

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
            ) = train_and_evaluate_all(
                df,
                comparison_features
            )

            st.session_state.model_store[
                "comparison_" + comparison_key
            ] = {
                "models": trained_models,
                "results": results_df,
                "X_test": X_test,
                "y_test": y_test
            }

            st.success(
                "آموزش و ارزیابی همه الگوریتم‌ها با موفقیت انجام شد."
            )

        except Exception as error:

            st.error(f"خطا در آموزش الگوریتم‌ها: {error}")

    result_key = "comparison_" + comparison_key

    if result_key in st.session_state.model_store:

        saved = st.session_state.model_store[result_key]

        results = saved["results"]

        section("انتخاب الگوریتم برای مشاهده جزئیات")

        selected_algorithm = st.selectbox(
            "الگوریتم موردنظر",
            results["الگوریتم"].tolist(),
            key="comparison_algorithm_select"
        )

        selected_row = results[
            results["الگوریتم"] == selected_algorithm
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            stat(
                "درصد درستی",
                f"{selected_row['درصد درستی']:.2f}%"
            )

        with col2:
            stat(
                "درصد خطا",
                f"{selected_row['درصد خطا']:.2f}%"
            )

        with col3:
            stat(
                "امتیاز F1",
                f"{selected_row['F1']:.2f}%"
            )

        section("جزئیات عملکرد الگوریتم انتخاب‌شده")

        detail_metrics = [
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy",
            "ROC-AUC",
            "PR-AUC"
        ]

        detail_df = pd.DataFrame({
            "معیار ارزیابی": detail_metrics,
            "مقدار": [
                f"{selected_row[name]:.2f}%"
                for name in detail_metrics
            ]
        })

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )

        section("جدول مقایسه همه الگوریتم‌ها")

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

        for column in percentage_columns:

            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.2f}%"
            )

        display_df["زمان آموزش"] = display_df[
            "زمان آموزش"
        ].apply(
            lambda value: f"{value:.2f} ثانیه"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        section("نمودار مقایسه الگوریتم‌ها")

        chart_df = results.set_index("الگوریتم")[
            ["درصد درستی", "F1", "ROC-AUC"]
        ]

        st.bar_chart(
            chart_df,
            use_container_width=True
        )

        section("ماتریس درهم‌ریختگی")

        selected_model = saved["models"][selected_algorithm]

        predictions = selected_model.predict(saved["X_test"])

        cm = confusion_matrix(
            saved["y_test"],
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

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
                    fontsize=12,
                    fontweight="bold"
                )

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "معیارهای ارزیابی از داده‌های آزمون محاسبه شده‌اند. "
            "نتیجه پیش‌بینی بیمار جدا از ارزیابی کلی الگوریتم‌هاست."
        )

    else:

        st.info(
            "برای مشاهده نتایج، ابتدا دکمه آموزش و ارزیابی همه الگوریتم‌ها را انتخاب کنید."
        )


# =========================================================
# بخش سوم: درباره پروژه
# =========================================================

elif page == "درباره پروژه":

    section("درباره سامانه")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">
            سامانه هوشمند تحلیل و غربالگری دیابت
        </div>

        <div class="panel-text">
            این پروژه با استفاده از الگوریتم‌های یادگیری ماشین،
            اطلاعات مرتبط با سلامت افراد را بررسی می‌کند و
            بر اساس داده‌های موجود، احتمال قرار گرفتن فرد در گروه
            مبتلا به دیابت یا بدون دیابت را پیش‌بینی می‌کند.
        </div>

        <div class="panel-text">
            همچنین امکان آموزش و مقایسه چند الگوریتم یادگیری ماشین
            برای بررسی عملکرد آن‌ها در یک مجموعه داده مشترک فراهم شده است.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("اطلاعات دیتاست")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">منبع داده</div>
        <div class="panel-text">
            Diabetes Prediction Dataset
            <br>
            ویژگی‌های داده شامل سن، جنسیت، شاخص توده بدنی،
            فشار خون، بیماری قلبی، سابقه مصرف دخانیات،
            سطح HbA1c و میزان قند خون است.
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("الگوریتم‌های استفاده‌شده")

    algorithm_descriptions = {
        "رگرسیون لجستیک":
            "یک الگوریتم طبقه‌بندی برای پیش‌بینی احتمال تعلق نمونه به هر کلاس.",

        "درخت تصمیم":
            "مدلی مبتنی بر مجموعه‌ای از قواعد و تصمیم‌های متوالی.",

        "جنگل تصادفی":
            "ترکیبی از چندین درخت تصمیم برای بهبود عملکرد پیش‌بینی.",

        "درخت‌های بسیار تصادفی":
            "الگوریتمی مبتنی بر مجموعه‌ای از درخت‌های تصمیم با تصادفی‌سازی بیشتر.",

        "گرادیان تقویتی":
            "الگوریتمی که با بهبود تدریجی خطاهای مدل‌های قبلی، عملکرد پیش‌بینی را تقویت می‌کند."
    }

    for name, description in algorithm_descriptions.items():

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">{name}</div>
            <div class="panel-text">{description}</div>
        </div>
        """, unsafe_allow_html=True)

    section("معیارهای ارزیابی")

    metrics_text = [
        ("Accuracy", "درصد پیش‌بینی‌های صحیح از کل نمونه‌ها."),
        ("Error Rate", "درصد پیش‌بینی‌های نادرست."),
        ("Precision", "دقت پیش‌بینی نمونه‌های مبتلا به دیابت."),
        ("Recall", "توانایی شناسایی نمونه‌های مبتلا به دیابت."),
        ("F1-Score", "میانگین هماهنگ Precision و Recall."),
        ("Balanced Accuracy", "میانگین عملکرد مدل در شناسایی کلاس‌ها."),
        ("ROC-AUC", "توانایی مدل در تفکیک دو کلاس."),
        ("PR-AUC", "معیار عملکرد مدل با تمرکز بر کلاس مثبت.")
    ]

    for name, description in metrics_text:

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">{name}</div>
            <div class="panel-text">{description}</div>
        </div>
        """, unsafe_allow_html=True)

    st.warning(
        "این سامانه یک پروژه پژوهشی و ابزار غربالگری است "
        "و جایگزین تشخیص پزشکی، آزمایش خون یا نظر پزشک نیست."
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
