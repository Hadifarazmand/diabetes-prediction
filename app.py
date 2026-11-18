
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier


# =========================================================
# تنظیمات اولیه
# =========================================================

st.set_page_config(
    page_title="سامانه هوشمند تحلیل دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

POSSIBLE_FILES = [
    BASE_DIR / "dataset" / "diabetes_prediction_dataset.csv",
    BASE_DIR / "diabetes_prediction_dataset.csv"
]

DATA_FILE = next(
    (path for path in POSSIBLE_FILES if path.exists()),
    None
)

TARGET = "diabetes"

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

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CLASS_NAMES = {
    0: "دیابت ندارد",
    1: "دیابت دارد"
}


# =========================================================
# طراحی ظاهری
# =========================================================

st.markdown("""
<style>

:root {
    --navy: #173B57;
    --navy-dark: #102D43;
    --teal: #087F78;
    --teal-dark: #06645F;
    --page: #F1F6F9;
    --white: #FFFFFF;
    --text: #263746;
    --muted: #596B7B;
    --border: #DCE5EB;
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

[data-testid="stMarkdownContainer"] {
    direction: rtl;
    text-align: right;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--text);
    line-height: 1.9;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: var(--navy);
    font-weight: 900;
}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: var(--navy) !important;
    font-weight: 700 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

/* هدر اصلی */

.hero {
    background: linear-gradient(120deg, #173B57, #205D75, #087F78);
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 25px;
    direction: rtl;
    text-align: right;
    box-shadow: 0 8px 25px rgba(23, 59, 87, 0.12);
}

.hero-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    background: #FFFFFF;
    color: #087F78;
    border-radius: 17px;
    font-size: 30px;
    margin-bottom: 15px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 26px;
    font-weight: 900;
    margin: 0 0 12px 0;
}

.hero p {
    color: #EAF5F7 !important;
    font-size: 14px;
    margin: 0;
}

/* عنوان بخش‌ها */

.section-title {
    color: var(--navy);
    font-size: 19px;
    font-weight: 900;
    margin: 25px 0 14px;
    direction: rtl;
    text-align: right;
}

/* کارت‌ها */

.panel {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 3px 12px rgba(23, 59, 87, 0.04);
    direction: rtl;
    text-align: right;
}

.panel-title {
    color: var(--navy);
    font-size: 16px;
    font-weight: 900;
    margin-bottom: 9px;
}

.panel-text {
    color: var(--muted);
    font-size: 13px;
    line-height: 2;
}

/* کارت نتیجه */

.result-card {
    border-radius: 18px;
    padding: 25px;
    margin: 20px 0;
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
    background: var(--teal) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--teal) !important;
    border-radius: 11px !important;
    min-height: 46px;
    font-weight: 800 !important;
    width: 100%;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: var(--teal-dark) !important;
    border-color: var(--teal-dark) !important;
}

.stButton > button p,
.stFormSubmitButton > button p {
    color: #FFFFFF !important;
}

/* ورودی‌ها */

input,
textarea {
    background: #FFFFFF !important;
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

[data-baseweb="input"] {
    background: #FFFFFF !important;
}

[data-baseweb="input"] input {
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
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

/* منوی کناری */

[data-testid="stSidebar"] {
    background: #173B57 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
}

/* جدول */

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* نمودار و پیام */

[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stAlert"] p {
    color: #263746 !important;
}

/* فوتر */

.footer {
    text-align: center;
    color: #596B7B;
    border-top: 1px solid #DCE5EB;
    padding-top: 20px;
    margin-top: 40px;
    font-size: 12px;
    direction: rtl;
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
            تحلیل اطلاعات سلامت با استفاده از الگوریتم‌های یادگیری ماشین
        </p>
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


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

    else:
        st.markdown("""
        <div class="result-card result-positive">
            <span class="result-icon">!</span>
            <h2>دیابت دارد</h2>
            <p>
                بر اساس اطلاعات واردشده، مدل فرد را در گروه دارای دیابت قرار داده است.
                برای بررسی دقیق‌تر به پزشک مراجعه کنید.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# بارگذاری دیتاست
# =========================================================

@st.cache_data(show_spinner=False)
def load_data(file_path, modified_time):

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()

    required_columns = FEATURES + [TARGET]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "ستون‌های زیر در دیتاست پیدا نشدند: "
            + ", ".join(missing)
        )

    df = df[required_columns].copy()

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])

    df[TARGET] = df[TARGET].astype(int)

    df = df[df[TARGET].isin([0, 1])]

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "ستون diabetes باید دارای دو کلاس 0 و 1 باشد."
        )

    return df


# =========================================================
# تعریف الگوریتم‌ها
# =========================================================

def create_models():

    return {
        "رگرسیون لجستیک": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        ),

        "درخت تصمیم": DecisionTreeClassifier(
            max_depth=14,
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

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
    }


# =========================================================
# ساخت پیش‌پردازش و مدل
# =========================================================

def create_pipeline(model):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# =========================================================
# آموزش و ارزیابی همه الگوریتم‌ها
# =========================================================

def calculate_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    error_rate = 1 - accuracy

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

    try:
        roc_auc = roc_auc_score(y_test, probabilities)
    except Exception:
        roc_auc = 0

    try:
        pr_auc = average_precision_score(y_test, probabilities)
    except Exception:
        pr_auc = 0

    return {
        "درصد درستی": accuracy * 100,
        "درصد خطا": error_rate * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "Balanced Accuracy": balanced * 100,
        "ROC-AUC": roc_auc * 100,
        "PR-AUC": pr_auc * 100
    }


@st.cache_resource(show_spinner=False)
def train_all_models(file_path, modified_time):

    df = load_data(file_path, modified_time)

    X = df[FEATURES]
    y = df[TARGET]

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

    total = len(models)

    progress = st.progress(0)
    status = st.empty()

    for index, (name, algorithm) in enumerate(models.items()):

        status.info(f"در حال آموزش الگوریتم {name} ...")

        start_time = time.time()

        pipeline = create_pipeline(algorithm)

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
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    return {
        "models": trained_models,
        "results": results_df,
        "X_test": X_test,
        "y_test": y_test,
        "data_size": len(df)
    }


# =========================================================
# آموزش یک الگوریتم برای تحلیل بیمار
# =========================================================

def train_single_model(model_name, file_path, modified_time):

    df = load_data(file_path, modified_time)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    algorithm = create_models()[model_name]

    pipeline = create_pipeline(algorithm)

    pipeline.fit(X_train, y_train)

    return pipeline


# =========================================================
# بررسی وجود دیتاست
# =========================================================

if DATA_FILE is None:

    st.error(
        "فایل دیتاست پیدا نشد. فایل "
        "diabetes_prediction_dataset.csv "
        "را داخل پوشه dataset قرار دهید."
    )

    st.stop()


FILE_MODIFIED_TIME = DATA_FILE.stat().st_mtime_ns


# =========================================================
# منوی کناری
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:15px 5px 20px;">
            <div style="
                width:65px;height:65px;margin:auto;
                display:flex;align-items:center;justify-content:center;
                background:#FFFFFF;color:#087F78;
                border-radius:18px;font-size:31px;">
                🩺
            </div>
            <h3 style="color:#FFFFFF;margin-top:13px;">
                پنل تحلیل دیابت
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "منوی اصلی",
        [
            "تحلیل بیمار",
            "تحلیل و مقایسه الگوریتم‌ها"
        ],
        label_visibility="visible"
    )

    st.markdown("---")


# =========================================================
# هدر
# =========================================================

header()


# =========================================================
# بارگذاری اطلاعات
# =========================================================

try:

    df = load_data(
        str(DATA_FILE),
        FILE_MODIFIED_TIME
    )

    X = df[FEATURES]
    y = df[TARGET]

except Exception as e:

    st.error(f"خطا در بارگذاری دیتاست: {e}")

    st.stop()


# =========================================================
# بخش اول: تحلیل بیمار
# =========================================================

if page == "تحلیل بیمار":

    section("تحلیل اطلاعات بیمار")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">👤 فرم اطلاعات سلامت</div>
        <div class="panel-text">
            ابتدا الگوریتم موردنظر را انتخاب کنید.
            سپس اطلاعات فرد را وارد کرده و روی دکمه تحلیل کلیک کنید.
        </div>
    </div>
    """, unsafe_allow_html=True)

    algorithm_names = list(create_models().keys())

    selected_algorithm = st.selectbox(
        "🤖 انتخاب الگوریتم یادگیری ماشین",
        algorithm_names,
        key="patient_algorithm"
    )

    section("مشخصات فردی و اطلاعات سلامت")

    patient = {}

    with st.form("patient_form"):

        col1, col2 = st.columns(2)

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
                min_value=1,
                max_value=100,
                value=35
            )

            patient["bmi"] = st.number_input(
                "شاخص توده بدنی (BMI)",
                min_value=10.0,
                max_value=80.0,
                value=25.0,
                step=0.5
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

        with col2:

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
                    "current": "مصرف‌کننده فعلی",
                    "former": "مصرف در گذشته",
                    "ever": "سابقه مصرف",
                    "not current": "در حال حاضر مصرف نمی‌کنم"
                }[x]
            )

            patient["HbA1c_level"] = st.number_input(
                "میزان HbA1c",
                min_value=3.0,
                max_value=20.0,
                value=5.5,
                step=0.1
            )

            patient["blood_glucose_level"] = st.number_input(
                "قند خون (mg/dL)",
                min_value=50,
                max_value=500,
                value=100
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🔍 تحلیل اطلاعات بیمار",
            use_container_width=True
        )

    if submitted:

        try:

            with st.spinner("در حال تحلیل اطلاعات بیمار ..."):

                model = train_single_model(
                    selected_algorithm,
                    str(DATA_FILE),
                    FILE_MODIFIED_TIME
                )

                input_df = pd.DataFrame(
                    [patient],
                    columns=FEATURES
                )

                prediction = int(model.predict(input_df)[0])

                probability = float(
                    model.predict_proba(input_df)[0][1]
                )

                st.session_state.patient_result = prediction
                st.session_state.patient_probability = probability
                st.session_state.patient_model_name = selected_algorithm

        except Exception as e:

            st.error(f"خطا در تحلیل اطلاعات: {e}")

    if "patient_result" in st.session_state:

        section("نتیجه تحلیل بیمار")

        show_result(st.session_state.patient_result)

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">جزئیات تحلیل</div>
                <div class="panel-text">
                    الگوریتم استفاده‌شده:
                    <b>{st.session_state.patient_model_name}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        probability = st.session_state.patient_probability

        st.metric(
            "احتمال پیش‌بینی‌شده توسط مدل برای کلاس دیابت",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "این نتیجه صرفاً خروجی یک مدل یادگیری ماشین و برای اهداف پژوهشی است. "
            "این سامانه جایگزین آزمایش پزشکی یا تشخیص پزشک نیست."
        )


# =========================================================
# بخش دوم: تحلیل و مقایسه الگوریتم‌ها
# =========================================================

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    section("آموزش و مقایسه عملکرد الگوریتم‌ها")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">📊 ارزیابی الگوریتم‌های یادگیری ماشین</div>
        <div class="panel-text">
            در این قسمت عملکرد الگوریتم‌ها روی مجموعه داده آموزشی و آزمون
            بررسی می‌شود. تمام الگوریتم‌ها با داده‌های آزمون یکسان ارزیابی می‌شوند.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "🚀 آموزش و ارزیابی همه الگوریتم‌ها",
        use_container_width=True
    ):

        try:

            with st.spinner("در حال آموزش الگوریتم‌ها ..."):

                bundle = train_all_models(
                    str(DATA_FILE),
                    FILE_MODIFIED_TIME
                )

                st.session_state.evaluation_bundle = bundle

            st.success("آموزش و ارزیابی با موفقیت انجام شد.")

        except Exception as e:

            st.error(f"خطا در آموزش الگوریتم‌ها: {e}")

    if "evaluation_bundle" in st.session_state:

        bundle = st.session_state.evaluation_bundle

        results = bundle["results"]

        st.markdown("---")

        section("انتخاب الگوریتم")

        selected_algorithm = st.selectbox(
            "الگوریتم موردنظر را انتخاب کنید",
            results["الگوریتم"].tolist(),
            key="evaluation_algorithm"
        )

        selected_row = results[
            results["الگوریتم"] == selected_algorithm
        ].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "درصد درستی",
                f"{selected_row['درصد درستی']:.2f}%"
            )

        with col2:
            st.metric(
                "درصد خطا",
                f"{selected_row['درصد خطا']:.2f}%"
            )

        with col3:
            st.metric(
                "امتیاز F1",
                f"{selected_row['F1']:.2f}%"
            )

        section("انتخاب نوع آمار و جزئیات")

        statistics_type = st.selectbox(
            "نوع آمار موردنظر",
            [
                "معیارهای اصلی ارزیابی",
                "معیارهای تکمیلی",
                "ماتریس درهم‌ریختگی"
            ],
            key="statistics_type"
        )

        if statistics_type == "معیارهای اصلی ارزیابی":

            statistics_df = pd.DataFrame({
                "معیار": [
                    "Accuracy - درصد درستی",
                    "Error Rate - درصد خطا",
                    "Precision",
                    "Recall",
                    "F1-Score"
                ],
                "مقدار": [
                    f"{selected_row['درصد درستی']:.2f}%",
                    f"{selected_row['درصد خطا']:.2f}%",
                    f"{selected_row['Precision']:.2f}%",
                    f"{selected_row['Recall']:.2f}%",
                    f"{selected_row['F1']:.2f}%"
                ]
            })

            st.dataframe(
                statistics_df,
                use_container_width=True,
                hide_index=True
            )

        elif statistics_type == "معیارهای تکمیلی":

            statistics_df = pd.DataFrame({
                "معیار": [
                    "Balanced Accuracy",
                    "ROC-AUC",
                    "PR-AUC",
                    "زمان آموزش"
                ],
                "مقدار": [
                    f"{selected_row['Balanced Accuracy']:.2f}%",
                    f"{selected_row['ROC-AUC']:.2f}%",
                    f"{selected_row['PR-AUC']:.2f}%",
                    f"{selected_row['زمان آموزش']:.2f} ثانیه"
                ]
            })

            st.dataframe(
                statistics_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            model = bundle["models"][selected_algorithm]

            predictions = model.predict(bundle["X_test"])

            cm = confusion_matrix(
                bundle["y_test"],
                predictions,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(figsize=(6, 4))

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

            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.set_xticklabels([
                "No Diabetes",
                "Diabetes"
            ])

            ax.set_yticklabels([
                "No Diabetes",
                "Diabetes"
            ])

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

        # -------------------------------------------------
        # جدول مقایسه همه الگوریتم‌ها
        # -------------------------------------------------

        section("مقایسه عملکرد تمام الگوریتم‌ها")

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

        # -------------------------------------------------
        # نمودار مقایسه
        # -------------------------------------------------

        section("نمودار مقایسه الگوریتم‌ها")

        chart_type = st.selectbox(
            "انتخاب معیار برای نمایش نمودار",
            [
                "درصد درستی",
                "درصد خطا",
                "Precision",
                "Recall",
                "F1",
                "Balanced Accuracy"
            ],
            key="chart_metric"
        )

        chart_data = results.set_index("الگوریتم")[
            [chart_type]
        ]

        st.bar_chart(
            chart_data,
            use_container_width=True
        )

        st.info(
            "معیارهای ارزیابی بر اساس داده‌های آزمون محاسبه شده‌اند. "
            "این معیارها عملکرد مدل را روی مجموعه آزمون نشان می‌دهند "
            "و با ورود اطلاعات یک بیمار جدید تغییر نمی‌کنند."
        )

    else:

        st.info(
            "برای مشاهده آمار و مقایسه الگوریتم‌ها، "
            "ابتدا دکمه آموزش و ارزیابی همه الگوریتم‌ها را انتخاب کنید."
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
