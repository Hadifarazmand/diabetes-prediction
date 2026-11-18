
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
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier
)

# =====================================================
# PAGE CONFIG
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

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

BINARY_FEATURES = [
    "hypertension",
    "heart_disease"
]

ALGORITHM_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Extra Trees": "Extra Trees",
    "Gradient Boosting": "Gradient Boosting"
}

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800;900&display=swap');

:root {
    --navy: #173B57;
    --navy-dark: #102D43;
    --teal: #087F78;
    --teal-dark: #06635D;
    --page: #F2F6FA;
    --white: #FFFFFF;
    --text: #263746;
    --muted: #596B7B;
    --border: #D9E3EA;
}

html, body, .stApp, .stApp * {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--page) !important;
    color: var(--text) !important;
}

.block-container {
    max-width: 1400px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--text) !important;
    line-height: 1.9 !important;
}

h1, h2, h3, h4 {
    color: var(--navy) !important;
    font-weight: 800 !important;
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

/* Hero */

.hero {
    background: linear-gradient(120deg, #173B57, #205D75, #087F78);
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 25px;
    direction: rtl;
    text-align: right;
    box-shadow: 0 8px 25px rgba(23,59,87,.12);
}

.hero-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 58px;
    height: 58px;
    background: white;
    color: #087F78 !important;
    border-radius: 17px;
    font-size: 29px;
    margin-bottom: 12px;
}

.hero h1 {
    color: white !important;
    font-size: 26px;
    font-weight: 900;
    margin: 0 0 10px;
}

.hero p {
    color: #F0F8FA !important;
    font-size: 14px;
    margin: 0;
}

/* Section */

.section-title {
    color: var(--navy) !important;
    font-size: 19px;
    font-weight: 900;
    margin: 24px 0 13px;
    text-align: right;
    direction: rtl;
}

/* Panels */

.panel {
    background: white !important;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 3px 12px rgba(23,59,87,.04);
    direction: rtl;
    text-align: right;
}

.panel-title {
    color: var(--navy) !important;
    font-size: 16px;
    font-weight: 900;
    margin-bottom: 8px;
}

.panel-text {
    color: #526577 !important;
    font-size: 13px;
    line-height: 2;
}

/* Notice */

.notice {
    background: #E4F1F8 !important;
    border: 1px solid #B7D5E6;
    border-right: 5px solid #287DA5;
    border-radius: 13px;
    padding: 17px 20px;
    margin: 16px 0;
    direction: rtl;
    text-align: right;
    color: #173B57 !important;
    font-size: 14px;
    line-height: 2;
    font-weight: 600;
}

.notice strong {
    color: #102D43 !important;
}

/* Result */

.result-card {
    border-radius: 17px;
    padding: 25px;
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

.result-icon {
    font-size: 38px;
    display: block;
}

.result-negative {
    background: #EAF7EF !important;
    border: 2px solid #A9D8BA;
}

.result-negative h2,
.result-negative p {
    color: #145C36 !important;
}

.result-positive {
    background: #FCEBEC !important;
    border: 2px solid #E9B5BA;
}

.result-positive h2,
.result-positive p {
    color: #8F2430 !important;
}

/* Statistics */

.stat-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 17px;
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

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: #087F78 !important;
    color: white !important;
    border: 1px solid #087F78 !important;
    border-radius: 11px !important;
    min-height: 46px;
    font-weight: 800 !important;
    transition: .2s;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: white !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #065F59 !important;
    border-color: #065F59 !important;
}

/* Inputs */

input, textarea {
    background: white !important;
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
    border-color: #C9D6E0 !important;
}

input::placeholder,
textarea::placeholder {
    color: #8292A0 !important;
}

[data-baseweb="input"] {
    background: white !important;
    border-radius: 10px !important;
}

[data-baseweb="input"] input {
    color: #263746 !important;
    -webkit-text-fill-color: #263746 !important;
}

/* Select boxes */

[data-baseweb="select"] > div {
    background: white !important;
    color: #263746 !important;
    border: 1px solid #C9D6E0 !important;
    border-radius: 11px !important;
    min-height: 44px !important;
}

[data-baseweb="select"] span {
    color: #263746 !important;
}

[data-baseweb="select"] svg {
    fill: #087F78 !important;
    color: #087F78 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background: white !important;
    border: 1px solid #D9E3EA !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 25px rgba(23,59,87,.14) !important;
}

[role="option"] {
    background: white !important;
    color: #263746 !important;
    padding: 10px 12px !important;
}

[role="option"]:hover {
    background: #E7F5F3 !important;
    color: #087F78 !important;
}

[aria-selected="true"] {
    background: #E7F5F3 !important;
    color: #087F78 !important;
}

/* Sidebar */

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #173B57 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label {
    color: white !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #D0DFE8 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: white !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #263746 !important;
}

/* Radio */

[data-testid="stRadio"] label {
    color: #263746 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: white !important;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 13px !important;
}

[data-testid="stAlert"] p {
    color: #263746 !important;
}

/* Expander */

[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: var(--navy) !important;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 11px !important;
}

/* Footer */

.footer {
    text-align: center;
    color: #596B7B !important;
    border-top: 1px solid var(--border);
    padding-top: 18px;
    margin-top: 35px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# DESIGN FUNCTIONS
# =====================================================

def header():
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🩺</div>
        <h1>سامانه هوشمند تحلیل و غربالگری دیابت</h1>
        <p>بررسی اطلاعات سلامت با الگوریتم‌های یادگیری ماشین</p>
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


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "فایل دیتاست پیدا نشد. مسیر مورد انتظار:\n"
            "dataset/diabetes_prediction_dataset.csv"
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"ستون‌های زیر در دیتاست وجود ندارند: {missing}"
        )

    df = df[required].copy()

    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["smoking_history"] = (
        df["smoking_history"].astype(str).str.strip().str.lower()
    )

    df["gender"] = df["gender"].map({
        "female": 0,
        "male": 1
    })

    smoking_map = {
        "never": 0,
        "no info": 1,
        "current": 2,
        "former": 3,
        "ever": 4,
        "not current": 5
    }

    df["smoking_history"] = df["smoking_history"].map(smoking_map)

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])]

    if df[TARGET].nunique() != 2:
        raise ValueError("دیتاست باید شامل هر دو کلاس 0 و 1 باشد.")

    return df


# =====================================================
# MODELS
# =====================================================

def create_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )
    }


def create_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def train_one_model(model_name, X_train, y_train):
    models = create_models()

    if model_name not in models:
        raise ValueError("الگوریتم انتخاب‌شده معتبر نیست.")

    pipeline = create_pipeline(models[model_name])
    pipeline.fit(X_train, y_train)

    return pipeline


# =====================================================
# EVALUATION
# =====================================================

def train_and_evaluate_all(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    trained_models = {}
    results = []

    progress = st.progress(0)
    status = st.empty()

    models = create_models()

    for index, (name, model) in enumerate(models.items()):

        status.info(f"در حال آموزش {name} ...")

        start = time.time()

        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)

        elapsed = time.time() - start

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        balanced = balanced_accuracy_score(y_test, predictions)

        results.append({
            "Algorithm": name,
            "Accuracy": accuracy * 100,
            "Error": (1 - accuracy) * 100,
            "Precision": precision * 100,
            "Recall": recall * 100,
            "F1": f1 * 100,
            "Balanced Accuracy": balanced * 100,
            "Training Time": elapsed
        })

        trained_models[name] = pipeline
        progress.progress((index + 1) / len(models))

    status.success("آموزش و ارزیابی الگوریتم‌ها کامل شد.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "F1", ascending=False
    ).reset_index(drop=True)

    return trained_models, results_df, X_test, y_test


# =====================================================
# SESSION STATE
# =====================================================

defaults = {
    "trained_models": {},
    "results_df": None,
    "X_test": None,
    "y_test": None,
    "patient_result": None,
    "patient_model_name": None,
    "data_signature": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# SIDEBAR
# =====================================================

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
# LOAD DATA
# =====================================================

header()

try:
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    signature = (
        len(df),
        str(df[TARGET].value_counts().sort_index().to_dict())
    )

    if st.session_state.data_signature != signature:
        st.session_state.trained_models = {}
        st.session_state.results_df = None
        st.session_state.X_test = None
        st.session_state.y_test = None
        st.session_state.patient_result = None
        st.session_state.data_signature = signature

except Exception as error:
    st.error(str(error))
    st.stop()


# =====================================================
# PATIENT ANALYSIS
# =====================================================

if page == "تحلیل اطلاعات بیمار":

    section("اطلاعات بیمار و انتخاب الگوریتم")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">👤 فرم اطلاعات سلامت</div>
        <p class="panel-text">
            اطلاعات سلامت فرد را وارد کنید، الگوریتم موردنظر را انتخاب کنید
            و سپس دکمه تحلیل اطلاعات را بزنید.
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected_algorithm = st.selectbox(
        "🤖 Machine Learning Algorithm",
        list(ALGORITHM_NAMES.keys()),
        key="patient_algorithm"
    )

    section("مشخصات فردی و سوابق سلامت")

    patient = {}

    col1, col2 = st.columns(2)

    with col1:

        patient["gender"] = st.selectbox(
            "جنسیت",
            ["زن", "مرد"],
            key="patient_gender"
        )
        patient["gender"] = 0 if patient["gender"] == "زن" else 1

        patient["age"] = st.number_input(
            "سن (سال)",
            min_value=0.0,
            max_value=120.0,
            value=35.0,
            step=1.0
        )

        patient["bmi"] = st.number_input(
            "شاخص توده بدنی (BMI)",
            min_value=10.0,
            max_value=80.0,
            value=25.0,
            step=0.1
        )

        patient["hypertension"] = st.selectbox(
            "آیا سابقه فشار خون بالا دارید؟",
            ["خیر", "بله"]
        )
        patient["hypertension"] = (
            1 if patient["hypertension"] == "بله" else 0
        )

        patient["heart_disease"] = st.selectbox(
            "آیا سابقه بیماری قلبی دارید؟",
            ["خیر", "بله"]
        )
        patient["heart_disease"] = (
            1 if patient["heart_disease"] == "بله" else 0
        )

    with col2:

        smoking_labels = {
            "هرگز سیگار نکشیده‌ام": "never",
            "در حال حاضر سیگار می‌کشم": "current",
            "قبلاً سیگار می‌کشیدم": "former",
            "سابقه مصرف دارم": "ever",
            "در حال حاضر مصرف نمی‌کنم": "not current",
            "اطلاعی ندارم": "no info"
        }

        smoking_choice = st.selectbox(
            "وضعیت مصرف سیگار",
            list(smoking_labels.keys())
        )

        patient["smoking_history"] = {
            "never": 0,
            "no info": 1,
            "current": 2,
            "former": 3,
            "ever": 4,
            "not current": 5
        }[smoking_labels[smoking_choice]]

        patient["HbA1c_level"] = st.number_input(
            "مقدار HbA1c (درصد)",
            min_value=3.0,
            max_value=20.0,
            value=5.5,
            step=0.1,
            help=(
                "عدد HbA1c را دقیقاً مطابق برگه آزمایش وارد کنید. "
                "این مقدار معمولاً به صورت درصد (%) گزارش می‌شود؛ "
                "مثلاً 5.7 یا 6.5."
            )
        )

        st.caption(
            "راهنما: مقدار HbA1c را از قسمت HbA1c یا Hemoglobin A1c برگه آزمایش بخوانید."
        )

        patient["blood_glucose_level"] = st.number_input(
            "قند خون (mg/dL)",
            min_value=50.0,
            max_value=500.0,
            value=100.0,
            step=1.0,
            help="عدد قند خون را مطابق نتیجه آزمایش و با واحد mg/dL وارد کنید."
        )

    st.markdown("""
    <div class="notice">
        <strong>نکته مهم:</strong>
        مقادیر HbA1c و قند خون باید از برگه آزمایش وارد شوند.
        این سامانه ابزار پژوهشی است و جایگزین آزمایش یا تشخیص پزشک نیست.
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "🔍 تحلیل اطلاعات بیمار",
        use_container_width=True
    ):

        try:
            with st.spinner("در حال اجرای الگوریتم یادگیری ماشین..."):

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )

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

        section("نتیجه تحلیل بیمار")

        show_result(st.session_state.patient_result)

        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">اطلاعات تحلیل</div>
            <p class="panel-text">
                الگوریتم استفاده‌شده:
                <strong>{st.session_state.patient_model_name}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "این خروجی یک پیش‌بینی پژوهشی است و تشخیص پزشکی قطعی محسوب نمی‌شود."
        )


# =====================================================
# ALGORITHM COMPARISON
# =====================================================

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    section("آموزش و مقایسه عملکرد الگوریتم‌ها")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">📊 مقایسه الگوریتم‌های یادگیری ماشین</div>
        <p class="panel-text">
            الگوریتم‌ها با داده‌های آموزشی یکسان آموزش داده می‌شوند
            و عملکرد آن‌ها روی داده‌های آزمون مقایسه خواهد شد.
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

            st.success("آموزش و ارزیابی الگوریتم‌ها با موفقیت انجام شد.")

        except Exception as error:
            st.error(f"خطا در آموزش الگوریتم‌ها: {error}")

    if st.session_state.results_df is not None:

        results = st.session_state.results_df

        section("انتخاب الگوریتم برای مشاهده آمار")

        selected = st.selectbox(
            "Machine Learning Algorithm",
            results["Algorithm"].tolist(),
            key="comparison_algorithm"
        )

        row = results[
            results["Algorithm"] == selected
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            stat("✓", "درصد درستی", f"{row['Accuracy']:.2f}%")

        with c2:
            stat("×", "درصد خطا", f"{row['Error']:.2f}%")

        with c3:
            stat("★", "امتیاز F1", f"{row['F1']:.2f}%")

        section("آمار تکمیلی الگوریتم انتخاب‌شده")

        detail_df = pd.DataFrame({
            "معیار": [
                "Precision",
                "Recall",
                "F1",
                "Balanced Accuracy",
                "زمان آموزش"
            ],
            "مقدار": [
                f"{row['Precision']:.2f}%",
                f"{row['Recall']:.2f}%",
                f"{row['F1']:.2f}%",
                f"{row['Balanced Accuracy']:.2f}%",
                f"{row['Training Time']:.2f} ثانیه"
            ]
        })

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True
        )

        section("مقایسه عملکرد تمام الگوریتم‌ها")

        display_df = results.copy()

        for column in [
            "Accuracy",
            "Error",
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy"
        ]:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.2f}%"
            )

        display_df["Training Time"] = display_df[
            "Training Time"
        ].map(lambda value: f"{value:.2f} ثانیه")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        section("نمودار مقایسه درصد درستی و خطا")

        chart = results.set_index("Algorithm")[
            ["Accuracy", "Error"]
        ]

        st.bar_chart(chart, use_container_width=True)

        section("ماتریس درهم‌ریختگی")

        model = st.session_state.trained_models[selected]

        predictions = model.predict(st.session_state.X_test)

        cm = confusion_matrix(
            st.session_state.y_test,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))

        im = ax.imshow(cm, cmap="GnBu", interpolation="nearest")

        ax.set_title("Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")

        labels = ["No Diabetes", "Diabetes"]

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        threshold = cm.max() / 2 if cm.max() > 0 else 0

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "#102D43",
                    fontweight="bold"
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        <div class="notice">
            <strong>توضیح:</strong>
            درصد درستی و خطا از داده‌های آزمون محاسبه می‌شوند.
            این معیارها عملکرد کلی الگوریتم را نشان می‌دهند،
            نه احتمال ابتلای یک بیمار مشخص.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="notice">
            <strong>هنوز آماری برای نمایش وجود ندارد.</strong>
            برای مشاهده آمار و مقایسه الگوریتم‌ها،
            ابتدا دکمه «آموزش و ارزیابی همه الگوریتم‌ها» را انتخاب کنید.
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# ABOUT
# =====================================================

elif page == "درباره پروژه":

    section("معرفی سامانه")

    st.markdown("""
    <div class="panel">
        <div class="panel-title">
            سامانه هوشمند تحلیل و غربالگری دیابت
        </div>
        <p class="panel-text">
            این سامانه با استفاده از الگوریتم‌های یادگیری ماشین،
            اطلاعات سلامت افراد را بررسی کرده و آن‌ها را در یکی از
            دو گروه بدون دیابت یا دیابت دسته‌بندی می‌کند.
        </p>
        <p class="panel-text">
            همچنین امکان آموزش، ارزیابی و مقایسه عملکرد چند الگوریتم
            یادگیری ماشین در این سامانه فراهم شده است.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section("الگوریتم‌های مورد استفاده")

    for name in create_models().keys():
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🧠 {name}</div>
        </div>
        """, unsafe_allow_html=True)

    section("معیارهای ارزیابی")

    st.markdown("""
    - **Accuracy:** درصد پیش‌بینی‌های صحیح.
    - **Error:** درصد پیش‌بینی‌های نادرست.
    - **Precision:** صحت پیش‌بینی‌های مثبت.
    - **Recall:** توانایی شناسایی نمونه‌های مثبت.
    - **F1:** میانگین هماهنگ Precision و Recall.
    - **Balanced Accuracy:** میانگین عملکرد مدل در کلاس‌ها.
    """)

    st.warning(
        "این سامانه یک پروژه پژوهشی است و جایگزین تشخیص پزشکی نیست."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    سامانه هوشمند تحلیل و غربالگری دیابت
    <br>
    Diabetes Machine Learning Research System
</div>
""", unsafe_allow_html=True)
