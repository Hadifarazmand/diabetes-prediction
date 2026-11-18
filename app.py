
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

warnings.filterwarnings("ignore")


# ==================================================
# 1. تنظیمات صفحه
# ==================================================

st.set_page_config(
    page_title="سامانه پیش‌بینی دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. طراحی و رنگ‌بندی
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif !important;
}

.stApp {
    background-color: #F3F6FA;
    color: #24364B;
}

/* تیترهای اصلی */

h1, h2, h3, h4 {
    color: #17365D !important;
    font-family: 'Vazirmatn', sans-serif !important;
    font-weight: 800 !important;
}

/* متن‌های معمولی */

.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] {
    color: #26394F;
}

/* منوی کناری */

section[data-testid="stSidebar"] {
    background-color: #142D4E;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

/* تیتر اصلی بالای صفحه */

.hero {
    background: linear-gradient(120deg, #142D4E, #176B87);
    padding: 27px;
    border-radius: 16px;
    margin-bottom: 25px;
    color: #FFFFFF;
}

.hero-title {
    color: #FFFFFF !important;
    font-size: 27px;
    font-weight: 800;
    margin: 0;
}

.hero-description {
    color: #E4F1F8 !important;
    margin-top: 12px;
    font-size: 14px;
}

/* کارت‌های آماری */

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE5EF;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(20,45,78,0.05);
}

div[data-testid="stMetricLabel"] {
    color: #43566D !important;
}

div[data-testid="stMetricValue"] {
    color: #17365D !important;
    font-weight: 800;
}

/* ورودی‌ها */

div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF;
    border-radius: 9px;
}

div[data-testid="stNumberInput"] input {
    background-color: #FFFFFF;
    color: #17365D !important;
    border-radius: 9px;
}

/* فرم */

div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE5EF;
    border-radius: 15px;
    padding: 20px;
}

/* دکمه‌ها */

.stButton > button,
.stFormSubmitButton > button {
    background-color: #176B87;
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #10536B;
    color: #FFFFFF !important;
}

/* پیام راهنما */

.notice {
    background-color: #E5F1FA;
    border-right: 5px solid #176B87;
    border-radius: 10px;
    padding: 16px;
    color: #17365D !important;
    line-height: 2;
    margin: 12px 0;
}

.success-box {
    background-color: #E5F5EC;
    border-right: 5px solid #198754;
    border-radius: 10px;
    padding: 17px;
    color: #14532D !important;
    line-height: 2;
}

.warning-box {
    background-color: #FFF3DF;
    border-right: 5px solid #E59B27;
    border-radius: 10px;
    padding: 17px;
    color: #754C00 !important;
    line-height: 2;
}

/* جدول */

div[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border-radius: 12px;
}

/* جداکننده */

hr {
    border-color: #DCE5EF;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. مسیر دیتاست
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "diabetes_prediction_dataset.csv"
)


# ==================================================
# 4. مشخصات ستون‌ها
# ==================================================

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

TARGET = "diabetes"

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_history"
]

NUMERICAL_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]


# ==================================================
# 5. بارگذاری دیتاست
# ==================================================

@st.cache_data
def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"فایل دیتاست پیدا نشد: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"ستون‌های موردنیاز وجود ندارند: {missing}"
        )

    df = df[FEATURES + [TARGET]].copy()

    for col in NUMERICAL_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(subset=[TARGET], inplace=True)

    df = df[df[TARGET].isin([0, 1])].copy()

    df[TARGET] = df[TARGET].astype(int)

    df["gender"] = (
        df["gender"].astype(str).str.strip().str.title()
    )

    df["smoking_history"] = (
        df["smoking_history"].astype(str).str.strip().str.lower()
    )

    df.drop_duplicates(inplace=True)

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "دیتاست باید شامل دو کلاس صفر و یک باشد."
        )

    if df[TARGET].value_counts().min() < 2:
        raise ValueError(
            "تعداد نمونه‌های یکی از کلاس‌ها کافی نیست."
        )

    return df


# ==================================================
# 6. تقسیم داده‌ها
# ==================================================

@st.cache_data
def prepare_data():

    df = load_dataset()

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ==================================================
# 7. تعریف الگوریتم‌ها
# ==================================================

def get_algorithms():

    algorithms = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=150,
                max_depth=18,
                min_samples_split=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=150,
                max_depth=18,
                min_samples_split=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
    }

    try:
        from xgboost import XGBClassifier

        algorithms["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )

    except ImportError:
        pass

    return algorithms


# ==================================================
# 8. ساخت Pipeline
# ==================================================

def build_pipeline(classifier):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


# ==================================================
# 9. آموزش و ارزیابی الگوریتم
# ==================================================

def train_one_model(algorithm_name):

    X_train, X_test, y_train, y_test = prepare_data()

    algorithms = get_algorithms()

    model = build_pipeline(
        algorithms[algorithm_name]
    )

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    correct = int(np.sum(np.asarray(y_test) == y_pred))
    incorrect = int(np.sum(np.asarray(y_test) != y_pred))

    metrics = {

        "Algorithm": algorithm_name,

        "Total": len(y_test),

        "Correct": correct,

        "Incorrect": incorrect,

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp),

        "Accuracy": accuracy_score(y_test, y_pred),

        "Error Rate": 1 - accuracy_score(y_test, y_pred),

        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),

        "F1-score": f1_score(
            y_test, y_pred, zero_division=0
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test, y_pred
        ),

        "Training Time": training_time
    }

    return model, metrics, y_pred


# ==================================================
# 10. وضعیت ذخیره‌سازی
# ==================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}

if "model_predictions" not in st.session_state:
    st.session_state.model_predictions = {}

if "patient_result" not in st.session_state:
    st.session_state.patient_result = None


# ==================================================
# 11. سربرگ
# ==================================================

def show_header():

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🩺 سامانه هوشمند پیش‌بینی دیابت
        </div>
        <div class="hero-description">
            تحلیل اطلاعات پزشکی با استفاده از الگوریتم‌های یادگیری ماشین
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 12. اطلاعات دیتاست
# ==================================================

def show_dataset_info():

    df = load_dataset()

    st.subheader("اطلاعات دیتاست")

    c1, c2, c3 = st.columns(3)

    c1.metric("تعداد کل رکوردها", f"{len(df):,}")
    c2.metric("تعداد ویژگی‌های ورودی", len(FEATURES))
    c3.metric("تعداد کلاس‌ها", 2)

    st.markdown("### توزیع وضعیت دیابت")

    counts = (
        df[TARGET]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    chart_data = pd.DataFrame({
        "تعداد افراد": [
            int(counts.loc[0]),
            int(counts.loc[1])
        ]
    }, index=["بدون دیابت", "دارای دیابت"])

    st.bar_chart(chart_data)


# ==================================================
# 13. تحلیل بیمار
# ==================================================

def show_patient_analysis():

    st.subheader("تحلیل وضعیت بیمار")

    st.markdown("""
    <div class="notice">
    اطلاعات پزشکی را وارد کنید تا الگوریتم انتخابی
    وضعیت دیابت را پیش‌بینی کند.
    </div>
    """, unsafe_allow_html=True)

    algorithms = list(get_algorithms().keys())

    with st.form("patient_form"):

        st.markdown("### اطلاعات عمومی")

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "جنسیت",
                ["زن", "مرد"]
            )

            age = st.number_input(
                "سن",
                min_value=1,
                max_value=100,
                value=35
            )

            hypertension = st.selectbox(
                "سابقه فشار خون بالا",
                ["خیر", "بله"]
            )

            heart_disease = st.selectbox(
                "سابقه بیماری قلبی",
                ["خیر", "بله"]
            )

        with col2:

            smoking_history = st.selectbox(
                "سابقه مصرف دخانیات",
                [
                    "هرگز مصرف نکرده‌ام",
                    "در حال حاضر مصرف می‌کنم",
                    "قبلاً مصرف می‌کردم",
                    "سابقه مصرف دارم",
                    "اطلاعاتی ندارم",
                    "در حال حاضر مصرف نمی‌کنم"
                ]
            )

            bmi = st.number_input(
                "شاخص توده بدنی (BMI)",
                min_value=10.0,
                max_value=80.0,
                value=25.0,
                step=0.1
            )

            hba1c = st.number_input(
                "میانگین قند خون سه‌ماهه (HbA1c) - درصد",
                min_value=3.0,
                max_value=20.0,
                value=5.5,
                step=0.1,
                help=(
                    "عدد درج‌شده در برگه آزمایش با عنوان HbA1c یا A1C "
                    "را برحسب درصد وارد کنید. این مقدار با قند خون "
                    "لحظه‌ای متفاوت است."
                )
            )

            glucose = st.number_input(
                "میزان قند خون (mg/dL)",
                min_value=40,
                max_value=500,
                value=100
            )

        st.divider()

        selected_algorithm = st.selectbox(
            "انتخاب الگوریتم یادگیری ماشین",
            algorithms
        )

        submitted = st.form_submit_button(
            "انجام تحلیل و پیش‌بینی",
            use_container_width=True
        )

    if submitted:

        smoking_mapping = {
            "هرگز مصرف نکرده‌ام": "never",
            "در حال حاضر مصرف می‌کنم": "current",
            "قبلاً مصرف می‌کردم": "former",
            "سابقه مصرف دارم": "ever",
            "اطلاعاتی ندارم": "No Info",
            "در حال حاضر مصرف نمی‌کنم": "not current"
        }

        patient_data = pd.DataFrame([{

            "gender": "Female" if gender == "زن" else "Male",

            "age": age,

            "hypertension": 1 if hypertension == "بله" else 0,

            "heart_disease": 1 if heart_disease == "بله" else 0,

            "smoking_history": smoking_mapping[smoking_history],

            "bmi": bmi,

            "HbA1c_level": hba1c,

            "blood_glucose_level": glucose

        }])

        try:

            if selected_algorithm not in st.session_state.trained_models:

                with st.spinner("در حال آموزش الگوریتم..."):

                    model, metrics, predictions = train_one_model(
                        selected_algorithm
                    )

                    st.session_state.trained_models[selected_algorithm] = model

                    st.session_state.model_metrics[selected_algorithm] = metrics

                    st.session_state.model_predictions[selected_algorithm] = predictions

            model = st.session_state.trained_models[selected_algorithm]

            prediction = int(model.predict(patient_data)[0])

            st.session_state.patient_result = {
                "prediction": prediction,
                "algorithm": selected_algorithm
            }

        except Exception as e:

            st.error(f"خطا در پیش‌بینی: {e}")

    if st.session_state.patient_result is not None:

        result = st.session_state.patient_result

        st.divider()

        st.subheader("نتیجه تحلیل بیمار")

        if result["prediction"] == 1:

            st.error("نتیجه مدل: دیابت دارد")

        else:

            st.success("نتیجه مدل: دیابت ندارد")

        st.write(
            f"الگوریتم استفاده‌شده: **{result['algorithm']}**"
        )

        st.markdown("""
        <div class="warning-box">
        توجه: این نتیجه صرفاً پیش‌بینی یک مدل پژوهشی است
        و جایگزین تشخیص پزشک یا آزمایش پزشکی معتبر نیست.
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# 14. آموزش تمام الگوریتم‌ها
# ==================================================

def train_all_models():

    algorithms = list(get_algorithms().keys())

    progress = st.progress(0)

    status = st.empty()

    for i, name in enumerate(algorithms):

        status.info(f"در حال آموزش الگوریتم {name}")

        try:

            model, metrics, predictions = train_one_model(name)

            st.session_state.trained_models[name] = model

            st.session_state.model_metrics[name] = metrics

            st.session_state.model_predictions[name] = predictions

        except Exception as e:

            st.warning(f"خطا در آموزش {name}: {e}")

        progress.progress((i + 1) / len(algorithms))

    status.success("آموزش و ارزیابی الگوریتم‌ها به پایان رسید.")


# ==================================================
# 15. نمایش تعداد بیماران درست و اشتباه
# ==================================================

def show_patient_counts(algorithm_name):

    if algorithm_name not in st.session_state.model_metrics:
        return

    metrics = st.session_state.model_metrics[algorithm_name]

    total = metrics["Total"]
    correct = metrics["Correct"]
    incorrect = metrics["Incorrect"]

    st.markdown("### آمار تشخیص بیماران")

    st.markdown(
        f"""
        <div class="notice">
            الگوریتم <b>{algorithm_name}</b> روی
            <b>{total:,} بیمار از داده‌های آزمون</b> ارزیابی شده است.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "تعداد تشخیص درست",
            f"{correct:,}"
        )

    with c2:
        st.metric(
            "تعداد تشخیص اشتباه",
            f"{incorrect:,}"
        )

    with c3:
        st.metric(
            "مجموع بیماران آزمون",
            f"{total:,}"
        )

    st.markdown("#### تفکیک دقیق نتایج")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="success-box">
                <b>تشخیص‌های درست</b><br><br>
                افراد بدون دیابت که درست شناسایی شدند:
                <b>{metrics['TN']:,}</b><br>
                افراد دارای دیابت که درست شناسایی شدند:
                <b>{metrics['TP']:,}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="warning-box">
                <b>تشخیص‌های اشتباه</b><br><br>
                افراد بدون دیابت که مبتلا تشخیص داده شدند:
                <b>{metrics['FP']:,}</b><br>
                افراد دارای دیابت که غیرمبتلا تشخیص داده شدند:
                <b>{metrics['FN']:,}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### نمودار تعداد تشخیص‌ها")

    chart = pd.DataFrame({
        "تعداد": [correct, incorrect]
    }, index=["تشخیص درست", "تشخیص اشتباه"])

    st.bar_chart(chart)

    st.caption(
        "تمام اعداد این قسمت تعداد واقعی رکوردهای مجموعه آزمون هستند، نه درصد."
    )

    # ماتریس درهم‌ریختگی

    st.markdown("#### ماتریس درهم‌ریختگی")

    matrix = np.array([
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]]
    ])

    fig, ax = plt.subplots(figsize=(6, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["بدون دیابت", "دارای دیابت"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=",d"
    )

    ax.set_title(
        f"Confusion Matrix - {algorithm_name}",
        color="#17365D",
        fontsize=12
    )

    st.pyplot(fig)

    plt.close(fig)


# ==================================================
# 16. مقایسه الگوریتم‌ها
# ==================================================

def show_algorithm_analysis():

    st.subheader("تحلیل و مقایسه الگوریتم‌ها")

    st.markdown("""
    <div class="notice">
    در این بخش می‌توانید الگوریتم‌ها را آموزش دهید و تعداد
    بیمارانی را که درست یا اشتباه تشخیص داده‌اند مشاهده کنید.
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "آموزش و ارزیابی همه الگوریتم‌ها",
        use_container_width=True
    ):

        train_all_models()

    available = list(
        st.session_state.trained_models.keys()
    )

    if not available:

        st.info(
            "برای مشاهده آمار، ابتدا دکمه آموزش و ارزیابی همه الگوریتم‌ها را انتخاب کنید."
        )

        return

    st.divider()

    selected = st.selectbox(
        "الگوریتم موردنظر را انتخاب کنید",
        available,
        key="analysis_algorithm"
    )

    # نمایش تعداد واقعی بیماران درست و اشتباه
    show_patient_counts(selected)

    metrics = st.session_state.model_metrics[selected]

    st.divider()

    st.markdown("### سایر معیارهای آماری")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Error Rate",
        f"{metrics['Error Rate'] * 100:.2f}%"
    )

    c3.metric(
        "Training Time",
        f"{metrics['Training Time']:.3f} ثانیه"
    )

    with st.expander("مشاهده جزئیات سایر معیارها"):

        a, b, c = st.columns(3)

        a.metric(
            "Precision",
            f"{metrics['Precision'] * 100:.2f}%"
        )

        b.metric(
            "Recall",
            f"{metrics['Recall'] * 100:.2f}%"
        )

        c.metric(
            "F1-score",
            f"{metrics['F1-score'] * 100:.2f}%"
        )

        st.metric(
            "Balanced Accuracy",
            f"{metrics['Balanced Accuracy'] * 100:.2f}%"
        )

    st.divider()

    st.markdown("### جدول مقایسه تمام الگوریتم‌ها")

    rows = []

    for name, result in st.session_state.model_metrics.items():

        rows.append({

            "Algorithm": name,

            "Total Patients": result["Total"],

            "Correct": result["Correct"],

            "Incorrect": result["Incorrect"],

            "Accuracy (%)": round(
                result["Accuracy"] * 100, 2
            ),

            "Precision (%)": round(
                result["Precision"] * 100, 2
            ),

            "Recall (%)": round(
                result["Recall"] * 100, 2
            ),

            "F1-score (%)": round(
                result["F1-score"] * 100, 2
            ),

            "Training Time (s)": round(
                result["Training Time"], 3
            )
        })

    comparison = pd.DataFrame(rows)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### مقایسه تعداد تشخیص درست و اشتباه")

    count_chart = comparison.set_index("Algorithm")[
        ["Correct", "Incorrect"]
    ]

    st.bar_chart(count_chart)

    st.markdown("### مقایسه دقت الگوریتم‌ها")

    accuracy_chart = comparison.set_index("Algorithm")[
        ["Accuracy (%)"]
    ]

    st.bar_chart(accuracy_chart)


# ==================================================
# 17. درباره پروژه
# ==================================================

def show_about():

    st.subheader("درباره پروژه")

    st.markdown("""
    این سامانه با هدف پیش‌بینی وضعیت دیابت
    با استفاده از الگوریتم‌های یادگیری ماشین طراحی شده است.

    اطلاعات پزشکی و ویژگی‌های فردی به‌عنوان ورودی دریافت می‌شوند
    و الگوریتم‌های مختلف، وضعیت دیابت را در دو کلاس
    مبتلا و غیرمبتلا پیش‌بینی می‌کنند.

    **دیتاست:** Diabetes Prediction Dataset

    **منبع دیتاست:** Kaggle

    **کتابخانه‌ها و فناوری‌ها:**

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - XGBoost (اختیاری)
    """)

    st.markdown(
        "[مشاهده و دانلود دیتاست در Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)"
    )

    st.markdown("""
    <div class="warning-box">
    این سامانه یک پروژه پژوهشی و آموزشی است و خروجی آن
    جایگزین تشخیص قطعی پزشک نیست.
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 18. منوی کناری
# ==================================================

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:20px 5px;">
        <div style="font-size:42px;">🩺</div>
        <div style="
            color:#FFFFFF;
            font-size:22px;
            font-weight:800;
        ">
            Diabetes AI
        </div>
        <div style="
            color:#DCEAF5;
            font-size:12px;
            margin-top:8px;
        ">
            سامانه پیش‌بینی دیابت
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "منوی اصلی",
        [
            "تحلیل بیمار",
            "تحلیل و مقایسه الگوریتم‌ها",
            "درباره پروژه"
        ]
    )


# ==================================================
# 19. اجرای صفحات
# ==================================================

show_header()

if page == "تحلیل بیمار":

    show_patient_analysis()

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    show_algorithm_analysis()

elif page == "درباره پروژه":

    show_about()
