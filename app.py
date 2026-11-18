
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
    HistGradientBoostingClassifier,
    GradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. تنظیمات صفحه
# --------------------------------------------------

st.set_page_config(
    page_title="سامانه پیش‌بینی دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. استایل رابط کاربری
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif !important;
}

.stApp {
    background-color: #F4F7FB;
    color: #172B4D;
}

section[data-testid="stSidebar"] {
    background-color: #142D4E;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-family: 'Vazirmatn', sans-serif !important;
}

h1, h2, h3, h4 {
    color: #17365D !important;
    font-family: 'Vazirmatn', sans-serif !important;
}

p, label, span, div {
    font-family: 'Vazirmatn', sans-serif;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E8F1;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(20,45,78,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #52657D !important;
}

div[data-testid="stMetricValue"] {
    color: #17365D !important;
    font-weight: 800;
}

div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E0E8F1;
    border-radius: 16px;
    padding: 20px;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF;
    border-radius: 9px;
}

div[data-testid="stNumberInput"] input {
    border-radius: 9px;
    background-color: #FFFFFF;
}

.stButton > button,
.stFormSubmitButton > button {
    background-color: #176B87;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #10536B;
    border: none;
}

.notice {
    background-color: #E5F1FA;
    border-right: 5px solid #176B87;
    border-radius: 10px;
    padding: 16px;
    color: #17365D !important;
    margin: 12px 0;
    line-height: 2;
}

.success-box {
    background-color: #E6F5EC;
    border-right: 5px solid #198754;
    border-radius: 10px;
    padding: 16px;
    color: #14532D !important;
    margin: 12px 0;
}

.warning-box {
    background-color: #FFF4E5;
    border-right: 5px solid #E59B27;
    border-radius: 10px;
    padding: 16px;
    color: #754C00 !important;
    margin: 12px 0;
}

hr {
    border-color: #DCE5EF;
}

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. مسیر دیتاست
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "diabetes_prediction_dataset.csv"


# --------------------------------------------------
# 4. ستون‌های دیتاست
# --------------------------------------------------

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


# --------------------------------------------------
# 5. بارگذاری دیتاست
# --------------------------------------------------

@st.cache_data
def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"فایل دیتاست پیدا نشد:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    missing_columns = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "ستون‌های زیر در دیتاست وجود ندارند: "
            + ", ".join(missing_columns)
        )

    df = df[FEATURES + [TARGET]].copy()

    for col in NUMERICAL_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df[TARGET].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET])

    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    df["gender"] = df["gender"].astype(str).str.strip().str.title()

    df["smoking_history"] = (
        df["smoking_history"].astype(str).str.strip().str.lower()
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.drop_duplicates()

    if df[TARGET].nunique() != 2:
        raise ValueError("ستون diabetes باید شامل دو کلاس صفر و یک باشد.")

    if df[TARGET].value_counts().min() < 2:
        raise ValueError("تعداد نمونه‌های یکی از کلاس‌ها برای تقسیم داده کافی نیست.")

    return df


# --------------------------------------------------
# 6. تقسیم داده‌ها
# --------------------------------------------------

@st.cache_data
def prepare_data():

    df = load_dataset()

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 7. تعریف الگوریتم‌ها
# --------------------------------------------------

def get_algorithms():

    algorithms = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_leaf_nodes=20,
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


# --------------------------------------------------
# 8. ساخت Pipeline
# --------------------------------------------------

def build_pipeline(classifier):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return model


# --------------------------------------------------
# 9. آموزش یک الگوریتم
# --------------------------------------------------

def train_one_model(algorithm_name):

    X_train, X_test, y_train, y_test = prepare_data()

    algorithms = get_algorithms()

    if algorithm_name not in algorithms:
        raise ValueError("الگوریتم انتخاب‌شده در دسترس نیست.")

    model = build_pipeline(algorithms[algorithm_name])

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "Algorithm": algorithm_name,
        "Accuracy": accuracy,
        "Error Rate": 1 - accuracy,
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
        "Training Time": training_time,

        # تعداد واقعی پیش‌بینی‌ها
        "Correct": int(np.sum(np.asarray(y_test) == y_pred)),
        "Incorrect": int(np.sum(np.asarray(y_test) != y_pred))
    }

    return model, metrics, y_pred


# --------------------------------------------------
# 10. ذخیره وضعیت برنامه
# --------------------------------------------------

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}

if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}

if "model_predictions" not in st.session_state:
    st.session_state.model_predictions = {}

if "patient_result" not in st.session_state:
    st.session_state.patient_result = None


# --------------------------------------------------
# 11. سربرگ
# --------------------------------------------------

def show_header():

    st.markdown("""
    <div style="
        background: linear-gradient(120deg,#142D4E,#176B87);
        padding:25px;
        border-radius:16px;
        color:white;
        margin-bottom:22px;
        direction:rtl;
    ">
        <h1 style="color:white !important;margin:0;">
            🩺 سامانه هوشمند پیش‌بینی دیابت
        </h1>
        <p style="color:#E3F4FC;margin-top:10px;">
            تحلیل داده‌های پزشکی با استفاده از الگوریتم‌های یادگیری ماشین
        </p>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# 12. اطلاعات دیتاست
# --------------------------------------------------

def show_dataset_info():

    try:
        df = load_dataset()

        st.subheader("اطلاعات دیتاست")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("تعداد کل رکوردها", f"{len(df):,}")

        with col2:
            st.metric("تعداد ویژگی‌های ورودی", len(FEATURES))

        with col3:
            st.metric("تعداد کلاس‌ها", df[TARGET].nunique())

        st.markdown("### توزیع داده‌های هدف")

        target_counts = (
            df[TARGET]
            .value_counts()
            .rename(index={0: "بدون دیابت", 1: "دارای دیابت"})
        )

        st.bar_chart(target_counts)

    except Exception as e:
        st.error(f"خطا در خواندن دیتاست: {e}")


# --------------------------------------------------
# 13. صفحه تحلیل بیمار
# --------------------------------------------------

def show_patient_analysis():

    st.subheader("تحلیل وضعیت بیمار")

    st.markdown("""
    <div class="notice">
    اطلاعات زیر را وارد کنید تا الگوریتم یادگیری ماشین
    وضعیت دیابت را بر اساس داده‌های واردشده پیش‌بینی کند.
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
                "آیا سابقه فشار خون بالا دارید؟",
                ["خیر", "بله"]
            )

            heart_disease = st.selectbox(
                "آیا سابقه بیماری قلبی دارید؟",
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
                    "را برحسب درصد وارد کنید. این عدد با قند خون لحظه‌ای "
                    "متفاوت است. عدد حدسی وارد نکنید."
                )
            )

            glucose = st.number_input(
                "میزان قند خون (Blood Glucose) - mg/dL",
                min_value=40,
                max_value=500,
                value=100
            )

        st.divider()

        st.markdown("### انتخاب الگوریتم")

        selected_algorithm = st.selectbox(
            "الگوریتم یادگیری ماشین",
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

                with st.spinner("در حال آموزش الگوریتم انتخابی..."):

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
            st.error(f"خطا در انجام پیش‌بینی: {e}")

    if st.session_state.patient_result is not None:

        result = st.session_state.patient_result

        st.divider()
        st.subheader("نتیجه تحلیل")

        if result["prediction"] == 1:

            st.error(
                "نتیجه مدل: دیابت دارد"
            )

        else:

            st.success(
                "نتیجه مدل: دیابت ندارد"
            )

        st.write(
            f"الگوریتم استفاده‌شده: **{result['algorithm']}**"
        )

        st.markdown("""
        <div class="warning-box">
        توجه: این خروجی صرفاً نتیجه یک مدل پژوهشی و ابزار کمکی غربالگری است.
        این سامانه تشخیص پزشکی قطعی ارائه نمی‌کند و جایگزین آزمایش معتبر
        یا نظر پزشک نیست.
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# 14. آموزش و ارزیابی همه الگوریتم‌ها
# --------------------------------------------------

def train_all_models():

    algorithms = list(get_algorithms().keys())

    progress = st.progress(0)
    status = st.empty()

    for index, algorithm_name in enumerate(algorithms):

        status.info(f"در حال آموزش {algorithm_name} ...")

        try:

            model, metrics, predictions = train_one_model(
                algorithm_name
            )

            st.session_state.trained_models[algorithm_name] = model
            st.session_state.model_metrics[algorithm_name] = metrics
            st.session_state.model_predictions[algorithm_name] = predictions

        except Exception as e:

            st.warning(
                f"آموزش {algorithm_name} با خطا مواجه شد: {e}"
            )

        progress.progress((index + 1) / len(algorithms))

    status.success("فرایند آموزش و ارزیابی به پایان رسید.")


# --------------------------------------------------
# 15. نمایش آمار تعداد درست و اشتباه
# --------------------------------------------------

def show_prediction_counts(algorithm_name):

    if algorithm_name not in st.session_state.model_metrics:
        st.info("ابتدا الگوریتم را آموزش دهید.")
        return

    metrics = st.session_state.model_metrics[algorithm_name]

    st.markdown("### تعداد پیش‌بینی‌های الگوریتم")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "تعداد پیش‌بینی درست",
            f"{metrics['Correct']:,}"
        )

    with col2:
        st.metric(
            "تعداد پیش‌بینی اشتباه",
            f"{metrics['Incorrect']:,}"
        )

    with col3:
        st.metric(
            "مجموع داده‌های آزمون",
            f"{metrics['Correct'] + metrics['Incorrect']:,}"
        )

    st.markdown("#### تفکیک نوع پیش‌بینی")

    if algorithm_name in st.session_state.model_predictions:

        _, X_test, _, y_test = prepare_data()

        y_pred = st.session_state.model_predictions[algorithm_name]

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = matrix.ravel()

        a, b = st.columns(2)

        with a:
            st.metric("تشخیص درست افراد بدون دیابت (TN)", f"{tn:,}")
            st.metric("تشخیص اشتباه دیابت (FP)", f"{fp:,}")

        with b:
            st.metric("عدم شناسایی دیابت (FN)", f"{fn:,}")
            st.metric("شناسایی درست دیابت (TP)", f"{tp:,}")

        fig, ax = plt.subplots(figsize=(6, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["بدون دیابت", "دارای دیابت"]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d"
        )

        ax.set_title(f"Confusion Matrix - {algorithm_name}")

        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 16. صفحه تحلیل و مقایسه الگوریتم‌ها
# --------------------------------------------------

def show_algorithm_analysis():

    st.subheader("تحلیل و مقایسه الگوریتم‌ها")

    st.markdown("""
    <div class="notice">
    در این بخش می‌توانید الگوریتم‌های یادگیری ماشین را آموزش دهید
    و عملکرد آن‌ها را از نظر تعداد پیش‌بینی‌های درست و اشتباه
    و معیارهای آماری مقایسه کنید.
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "آموزش و ارزیابی همه الگوریتم‌ها",
        use_container_width=True
    ):

        train_all_models()

    available_algorithms = list(
        st.session_state.trained_models.keys()
    )

    if not available_algorithms:

        st.info(
            "هنوز الگوریتمی آموزش داده نشده است. "
            "برای مشاهده آمار، ابتدا دکمه آموزش و ارزیابی همه الگوریتم‌ها را انتخاب کنید."
        )

        return

    st.divider()

    st.markdown("### بررسی الگوریتم انتخابی")

    selected_algorithm = st.selectbox(
        "انتخاب الگوریتم برای مشاهده آمار",
        available_algorithms,
        key="analysis_algorithm_select"
    )

    # محاسبه تعداد درست و اشتباه الگوریتم انتخاب‌شده
    show_prediction_counts(selected_algorithm)

    st.divider()

    st.markdown("### معیارهای آماری الگوریتم")

    metrics = st.session_state.model_metrics[selected_algorithm]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['Accuracy'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Error Rate",
            f"{metrics['Error Rate'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Training Time",
            f"{metrics['Training Time']:.3f} ثانیه"
        )

    with st.expander("مشاهده سایر معیارهای آماری"):

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Precision",
            f"{metrics['Precision'] * 100:.2f}%"
        )

        c2.metric(
            "Recall",
            f"{metrics['Recall'] * 100:.2f}%"
        )

        c3.metric(
            "F1-score",
            f"{metrics['F1-score'] * 100:.2f}%"
        )

        st.metric(
            "Balanced Accuracy",
            f"{metrics['Balanced Accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------
    # جدول مقایسه الگوریتم‌ها
    # --------------------------------------------------

    st.divider()

    st.markdown("### جدول مقایسه الگوریتم‌ها")

    comparison_data = []

    for name, result in st.session_state.model_metrics.items():

        comparison_data.append({
            "Algorithm": name,
            "Correct Predictions": result["Correct"],
            "Incorrect Predictions": result["Incorrect"],
            "Accuracy (%)": round(result["Accuracy"] * 100, 2),
            "Precision (%)": round(result["Precision"] * 100, 2),
            "Recall (%)": round(result["Recall"] * 100, 2),
            "F1-score (%)": round(result["F1-score"] * 100, 2),
            "Training Time (s)": round(result["Training Time"], 3)
        })

    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # نمودار تعداد پیش‌بینی‌های درست و اشتباه
    # --------------------------------------------------

    st.markdown("### مقایسه تعداد پیش‌بینی‌های درست و اشتباه")

    chart_df = comparison_df.set_index("Algorithm")[
        ["Correct Predictions", "Incorrect Predictions"]
    ]

    st.bar_chart(chart_df)

    # --------------------------------------------------
    # نمودار مقایسه دقت
    # --------------------------------------------------

    st.markdown("### مقایسه Accuracy الگوریتم‌ها")

    accuracy_df = comparison_df.set_index("Algorithm")[
        ["Accuracy (%)"]
    ]

    st.bar_chart(accuracy_df)


# --------------------------------------------------
# 17. صفحه درباره پروژه
# --------------------------------------------------

def show_about():

    st.subheader("درباره پروژه")

    st.markdown("""
    این سامانه با هدف بررسی و پیش‌بینی وضعیت دیابت
    با استفاده از الگوریتم‌های یادگیری ماشین طراحی شده است.

    در این پروژه، اطلاعات پزشکی و ویژگی‌های فردی به‌عنوان
    ورودی دریافت می‌شوند و مدل‌های مختلف یادگیری ماشین
    وضعیت دیابت را در دو کلاس مبتلا و غیرمبتلا پیش‌بینی می‌کنند.

    **دیتاست مورد استفاده:** Diabetes Prediction Dataset

    **منبع:** Kaggle

    **فناوری‌های مورد استفاده:**
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - XGBoost (اختیاری)
    """)

    st.markdown(
        "[مشاهده و دریافت دیتاست در Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)"
    )

    st.warning(
        "این پروژه ماهیت پژوهشی و آموزشی دارد و خروجی آن تشخیص پزشکی قطعی نیست."
    )


# --------------------------------------------------
# 18. منوی کناری
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div style="text-align:center;padding:20px 5px;">
        <div style="font-size:42px;">🩺</div>
        <h3 style="color:white !important;">Diabetes AI</h3>
        <p style="font-size:12px;color:#C5D8E8 !important;">
            سامانه پیش‌بینی دیابت
        </p>
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


# --------------------------------------------------
# 19. اجرای صفحات
# --------------------------------------------------

show_header()

if page == "تحلیل بیمار":

    show_patient_analysis()

elif page == "تحلیل و مقایسه الگوریتم‌ها":

    show_algorithm_analysis()

elif page == "درباره پروژه":

    show_about()

