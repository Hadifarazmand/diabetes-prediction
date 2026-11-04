
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="سامانه بررسی اولیه دیابت",
    page_icon="🩺",
    layout="centered"
)

# ---------------- STYLE ----------------

st.markdown("""
<style>
.stApp {
    direction: rtl;
    text-align: right;
}
h1, h2, h3, p, label {
    text-align: right !important;
}
.welcome {
    background: linear-gradient(135deg, #087f8c, #13a89e);
    color: white;
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 20px;
    line-height: 2;
}
.info {
    background: #e8f4f5;
    color: #174b50;
    padding: 16px;
    border-radius: 12px;
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.title("🩺 سامانه هوشمند بررسی اولیه دیابت")

st.markdown("""
<div class="welcome">
<h2 style="color:white">سلام! به سامانه بررسی اولیه دیابت خوش آمدید 🌱</h2>
<p>
سلامتی شما اهمیت زیادی دارد. این سامانه با استفاده از الگوریتم‌های
یادگیری ماشین، اطلاعات و علائم شما را بررسی می‌کند و یک ارزیابی اولیه
از شباهت وضعیت شما به نمونه‌های مبتلا به دیابت ارائه می‌دهد.
</p>
<p>
لطفاً اطلاعات زیر را با دقت وارد کنید و در پایان دکمه بررسی را بزنید.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info">
<strong>توجه:</strong>
این سامانه صرفاً یک ابزار آموزشی و پژوهشی است.
نتیجه آن تشخیص پزشکی یا احتمال قطعی ابتلا به دیابت نیست.
برای تشخیص و تصمیم‌گیری درمانی حتماً به پزشک مراجعه کنید.
</div>
""", unsafe_allow_html=True)

st.divider()


# ---------------- DATA ----------------

DATA_PATH = Path(__file__).parent / "dataset" / "diabetes.csv"

FEATURES = [
    "Age", "Gender", "Polyuria", "Polydipsia",
    "sudden weight loss", "weakness", "Polyphagia",
    "visual blurring", "Itching", "delayed healing"
]

CATEGORICAL = FEATURES[1:]


# ---------------- MODEL TRAINING ----------------

@st.cache_resource
def train_models():

    df = pd.read_csv(DATA_PATH)

    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip().str.title()

    df["class"] = df["class"].astype(str).str.strip().str.title()

    X = df[FEATURES].copy()
    X["Age"] = pd.to_numeric(X["Age"], errors="coerce")

    valid = (
        X["Age"].notna()
        & df["class"].isin(["Positive", "Negative"])
    )

    X = X.loc[valid].copy()
    y = df.loc[valid, "class"].map({
        "Negative": 0,
        "Positive": 1
    })

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL
        ),
        (
            "age",
            StandardScaler(),
            ["Age"]
        )
    ])

    algorithms = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )
    }

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    results = {}
    pipelines = {}

    for name, algorithm in algorithms.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", algorithm)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        results[name] = {
            metric: np.mean(scores["test_" + metric])
            for metric in scoring
        }

        pipelines[name] = pipeline

    best_name = max(
        results,
        key=lambda name: results[name]["f1"]
    )

    final_model = pipelines[best_name]
    final_model.fit(X, y)

    return results, final_model, best_name


# ---------------- CHECK DATASET ----------------

if not DATA_PATH.exists():
    st.error(
        "فایل دیتاست پیدا نشد. فایل diabetes.csv باید "
        "داخل پوشه dataset قرار داشته باشد."
    )
    st.stop()


# ---------------- LOAD MODELS ----------------

with st.spinner("در حال آماده‌سازی و آموزش مدل‌های هوش مصنوعی..."):
    try:
        results, model, best_name = train_models()

    except Exception as error:
        st.error(f"خطا در آموزش مدل: {error}")
        st.stop()


# ---------------- PATIENT FORM ----------------

st.header("📋 اطلاعات بیمار")

st.write(
    "لطفاً مشخصات و علائم زیر را وارد کنید. "
    "پاسخ‌ها باید تا حد امکان دقیق باشند."
)

with st.form("patient_form"):

    age = st.number_input(
        "سن",
        min_value=1,
        max_value=120,
        value=35
    )

    gender = st.selectbox(
        "جنسیت",
        ["Male", "Female"],
        format_func=lambda x:
            "مرد" if x == "Male" else "زن"
    )

    st.subheader("علائم و وضعیت جسمانی")

    labels = {
        "Polyuria": "آیا ادرار بیش از حد دارید؟",
        "Polydipsia": "آیا تشنگی بیش از حد دارید؟",
        "sudden weight loss": "آیا کاهش وزن ناگهانی داشته‌اید؟",
        "weakness": "آیا احساس ضعف عمومی دارید؟",
        "Polyphagia": "آیا افزایش غیرعادی اشتها دارید؟",
        "visual blurring": "آیا تاری دید دارید؟",
        "Itching": "آیا خارش بدن دارید؟",
        "delayed healing": "آیا زخم‌های بدن شما دیرتر بهبود پیدا می‌کنند؟"
    }

    answers = {}

    for feature, question in labels.items():

        answers[feature] = st.radio(
            question,
            ["No", "Yes"],
            format_func=lambda x:
                "خیر" if x == "No" else "بله",
            horizontal=True,
            key=feature
        )

    submitted = st.form_submit_button(
        "🔍 شروع بررسی",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if submitted:

    patient_data = {
        "Age": age,
        "Gender": gender,
        **answers
    }

    patient_df = pd.DataFrame(
        [patient_data],
        columns=FEATURES
    )

    prediction = model.predict(patient_df)[0]

    probabilities = model.predict_proba(patient_df)[0]

    positive_index = list(model.classes_).index(1)

    positive_score = float(
        probabilities[positive_index]
    )

    st.divider()
    st.header("📊 نتیجه بررسی اولیه")

    st.metric(
        "امتیاز خروجی مثبت مدل",
        f"{positive_score * 100:.2f}%"
    )

    st.progress(positive_score)

    if prediction == 1:

        st.error(
            "مدل، الگوی اطلاعات واردشده را مشابه "
            "نمونه‌های مثبت دیابت تشخیص داده است."
        )

        st.write(
            "پیشنهاد می‌شود برای بررسی دقیق‌تر "
            "به پزشک مراجعه کنید."
        )

    else:

        st.success(
            "مدل، الگوی اطلاعات واردشده را مشابه "
            "نمونه‌های منفی دیابت تشخیص داده است."
        )

        st.write(
            "با این حال، این نتیجه ابتلا به دیابت را رد نمی‌کند."
        )

    st.caption(
        "درصد نمایش‌داده‌شده امتیاز خروجی مدل است، "
        "نه احتمال واقعی ابتلا به دیابت. "
        "این خروجی به‌تنهایی برای تصمیم پزشکی کافی نیست."
    )


# ---------------- RESEARCH RESULTS ----------------

st.divider()

with st.expander("مشاهده نتایج پژوهشی و مقایسه الگوریتم‌ها"):

    st.subheader("مقایسه عملکرد الگوریتم‌ها")

    result_df = pd.DataFrame(results).T * 100

    result_df.columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ]

    st.dataframe(
        result_df.round(2),
        use_container_width=True
    )

    st.success(
        f"بهترین الگوریتم بر اساس F1-score: {best_name}"
    )

    st.caption(
        "نتایج بر اساس اعتبارسنجی متقابل طبقه‌بندی‌شده "
        "۱۰ بخشی و اجرای SMOTE درون هر بخش محاسبه شده‌اند."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "سامانه پژوهشی پیش‌بینی اولیه دیابت | "
    "Machine Learning Research Project"
)
