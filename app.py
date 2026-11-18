
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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. تنظیمات اولیه
# --------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"

st.set_page_config(
    page_title="سامانه هوشمند غربالگری دیابت",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# 2. ظاهر سایت
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #071b2c 0%, #102b42 55%, #123b4b 100%);
    color: #f4f8fb;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    font-family: 'Vazirmatn', sans-serif !important;
}

.hero {
    background: linear-gradient(120deg, #0c4050, #087f8c);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.12);
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(0,0,0,.18);
}

.hero h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
}

.hero p {
    color: #d5f4f1;
    font-size: 15px;
}

.metric-card {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.12);
    padding: 18px;
    border-radius: 17px;
    text-align: center;
    min-height: 110px;
}

.metric-card h3 {
    color: #a9dfe1;
    font-size: 14px;
}

.metric-card h2 {
    color: white;
    font-size: 27px;
}

.info-card {
    background: rgba(255,255,255,.07);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,.10);
    margin-bottom: 12px;
}

div.stButton > button {
    background: linear-gradient(90deg, #08a6a6, #087f8c);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #0bb9b1;
    color: white;
}

[data-testid="stForm"] {
    background: rgba(255,255,255,.045);
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.10);
}

[data-testid="stExpander"] {
    background: rgba(255,255,255,.055);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.12);
}

.stAlert {
    border-radius: 14px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. اطلاعات دیتاست‌ها
# --------------------------------------------------

CDC_FILE = DATA_DIR / "cdc_diabetes_health_indicators.csv"
KAGGLE_FILE = DATA_DIR / "diabetes_prediction_dataset.csv"

CDC_TARGET = "Diabetes_binary"
KAGGLE_TARGET = "diabetes"

CDC_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income"
]

KAGGLE_FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level",
    "blood_glucose_level"
]

CDC_LABELS = {
    "HighBP": "فشار خون بالا",
    "HighChol": "کلسترول بالا",
    "CholCheck": "آزمایش کلسترول در ۵ سال اخیر",
    "BMI": "شاخص توده بدنی (BMI)",
    "Smoker": "سابقه مصرف سیگار",
    "Stroke": "سابقه سکته مغزی",
    "HeartDiseaseorAttack": "سابقه بیماری قلبی یا حمله قلبی",
    "PhysActivity": "فعالیت بدنی در ۳۰ روز اخیر",
    "Fruits": "مصرف روزانه میوه",
    "Veggies": "مصرف روزانه سبزیجات",
    "HvyAlcoholConsump": "مصرف زیاد الکل",
    "AnyHealthcare": "دسترسی به خدمات درمانی",
    "NoDocbcCost": "عدم مراجعه به پزشک به دلیل هزینه",
    "GenHlth": "وضعیت عمومی سلامت",
    "MentHlth": "روزهای سلامت روان نامناسب در ماه",
    "PhysHlth": "روزهای سلامت جسمانی نامناسب در ماه",
    "DiffWalk": "مشکل در راه رفتن",
    "Sex": "جنسیت کدگذاری‌شده در دیتاست",
    "Age": "گروه سنی",
    "Education": "سطح تحصیلات",
    "Income": "گروه درآمدی"
}

# --------------------------------------------------
# 4. توابع خواندن و آماده‌سازی داده
# --------------------------------------------------

def read_csv(path):
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def convert_target(series):
    if series.dtype == "object":
        values = series.astype(str).str.strip().str.lower()

        mapping = {
            "1": 1,
            "yes": 1,
            "positive": 1,
            "diabetes": 1,
            "true": 1,
            "0": 0,
            "no": 0,
            "negative": 0,
            "no diabetes": 0,
            "false": 0
        }

        result = values.map(mapping)

        if result.isna().any():
            raise ValueError("مقادیر ستون هدف باید صفر و یک یا معادل متنی آن باشند.")

        return result.astype(int)

    result = pd.to_numeric(series, errors="coerce")

    if result.isna().any() or not set(result.unique()).issubset({0, 1}):
        raise ValueError("ستون هدف باید فقط شامل مقادیر صفر و یک باشد.")

    return result.astype(int)


def prepare_data(df, dataset_name, mode):
    if dataset_name == "CDC":
        target = CDC_TARGET
        features = CDC_FEATURES
    else:
        target = KAGGLE_TARGET

        if mode == "غربالگری اولیه بدون آزمایش":
            features = [
                "gender", "age", "hypertension",
                "heart_disease", "smoking_history", "bmi"
            ]
        else:
            features = KAGGLE_FEATURES

    missing = [c for c in [target] + features if c not in df.columns]

    if missing:
        raise ValueError(
            "ستون‌های زیر در فایل دیتاست وجود ندارند: "
            + ", ".join(missing)
        )

    data = df[features + [target]].copy()
    data = data.dropna(subset=[target])

    y = convert_target(data[target])
    X = data[features].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str).str.strip()
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, features


# --------------------------------------------------
# 5. ساخت الگوریتم‌ها
# --------------------------------------------------

def build_models(X):

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    numeric_process = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_process = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_process, numeric_cols),
        ("categorical", categorical_process, categorical_cols)
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
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
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=25,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42
        )
    }

    final_models = {}

    for name, estimator in models.items():
        final_models[name] = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator)
        ])

    # XGBoost اختیاری است؛ اگر نصب نشده باشد، سایر الگوریتم‌ها اجرا می‌شوند.
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            random_state=42
        )

        final_models["XGBoost"] = Pipeline([
            ("preprocessing", preprocessor),
            ("model", xgb)
        ])

    except Exception:
        pass

    return final_models


# --------------------------------------------------
# 6. آموزش و ارزیابی
# --------------------------------------------------

def train_and_evaluate(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = build_models(X)

    results = []
    fitted_models = {}
    predictions = {}

    progress = st.progress(0)
    status = st.empty()

    total = len(models)

    for index, (name, model) in enumerate(models.items()):

        status.write(f"در حال آموزش الگوریتم: {name}")

        try:
            # وزن‌دهی کلاس‌ها برای XGBoost
            if name == "XGBoost":
                from sklearn.utils.class_weight import compute_sample_weight

                weights = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train
                )

                model.fit(
                    X_train,
                    y_train,
                    model__sample_weight=weights
                )
            else:
                model.fit(X_train, y_train)

            predicted = model.predict(X_test)

            probability = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, predicted)
            error_rate = 1 - accuracy

            precision = precision_score(
                y_test, predicted, zero_division=0
            )

            recall = recall_score(
                y_test, predicted, zero_division=0
            )

            f1 = f1_score(
                y_test, predicted, zero_division=0
            )

            roc = roc_auc_score(y_test, probability)
            pr_auc = average_precision_score(y_test, probability)

            balanced_acc = balanced_accuracy_score(
                y_test, predicted
            )

            results.append({
                "Algorithm": name,
                "Accuracy (%)": accuracy * 100,
                "Error (%)": error_rate * 100,
                "Precision (%)": precision * 100,
                "Recall (%)": recall * 100,
                "F1 (%)": f1 * 100,
                "ROC-AUC (%)": roc * 100,
                "PR-AUC (%)": pr_auc * 100,
                "Balanced Accuracy (%)": balanced_acc * 100
            })

            fitted_models[name] = model
            predictions[name] = {
                "predicted": predicted,
                "probability": probability
            }

        except Exception as error:
            st.warning(f"الگوریتم {name} اجرا نشد: {error}")

        progress.progress((index + 1) / total)

    progress.empty()
    status.empty()

    if not results:
        raise RuntimeError("هیچ الگوریتمی با موفقیت اجرا نشد.")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by="F1 (%)",
        ascending=False
    ).reset_index(drop=True)

    return (
        result_df,
        fitted_models,
        predictions,
        X_test,
        y_test
    )


# --------------------------------------------------
# 7. سربرگ سایت
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🩺 سامانه هوشمند غربالگری دیابت</h1>
    <p>
        تحلیل داده‌های سلامت با استفاده از الگوریتم‌های یادگیری ماشین
        و مقایسه علمی عملکرد مدل‌ها
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "این سامانه یک پروژه آموزشی و پژوهشی است و جایگزین آزمایش "
    "قند خون، تشخیص یا نظر پزشک نیست."
)

# --------------------------------------------------
# 8. انتخاب دیتاست
# --------------------------------------------------

st.sidebar.title("تنظیمات سامانه")

dataset_name = st.sidebar.radio(
    "انتخاب منبع داده",
    ["CDC", "Kaggle"]
)

mode = "همه ویژگی‌ها"

if dataset_name == "Kaggle":
    mode = st.sidebar.radio(
        "نوع تحلیل",
        [
            "غربالگری اولیه بدون آزمایش",
            "همه ویژگی‌ها"
        ]
    )

if dataset_name == "CDC":
    file_path = CDC_FILE
    target_name = CDC_TARGET
else:
    file_path = KAGGLE_FILE
    target_name = KAGGLE_TARGET

df = read_csv(file_path)

if df is None:
    st.error(
        f"فایل دیتاست پیدا نشد: dataset/{file_path.name}"
    )
    st.stop()

try:
    X, y, used_features = prepare_data(
        df,
        dataset_name,
        mode
    )
except Exception as error:
    st.error(str(error))
    st.stop()

# --------------------------------------------------
# 9. آمار دیتاست
# --------------------------------------------------

st.subheader("📊 مشخصات دیتاست انتخاب‌شده")

a, b, c, d = st.columns(4)

with a:
    st.markdown(
        f'<div class="metric-card"><h3>تعداد رکورد</h3>'
        f'<h2>{len(X):,}</h2></div>',
        unsafe_allow_html=True
    )

with b:
    st.markdown(
        f'<div class="metric-card"><h3>تعداد ویژگی</h3>'
        f'<h2>{len(used_features)}</h2></div>',
        unsafe_allow_html=True
    )

with c:
    st.markdown(
        f'<div class="metric-card"><h3>کلاس مثبت</h3>'
        f'<h2>{int((y == 1).sum()):,}</h2></div>',
        unsafe_allow_html=True
    )

with d:
    st.markdown(
        f'<div class="metric-card"><h3>کلاس منفی</h3>'
        f'<h2>{int((y == 0).sum()):,}</h2></div>',
        unsafe_allow_html=True
    )

if dataset_name == "CDC":
    st.caption(
        "در این نسخه از CDC، کلاس مثبت شامل افراد دارای پیش‌دیابت یا دیابت است."
    )
else:
    st.caption(
        "دیتاست Kaggle باید از نظر منشأ داده و نحوه تولید برچسب بررسی شود."
    )

# --------------------------------------------------
# 10. فرم اطلاعات کاربر
# --------------------------------------------------

st.divider()
st.subheader("👤 ورود اطلاعات برای بررسی نمونه")

st.write(
    "اطلاعات را مطابق ویژگی‌های دیتاست انتخاب‌شده وارد کن."
)

def yes_no_input(label, key):
    value = st.selectbox(
        label,
        ["خیر", "بله"],
        key=key
    )
    return 1 if value == "بله" else 0


def make_user_input():

    values = {}

    if dataset_name == "CDC":

        st.markdown("#### اطلاعات سلامت و سبک زندگی")

        cols = st.columns(2)

        binary_labels = {
            "HighBP": "فشار خون بالا",
            "HighChol": "کلسترول بالا",
            "CholCheck": "آزمایش کلسترول در ۵ سال اخیر",
            "Smoker": "سابقه مصرف حداقل ۱۰۰ نخ سیگار",
            "Stroke": "سابقه سکته مغزی",
            "HeartDiseaseorAttack": "سابقه بیماری قلبی یا حمله قلبی",
            "PhysActivity": "فعالیت بدنی در ۳۰ روز اخیر",
            "Fruits": "مصرف میوه روزانه",
            "Veggies": "مصرف سبزیجات روزانه",
            "HvyAlcoholConsump": "مصرف زیاد الکل",
            "AnyHealthcare": "دسترسی به خدمات درمانی",
            "NoDocbcCost": "عدم مراجعه به پزشک به دلیل هزینه",
            "DiffWalk": "مشکل در راه رفتن"
        }

        binary_features = [
            "HighBP", "HighChol", "CholCheck", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "Fruits", "Veggies", "HvyAlcoholConsump",
            "AnyHealthcare", "NoDocbcCost", "DiffWalk"
        ]

        for i, feature in enumerate(binary_features):
            with cols[i % 2]:
                values[feature] = yes_no_input(
                    binary_labels[feature],
                    f"cdc_{feature}"
                )

        with cols[0]:
            values["BMI"] = st.number_input(
                "شاخص توده بدنی (BMI)",
                min_value=12,
                max_value=90,
                value=27
            )

            values["GenHlth"] = st.selectbox(
                "وضعیت عمومی سلامت",
                [1, 2, 3, 4, 5],
                format_func=lambda x: {
                    1: "عالی",
                    2: "بسیار خوب",
                    3: "خوب",
                    4: "متوسط",
                    5: "ضعیف"
                }[x]
            )

            values["MentHlth"] = st.number_input(
                "تعداد روزهای سلامت روان نامناسب در ماه",
                min_value=0,
                max_value=30,
                value=0
            )

            values["PhysHlth"] = st.number_input(
                "تعداد روزهای سلامت جسمانی نامناسب در ماه",
                min_value=0,
                max_value=30,
                value=0
            )

        with cols[1]:
            values["Sex"] = st.selectbox(
                "جنسیت (کدگذاری دیتاست)",
                [0, 1],
                format_func=lambda x: "زن (۰)" if x == 0 else "مرد (۱)"
            )

            values["Age"] = st.selectbox(
                "گروه سنی",
                list(range(1, 14)),
                format_func=lambda x: f"گروه {x}"
            )

            values["Education"] = st.selectbox(
                "سطح تحصیلات (کد دیتاست)",
                list(range(1, 7))
            )

            values["Income"] = st.selectbox(
                "گروه درآمدی (کد دیتاست)",
                list(range(1, 9))
            )

    else:

        st.markdown("#### اطلاعات فردی و شاخص‌های سلامت")

        col1, col2 = st.columns(2)

        with col1:
            values["gender"] = st.selectbox(
                "جنسیت",
                ["Female", "Male", "Other"]
            )

            values["age"] = st.number_input(
                "سن",
                min_value=1,
                max_value=100,
                value=35
            )

            values["hypertension"] = yes_no_input(
                "فشار خون بالا",
                "kg_hypertension"
            )

            values["heart_disease"] = yes_no_input(
                "سابقه بیماری قلبی",
                "kg_heart"
            )

        with col2:
            values["smoking_history"] = st.selectbox(
                "سابقه مصرف دخانیات",
                [
                    "never",
                    "No Info",
                    "current",
                    "former",
                    "ever",
                    "not current"
                ]
            )

            values["bmi"] = st.number_input(
                "شاخص توده بدنی (BMI)",
                min_value=10.0,
                max_value=80.0,
                value=27.0,
                step=0.1
            )

            if "HbA1c_level" in used_features:
                values["HbA1c_level"] = st.number_input(
                    "مقدار HbA1c",
                    min_value=3.0,
                    max_value=20.0,
                    value=5.5,
                    step=0.1
                )

                values["blood_glucose_level"] = st.number_input(
                    "قند خون (mg/dL)",
                    min_value=50,
                    max_value=500,
                    value=100
                )

    return values


with st.form("patient_form"):

    user_values = make_user_input()

    submitted = st.form_submit_button(
        "🔍 شروع بررسی",
        use_container_width=True
    )

if submitted:

    if "fitted_models" not in st.session_state:
        st.warning(
            "ابتدا مدل‌ها را از بخش پایین صفحه آموزش و ارزیابی کن."
        )

    else:
        model_name = st.session_state.get("best_model_name")

        if st.session_state.get("active_dataset") != dataset_name:
            st.warning("ابتدا مدل‌های همین دیتاست را آموزش بده.")
        elif st.session_state.get("active_mode") != mode:
            st.warning("مدل‌های این سناریو هنوز آموزش داده نشده‌اند.")
        else:
            model = st.session_state["fitted_models"][model_name]

            input_df = pd.DataFrame(
                [user_values],
                columns=used_features
            )

            score = model.predict_proba(input_df)[0, 1]
            prediction = int(score >= 0.5)

            st.divider()
            st.subheader("نتیجه بررسی مدل")

            left, right = st.columns(2)

            with left:
                st.metric(
                    "امتیاز مدل برای کلاس مثبت",
                    f"{score * 100:.1f}%"
                )

            with right:
                st.metric(
                    "خروجی طبقه‌بندی",
                    "کلاس مثبت" if prediction == 1 else "کلاس منفی"
                )

            if prediction == 1:
                st.warning(
                    "مدل این ورودی را در کلاس مثبت قرار داده است. "
                    "برای ارزیابی واقعی وضعیت سلامت باید به پزشک مراجعه و آزمایش معتبر انجام شود."
                )
            else:
                st.success(
                    "مدل این ورودی را در کلاس منفی قرار داده است. "
                    "این نتیجه بیماری را رد نمی‌کند."
                )

            st.caption(
                "عدد نمایش‌داده‌شده امتیاز خروجی مدل است و احتمال پزشکی "
                "کالیبره‌شده یا تشخیص قطعی محسوب نمی‌شود."
            )

# --------------------------------------------------
# 11. بخش آموزش الگوریتم‌ها
# --------------------------------------------------

st.divider()
st.subheader("🧠 آموزش و مقایسه الگوریتم‌ها")

st.write(
    "برای مقایسه منصفانه، تمام الگوریتم‌ها از یک تقسیم آموزش و آزمون استفاده می‌کنند."
)

if st.button(
    "🚀 آموزش و ارزیابی همه الگوریتم‌ها",
    type="primary",
    use_container_width=True
):

    try:
        with st.spinner(
            "در حال آموزش الگوریتم‌ها؛ لطفاً صبر کن..."
        ):

            (
                results,
                fitted_models,
                predictions,
                X_test,
                y_test
            ) = train_and_evaluate(X, y)

        best_name = results.iloc[0]["Algorithm"]

        st.session_state["results"] = results
        st.session_state["fitted_models"] = fitted_models
        st.session_state["predictions"] = predictions
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["best_model_name"] = best_name
        st.session_state["active_dataset"] = dataset_name
        st.session_state["active_mode"] = mode

        st.success(
            f"آموزش مدل‌ها تمام شد. بهترین مدل بر اساس F1: {best_name}"
        )

    except Exception as error:
        st.exception(error)

# --------------------------------------------------
# 12. گزارش کشویی عملکرد الگوریتم‌ها
# --------------------------------------------------

if (
    "results" in st.session_state
    and st.session_state.get("active_dataset") == dataset_name
    and st.session_state.get("active_mode") == mode
):

    results = st.session_state["results"]

    st.divider()

    with st.expander(
        "📊 گزارش علمی عملکرد الگوریتم‌ها | درصد صحت و خطا",
        expanded=True
    ):

        st.markdown("### جدول مقایسه عملکرد")

        display_results = results.copy()

        st.dataframe(
            display_results.style.format({
                col: "{:.2f}%"
                for col in display_results.columns
                if col != "Algorithm"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### درصد صحت و درصد خطا")

        chart_data = results.set_index("Algorithm")[
            ["Accuracy (%)", "Error (%)"]
        ]

        st.bar_chart(chart_data)

        best_row = results.iloc[0]

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "بهترین الگوریتم بر اساس F1",
                best_row["Algorithm"]
            )

        with m2:
            st.metric(
                "درصد صحت",
                f'{best_row["Accuracy (%)"]:.2f}%'
            )

        with m3:
            st.metric(
                "درصد خطا",
                f'{best_row["Error (%)"]:.2f}%'
            )

        with m4:
            st.metric(
                "F1 Score",
                f'{best_row["F1 (%)"]:.2f}%'
            )

        st.markdown("### ماتریس خطا برای الگوریتم منتخب")

        selected_model = st.selectbox(
            "انتخاب الگوریتم برای مشاهده جزئیات",
            list(st.session_state["fitted_models"].keys())
        )

        y_test = st.session_state["y_test"]

        pred = st.session_state["predictions"][
            selected_model
        ]["predicted"]

        cm = confusion_matrix(y_test, pred)

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.imshow(cm, cmap="Blues")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12
                )

        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()

        st.markdown("### تعداد پیش‌بینی‌ها")

        e1, e2, e3, e4 = st.columns(4)

        e1.metric("تشخیص درست مثبت (TP)", f"{tp:,}")
        e2.metric("تشخیص درست منفی (TN)", f"{tn:,}")
        e3.metric("مثبت اشتباه (FP)", f"{fp:,}")
        e4.metric("منفی اشتباه (FN)", f"{fn:,}")

        st.caption(
            "Accuracy = درصد کل پیش‌بینی‌های درست. "
            "Error = درصد کل پیش‌بینی‌های اشتباه. "
            "FP و FN نوع خطا را نشان می‌دهند و در کاربرد پزشکی اهمیت جداگانه دارند."
        )

        st.markdown("### توضیح معیارهای ارزیابی")

        st.write("""
        - **Accuracy:** نسبت کل پیش‌بینی‌های درست به کل نمونه‌های آزمون.
        - **Error Rate:** نسبت کل پیش‌بینی‌های اشتباه به کل نمونه‌های آزمون.
        - **Precision:** از مواردی که مدل مثبت اعلام کرده، چند درصد واقعاً مثبت بوده‌اند.
        - **Recall / Sensitivity:** از موارد مثبت واقعی، چند درصد توسط مدل شناسایی شده‌اند.
        - **F1 Score:** میانگین هماهنگ Precision و Recall.
        - **ROC-AUC:** توانایی مدل در تفکیک دو کلاس در آستانه‌های مختلف.
        - **PR-AUC:** معیار مفید برای بررسی عملکرد در داده‌های نامتوازن.
        """)

# --------------------------------------------------
# 13. محدودیت علمی
# --------------------------------------------------

st.divider()

with st.expander("📚 توضیحات علمی و محدودیت‌های پژوهش"):

    if dataset_name == "CDC":
        st.write("""
        دیتاست CDC از اطلاعات پیمایشی سلامت و سبک زندگی استفاده می‌کند.
        در نسخه دودویی UCI، مقدار مثبت هدف شامل پیش‌دیابت یا دیابت است.
        بنابراین نتیجه این مدل نباید تشخیص قطعی دیابت معرفی شود.
        """)
    else:
        st.write("""
        دیتاست Kaggle شامل متغیرهایی مانند HbA1c و قند خون است.
        ممکن است این متغیرها در ساخت برچسب هدف نقش مستقیم داشته باشند.
        بنابراین سناریوی بدون شاخص‌های آزمایشگاهی نیز در برنامه قرار داده شده
        تا اثر این ویژگی‌ها و احتمال نشت اطلاعات بررسی شود.
        """)

    st.write("""
    نتایج این نسخه بر اساس یک مجموعه آزمون جداشده از داده‌ها گزارش می‌شود.
    برای تبدیل پروژه به مقاله علمی، باید اعتبارسنجی متقاطع،
    تحلیل کالیبراسیون، فاصله اطمینان معیارها، بررسی داده‌های تکراری
    و در صورت امکان اعتبارسنجی خارجی نیز انجام شود.
    """)

st.markdown("""
<div style="text-align:center; padding:25px; color:#a8c6d2;">
سامانه پژوهشی غربالگری دیابت | نسخه دانشگاهی
</div>
""", unsafe_allow_html=True)
