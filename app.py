
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
    GradientBoostingClassifier
)

warnings.filterwarnings("ignore")


# ==================================================
# 1. تنظیمات اصلی
# ==================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"

CDC_FILE = DATA_DIR / "cdc_diabetes_health_indicators.csv"
KAGGLE_FILE = DATA_DIR / "diabetes_prediction_dataset.csv"

CDC_TARGET = "Diabetes_binary"
KAGGLE_TARGET = "diabetes"

CDC_FEATURES = [
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

KAGGLE_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]


# ==================================================
# 2. تنظیمات صفحه
# ==================================================

st.set_page_config(
    page_title="سامانه هوشمند غربالگری دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 3. طراحی ظاهری
# ==================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif;
}

.stApp {
    background:
    linear-gradient(
        135deg,
        #071827 0%,
        #102b40 55%,
        #123d4b 100%
    );

    color: #f5f8fc;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, label {
    font-family: 'Vazirmatn', sans-serif !important;
}

.hero {
    background:
    linear-gradient(
        120deg,
        #0b4051,
        #087f8c
    );

    padding: 35px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.12);

    margin-bottom: 25px;

    box-shadow:
    0 12px 35px rgba(0,0,0,.20);
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
    background: rgba(255,255,255,.075);
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

[data-testid="stSidebar"] {
    background: #091e30;
}

.stAlert {
    border-radius: 14px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. دریافت خودکار دیتاست CDC
# ==================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_cdc_from_uci():

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=891)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.Series):
        y = y.to_frame()

    if CDC_TARGET not in y.columns:

        if len(y.columns) == 1:
            y.columns = [CDC_TARGET]

        else:
            raise ValueError(
                "ستون Diabetes_binary در دیتاست پیدا نشد."
            )

    data = pd.concat(
        [
            X.reset_index(drop=True),
            y[[CDC_TARGET]].reset_index(drop=True)
        ],
        axis=1
    )

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    return data


# ==================================================
# 5. خواندن فایل CSV
# ==================================================

def read_csv(path):

    if not path.exists():
        return None

    df = pd.read_csv(path)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ==================================================
# 6. تبدیل ستون هدف به صفر و یک
# ==================================================

def convert_target(series):

    if series.dtype == "object":

        values = (
            series.astype(str)
            .str.strip()
            .str.lower()
        )

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
            raise ValueError(
                "مقادیر ستون هدف قابل تبدیل به صفر و یک نیستند."
            )

        return result.astype(int)

    result = pd.to_numeric(
        series,
        errors="coerce"
    )

    if result.isna().any():
        raise ValueError(
            "ستون هدف دارای مقادیر نامعتبر است."
        )

    if not set(result.unique()).issubset({0, 1}):
        raise ValueError(
            "ستون هدف باید فقط شامل صفر و یک باشد."
        )

    return result.astype(int)


# ==================================================
# 7. آماده‌سازی دیتاست
# ==================================================

def prepare_data(df, dataset_name, mode):

    if dataset_name == "CDC":

        target = CDC_TARGET
        features = CDC_FEATURES

    else:

        target = KAGGLE_TARGET

        if mode == "غربالگری اولیه بدون آزمایش":

            features = [
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "smoking_history",
                "bmi"
            ]

        else:

            features = KAGGLE_FEATURES

    required_columns = features + [target]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "ستون‌های زیر در دیتاست وجود ندارند: "
            + ", ".join(missing_columns)
        )

    data = df[required_columns].copy()

    data = data.dropna(
        subset=[target]
    )

    y = convert_target(
        data[target]
    )

    X = data[features].copy()

    for col in X.columns:

        if X[col].dtype == "object":

            X[col] = (
                X[col]
                .astype(str)
                .str.strip()
            )

        else:

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

    if y.nunique() != 2:
        raise ValueError(
            "دیتاست باید دارای هر دو کلاس صفر و یک باشد."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "تعداد نمونه‌های یکی از کلاس‌ها برای تقسیم آموزش و آزمون کافی نیست."
        )

    return X, y, features


# ==================================================
# 8. ساخت الگوریتم‌های یادگیری ماشین
# ==================================================

def build_models(X):

    categorical_cols = (
        X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    )

    numeric_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    transformers = []

    if numeric_cols:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_cols
            )
        )

    if categorical_cols:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_cols
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    estimators = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=100,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=3,
                random_state=42
            )
    }

    final_models = {}

    for name, estimator in estimators.items():

        final_models[name] = Pipeline([
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                estimator
            )
        ])

    # XGBoost به صورت اختیاری
    try:

        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=150,
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
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                xgb_model
            )
        ])

    except ImportError:
        pass

    return final_models


# ==================================================
# 9. آموزش و ارزیابی الگوریتم‌ها
# ==================================================

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

        status.write(
            f"در حال آموزش الگوریتم {name} ..."
        )

        try:

            if name == "XGBoost":

                from sklearn.utils.class_weight import (
                    compute_sample_weight
                )

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

                model.fit(
                    X_train,
                    y_train
                )

            predicted = model.predict(
                X_test
            )

            probability = model.predict_proba(
                X_test
            )[:, 1]

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            error_rate = 1 - accuracy

            precision = precision_score(
                y_test,
                predicted,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predicted,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predicted,
                zero_division=0
            )

            roc_auc = roc_auc_score(
                y_test,
                probability
            )

            pr_auc = average_precision_score(
                y_test,
                probability
            )

            balanced_acc = balanced_accuracy_score(
                y_test,
                predicted
            )

            results.append({

                "Algorithm": name,

                "Accuracy (%)":
                    accuracy * 100,

                "Error (%)":
                    error_rate * 100,

                "Precision (%)":
                    precision * 100,

                "Recall (%)":
                    recall * 100,

                "F1 (%)":
                    f1 * 100,

                "ROC-AUC (%)":
                    roc_auc * 100,

                "PR-AUC (%)":
                    pr_auc * 100,

                "Balanced Accuracy (%)":
                    balanced_acc * 100
            })

            fitted_models[name] = model

            predictions[name] = {

                "predicted": predicted,

                "probability": probability
            }

        except Exception as error:

            st.warning(
                f"اجرای {name} با خطا مواجه شد: {error}"
            )

        progress.progress(
            (index + 1) / total
        )

    progress.empty()
    status.empty()

    if not results:

        raise RuntimeError(
            "هیچ الگوریتمی با موفقیت اجرا نشد."
        )

    result_df = pd.DataFrame(results)

    # انتخاب مدل بر اساس F1
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


# ==================================================
# 10. سربرگ
# ==================================================

st.markdown("""
<div class="hero">

    <h1>🩺 سامانه هوشمند غربالگری دیابت</h1>

    <p>
        تحلیل اطلاعات سلامت با استفاده از الگوریتم‌های
        یادگیری ماشین و مقایسه علمی عملکرد مدل‌ها
    </p>

</div>
""", unsafe_allow_html=True)

st.info(
    "این سامانه یک پروژه آموزشی و پژوهشی است. "
    "نتایج آن جایگزین آزمایش پزشکی یا تشخیص پزشک نیست."
)


# ==================================================
# 11. انتخاب دیتاست
# ==================================================

st.sidebar.title("⚙️ تنظیمات سامانه")

dataset_name = st.sidebar.radio(
    "انتخاب دیتاست",
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


# ==================================================
# 12. دریافت دیتاست انتخاب‌شده
# ==================================================

if dataset_name == "CDC":

    try:

        if CDC_FILE.exists():

            df = read_csv(
                CDC_FILE
            )

            st.sidebar.success(
                "دیتاست CDC از فایل محلی بارگذاری شد."
            )

        else:

            with st.spinner(
                "در حال دریافت خودکار دیتاست CDC از UCI ..."
            ):

                df = load_cdc_from_uci()

            st.sidebar.success(
                "دیتاست CDC دریافت شد."
            )

    except Exception as error:

        st.error(
            "دریافت دیتاست CDC ناموفق بود."
        )

        st.warning(
            "اتصال اینترنت سرور یا دسترسی به مخزن UCI را بررسی کن."
        )

        st.code(
            str(error)
        )

        st.stop()

else:

    df = read_csv(
        KAGGLE_FILE
    )

    if df is None:

        st.error(
            "فایل دیتاست Kaggle پیدا نشد."
        )

        st.info(
            "فایل باید با نام diabetes_prediction_dataset.csv "
            "داخل پوشه dataset قرار بگیرد."
        )

        st.stop()


# ==================================================
# 13. آماده‌سازی داده
# ==================================================

try:

    X, y, used_features = prepare_data(
        df,
        dataset_name,
        mode
    )

except Exception as error:

    st.error(
        "خطا در آماده‌سازی اطلاعات دیتاست:"
    )

    st.code(
        str(error)
    )

    st.stop()


# ==================================================
# 14. نمایش آمار دیتاست
# ==================================================

st.subheader(
    "📊 مشخصات دیتاست"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <h3>تعداد کل رکوردها</h3>
            <h2>{len(X):,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <h3>تعداد ویژگی‌ها</h3>
            <h2>{len(used_features)}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <h3>نمونه‌های کلاس مثبت</h3>
            <h2>{int((y == 1).sum()):,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <h3>نمونه‌های کلاس منفی</h3>
            <h2>{int((y == 0).sum()):,}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


if dataset_name == "CDC":

    st.caption(
        "در نسخه دودویی این دیتاست، کلاس مثبت شامل افراد دارای "
        "پیش‌دیابت یا دیابت است."
    )

else:

    st.caption(
        "منشأ داده و نحوه تولید برچسب در دیتاست Kaggle باید "
        "پیش از نتیجه‌گیری علمی بررسی شود."
    )


# ==================================================
# 15. فرم ورود اطلاعات
# ==================================================

st.divider()

st.subheader(
    "👤 ورود اطلاعات فرد"
)

st.write(
    "اطلاعات زیر را مطابق ویژگی‌های دیتاست انتخاب‌شده وارد کن."
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

        st.markdown(
            "#### اطلاعات سلامت و سبک زندگی"
        )

        col1, col2 = st.columns(2)

        binary_labels = {

            "HighBP": "فشار خون بالا",

            "HighChol": "کلسترول بالا",

            "CholCheck": "آزمایش کلسترول در ۵ سال اخیر",

            "Smoker": "سابقه مصرف حداقل ۱۰۰ نخ سیگار",

            "Stroke": "سابقه سکته مغزی",

            "HeartDiseaseorAttack": "سابقه بیماری قلبی یا حمله قلبی",

            "PhysActivity": "فعالیت بدنی در ۳۰ روز اخیر",

            "Fruits": "مصرف روزانه میوه",

            "Veggies": "مصرف روزانه سبزیجات",

            "HvyAlcoholConsump": "مصرف زیاد الکل",

            "AnyHealthcare": "دسترسی به خدمات درمانی",

            "NoDocbcCost": "عدم مراجعه به پزشک به دلیل هزینه",

            "DiffWalk": "مشکل در راه رفتن"
        }

        binary_features = list(
            binary_labels.keys()
        )

        for index, feature in enumerate(binary_features):

            with (
                col1 if index % 2 == 0 else col2
            ):

                values[feature] = yes_no_input(
                    binary_labels[feature],
                    f"cdc_{feature}"
                )

        with col1:

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
                "روزهای سلامت روان نامناسب در ماه",
                min_value=0,
                max_value=30,
                value=0
            )

            values["PhysHlth"] = st.number_input(
                "روزهای سلامت جسمانی نامناسب در ماه",
                min_value=0,
                max_value=30,
                value=0
            )

        with col2:

            values["Sex"] = st.selectbox(
                "جنسیت (کدگذاری دیتاست)",
                [0, 1],
                format_func=lambda x:
                    "زن (۰)" if x == 0 else "مرد (۱)"
            )

            values["Age"] = st.selectbox(
                "گروه سنی",
                list(range(1, 14)),
                format_func=lambda x:
                    f"گروه سنی {x}"
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

        st.markdown(
            "#### اطلاعات فردی و شاخص‌های سلامت"
        )

        col1, col2 = st.columns(2)

        with col1:

            values["gender"] = st.selectbox(
                "جنسیت",
                [
                    "Female",
                    "Male",
                    "Other"
                ]
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
        "🔍 بررسی اطلاعات",
        use_container_width=True
    )


# ==================================================
# 16. پیش‌بینی برای کاربر
# ==================================================

if submitted:

    if "fitted_models" not in st.session_state:

        st.warning(
            "ابتدا از بخش آموزش الگوریتم‌ها، مدل‌ها را آموزش بده."
        )

    elif (
        st.session_state.get("active_dataset") != dataset_name
        or st.session_state.get("active_mode") != mode
    ):

        st.warning(
            "مدل‌های دیتاست و سناریوی انتخاب‌شده هنوز آموزش داده نشده‌اند."
        )

    else:

        best_name = st.session_state[
            "best_model_name"
        ]

        model = st.session_state[
            "fitted_models"
        ][best_name]

        input_df = pd.DataFrame(
            [user_values],
            columns=used_features
        )

        score = model.predict_proba(
            input_df
        )[0, 1]

        prediction = int(
            score >= 0.5
        )

        st.divider()

        st.subheader(
            "📋 نتیجه بررسی مدل"
        )

        left, right = st.columns(2)

        with left:

            st.metric(
                "امتیاز مدل برای کلاس مثبت",
                f"{score * 100:.2f}%"
            )

        with right:

            st.metric(
                "خروجی طبقه‌بندی",
                "کلاس مثبت" if prediction == 1 else "کلاس منفی"
            )

        if prediction == 1:

            st.warning(
                "مدل این ورودی را در کلاس مثبت قرار داده است. "
                "برای ارزیابی وضعیت سلامت باید آزمایش معتبر و "
                "نظر پزشک انجام شود."
            )

        else:

            st.success(
                "مدل این ورودی را در کلاس منفی قرار داده است. "
                "این نتیجه بیماری را رد نمی‌کند."
            )

        st.caption(
            "این عدد امتیاز خروجی مدل است، نه احتمال پزشکی "
            "کالیبره‌شده و نه تشخیص قطعی."
        )


# ==================================================
# 17. آموزش الگوریتم‌ها
# ==================================================

st.divider()

st.subheader(
    "🧠 آموزش و مقایسه الگوریتم‌ها"
)

st.write(
    "تمام الگوریتم‌ها با تقسیم آموزش و آزمون یکسان ارزیابی می‌شوند."
)

if st.button(
    "🚀 آموزش و ارزیابی همه الگوریتم‌ها",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "در حال آموزش الگوریتم‌ها؛ لطفاً منتظر بمان..."
        ):

            (
                results,
                fitted_models,
                predictions,
                X_test,
                y_test
            ) = train_and_evaluate(
                X,
                y
            )

        best_name = results.iloc[0][
            "Algorithm"
        ]

        st.session_state["results"] = results

        st.session_state["fitted_models"] = (
            fitted_models
        )

        st.session_state["predictions"] = (
            predictions
        )

        st.session_state["X_test"] = X_test

        st.session_state["y_test"] = y_test

        st.session_state["best_model_name"] = (
            best_name
        )

        st.session_state["active_dataset"] = (
            dataset_name
        )

        st.session_state["active_mode"] = mode

        st.success(
            f"آموزش با موفقیت انجام شد. "
            f"مدل برتر بر اساس F1: {best_name}"
        )

    except Exception as error:

        st.error(
            "در زمان آموزش الگوریتم‌ها خطایی رخ داد."
        )

        st.exception(
            error
        )


# ==================================================
# 18. گزارش علمی کشویی
# ==================================================

if (
    "results" in st.session_state
    and st.session_state.get("active_dataset") == dataset_name
    and st.session_state.get("active_mode") == mode
):

    results = st.session_state[
        "results"
    ]

    st.divider()

    with st.expander(
        "📊 گزارش علمی عملکرد الگوریتم‌ها | درصد صحت و خطا",
        expanded=True
    ):

        st.markdown(
            "### جدول مقایسه الگوریتم‌ها"
        )

        st.caption(
            "تمام معیارها روی مجموعه آزمون مشترک محاسبه شده‌اند."
        )

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

        st.markdown(
            "### نمودار درصد صحت و خطا"
        )

        chart_data = results.set_index(
            "Algorithm"
        )[
            [
                "Accuracy (%)",
                "Error (%)"
            ]
        ]

        st.bar_chart(
            chart_data
        )

        best_row = results.iloc[0]

        st.markdown(
            "### خلاصه عملکرد مدل منتخب"
        )

        a, b, c, d = st.columns(4)

        with a:

            st.metric(
                "مدل منتخب",
                best_row["Algorithm"]
            )

        with b:

            st.metric(
                "درصد صحت",
                f'{best_row["Accuracy (%)"]:.2f}%'
            )

        with c:

            st.metric(
                "درصد خطا",
                f'{best_row["Error (%)"]:.2f}%'
            )

        with d:

            st.metric(
                "F1 Score",
                f'{best_row["F1 (%)"]:.2f}%'
            )

        st.divider()

        st.markdown(
            "### بررسی جزئیات هر الگوریتم"
        )

        selected_model = st.selectbox(
            "الگوریتم موردنظر را انتخاب کن",
            list(
                st.session_state[
                    "fitted_models"
                ].keys()
            )
        )

        y_test = st.session_state[
            "y_test"
        ]

        predicted = st.session_state[
            "predictions"
        ][selected_model]["predicted"]

        cm = confusion_matrix(
            y_test,
            predicted,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.imshow(
            cm,
            cmap="Blues"
        )

        ax.set_title(
            "Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted Label"
        )

        ax.set_ylabel(
            "Actual Label"
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels([
            "Negative",
            "Positive"
        ])

        ax.set_yticklabels([
            "Negative",
            "Positive"
        ])

        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=13
                )

        st.pyplot(fig)

        plt.close(fig)

        st.markdown(
            "### جزئیات پیش‌بینی‌های درست و اشتباه"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "مثبت درست (TP)",
            f"{tp:,}"
        )

        c2.metric(
            "منفی درست (TN)",
            f"{tn:,}"
        )

        c3.metric(
            "مثبت اشتباه (FP)",
            f"{fp:,}"
        )

        c4.metric(
            "منفی اشتباه (FN)",
            f"{fn:,}"
        )

        st.divider()

        st.markdown(
            "### تعریف معیارهای ارزیابی"
        )

        st.write("""
        **Accuracy:** درصد کل نمونه‌هایی که درست طبقه‌بندی شده‌اند.

        **Error Rate:** درصد کل نمونه‌هایی که اشتباه طبقه‌بندی شده‌اند.

        **Precision:** از مواردی که مدل مثبت اعلام کرده، چه تعداد واقعاً مثبت بوده‌اند.

        **Recall / Sensitivity:** از تمام نمونه‌های مثبت واقعی، چه تعداد توسط مدل شناسایی شده‌اند.

        **F1 Score:** میانگین هماهنگ Precision و Recall.

        **ROC-AUC:** توانایی مدل در تفکیک دو کلاس در آستانه‌های مختلف.

        **PR-AUC:** معیار مهم برای ارزیابی مدل در داده‌های نامتوازن.

        **Balanced Accuracy:** میانگین حساسیت دو کلاس.
        """)

        st.caption(
            "درصد خطا برابر با ۱۰۰ منهای درصد صحت است. "
            "FP و FN نوع خطا را مشخص می‌کنند و در کاربرد پزشکی اهمیت ویژه دارند."
        )


# ==================================================
# 19. توضیحات پژوهشی
# ==================================================

st.divider()

with st.expander(
    "📚 مبانی علمی و محدودیت‌های پژوهش"
):

    st.markdown(
        "### الگوریتم‌های استفاده‌شده"
    )

    st.write("""
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Extra Trees
    5. Gradient Boosting
    6. XGBoost (در صورت نصب موفق)
    """)

    st.markdown(
        "### روش ارزیابی"
    )

    st.write("""
    داده‌ها با روش Stratified Train-Test Split به دو بخش تقسیم می‌شوند:

    - ۸۰ درصد برای آموزش مدل‌ها
    - ۲۰ درصد برای آزمون مدل‌ها

    تمام الگوریتم‌ها از مجموعه آزمون یکسان استفاده می‌کنند.
    """)

    if dataset_name == "CDC":

        st.markdown(
            "### توضیح دیتاست CDC"
        )

        st.write("""
        این دیتاست بر اساس اطلاعات پیمایشی سلامت و سبک زندگی تهیه شده است.
        در نسخه دودویی مورد استفاده، کلاس مثبت شامل پیش‌دیابت یا دیابت است.
        بنابراین نباید خروجی مدل را تشخیص قطعی دیابت معرفی کرد.
        """)

    else:

        st.markdown(
            "### توضیح دیتاست Kaggle"
        )

        st.write("""
        این دیتاست شامل متغیرهایی مانند HbA1c و قند خون است.
        باید منشأ داده‌ها و نحوه تولید برچسب بررسی شود.

        همچنین مقایسه حالت بدون متغیرهای آزمایشگاهی با حالت دارای
        این متغیرها برای بررسی احتمال نشت اطلاعات اهمیت دارد.
        """)

    st.markdown(
        "### محدودیت‌های نسخه فعلی"
    )

    st.write("""
    این نسخه یک چارچوب اولیه برای مقایسه الگوریتم‌هاست.

    برای استفاده در مقاله علمی، مراحل زیر نیز ضروری هستند:

    - اعتبارسنجی متقاطع ۵ یا ۱۰ بخشی
    - محاسبه فاصله اطمینان معیارهای اصلی
    - بررسی کالیبراسیون مدل‌ها
    - تحلیل اهمیت ویژگی‌ها با روش‌هایی مانند SHAP
    - بررسی رکوردهای تکراری و نشت اطلاعات
    - اعتبارسنجی خارجی روی داده مستقل
    - تحلیل عملکرد مدل در گروه‌های مختلف جمعیتی
    """)


# ==================================================
# 20. پانوشت
# ==================================================

st.markdown("""
<div style="
    text-align:center;
    padding:30px;
    color:#a8c6d2;
    font-size:13px;
">

    سامانه هوشمند غربالگری دیابت

    <br>

    پروژه پژوهشی یادگیری ماشین

</div>
""", unsafe_allow_html=True)
