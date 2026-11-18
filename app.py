
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
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
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income"
]

KAGGLE_FEATURES = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "HbA1c_level",
    "blood_glucose_level"
]


# ==================================================
# 2. تنظیمات صفحه
# ==================================================

st.set_page_config(
    page_title="سامانه غربالگری دیابت",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 3. طراحی ظاهری یکپارچه
# ==================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap'
);

:root {
    --primary: #087E8B;
    --primary-dark: #075B66;
    --ink: #18313B;
    --muted: #637780;
    --surface: #FFFFFF;
    --page: #F3F7F8;
    --line: #DCE7E9;
}

html, body, [class*="css"] {
    font-family: 'Vazirmatn', sans-serif !important;
}

.stApp {
    background: var(--page);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: #F3F7F8;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

h1, h2, h3, h4, p, label {
    font-family: 'Vazirmatn', sans-serif !important;
    color: var(--ink);
}

.hero {
    background: linear-gradient(
        110deg,
        #075B66 0%,
        #087E8B 100%
    );
    color: white;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(7,91,102,.12);
}

.hero h1 {
    color: white !important;
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #E5F5F6 !important;
    font-size: 14px;
    margin: 0;
}

.section-title {
    font-size: 19px;
    font-weight: 800;
    color: var(--ink);
    margin: 10px 0 14px 0;
}

.stat-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    min-height: 95px;
    box-shadow: 0 2px 8px rgba(24,49,59,.035);
}

.stat-label {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 5px;
}

.stat-value {
    color: var(--primary-dark);
    font-size: 23px;
    font-weight: 800;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: var(--primary-dark);
    color: white;
    border: none;
}

[data-testid="stSidebar"] {
    background: #EAF1F3;
    border-left: 1px solid var(--line);
}

[data-testid="stForm"],
[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    padding: 12px;
    border-radius: 12px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: var(--line);
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 4. دریافت دیتاست CDC
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
                "ستون Diabetes_binary پیدا نشد."
            )

    data = pd.concat(
        [
            X.reset_index(drop=True),
            y[[CDC_TARGET]].reset_index(drop=True)
        ],
        axis=1
    )

    data.columns = [
        str(c).strip()
        for c in data.columns
    ]

    return data


# ==================================================
# 5. خواندن فایل CSV
# ==================================================

@st.cache_data(show_spinner=False)
def read_csv_cached(path_string, modified_time):

    df = pd.read_csv(path_string)

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


def read_csv(path):

    if not path.exists():
        return None

    return read_csv_cached(
        str(path),
        path.stat().st_mtime
    )


# ==================================================
# 6. تبدیل ستون هدف
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
                "مقادیر ستون هدف نامعتبر هستند."
            )

        return result.astype(int)

    result = pd.to_numeric(
        series,
        errors="coerce"
    )

    if (
        result.isna().any()
        or not set(result.unique()).issubset({0, 1})
    ):
        raise ValueError(
            "ستون هدف باید شامل صفر و یک باشد."
        )

    return result.astype(int)


# ==================================================
# 7. آماده‌سازی داده
# ==================================================

def prepare_data(df, dataset_name, mode):

    if dataset_name == "CDC":

        target = CDC_TARGET
        features = CDC_FEATURES

    else:

        target = KAGGLE_TARGET

        if mode == "غربالگری بدون آزمایش":

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

    missing = [
        c for c in features + [target]
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "ستون‌های لازم پیدا نشدند: "
            + ", ".join(missing)
        )

    data = df[
        features + [target]
    ].dropna(subset=[target]).copy()

    y = convert_target(data[target])

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
            "دیتاست باید هر دو کلاس را داشته باشد."
        )

    if y.value_counts().min() < 2:

        raise ValueError(
            "تعداد نمونه‌های یکی از کلاس‌ها کافی نیست."
        )

    return X, y, features


# ==================================================
# 8. ساخت الگوریتم‌ها
# ==================================================

def build_models(X):

    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric = [
        c for c in X.columns
        if c not in categorical
    ]

    transformers = []

    if numeric:

        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    if categorical:

        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            categorical
        ))

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

    models = {}

    for name, estimator in estimators.items():

        models[name] = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator)
        ])

    try:

        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline([
            ("preprocessing", preprocessor),
            (
                "model",
                XGBClassifier(
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
            )
        ])

    except ImportError:
        pass

    return models


# ==================================================
# 9. آموزش و ارزیابی
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
    fitted = {}
    predictions = {}

    progress = st.progress(0)
    status = st.empty()

    for i, (name, model) in enumerate(models.items()):

        status.caption(
            f"در حال آموزش {name}..."
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

            pred = model.predict(X_test)

            prob = model.predict_proba(
                X_test
            )[:, 1]

            acc = accuracy_score(
                y_test,
                pred
            )

            results.append({

                "Algorithm": name,

                "Accuracy (%)":
                    acc * 100,

                "Error (%)":
                    (1 - acc) * 100,

                "Precision (%)":
                    precision_score(
                        y_test,
                        pred,
                        zero_division=0
                    ) * 100,

                "Recall (%)":
                    recall_score(
                        y_test,
                        pred,
                        zero_division=0
                    ) * 100,

                "F1 (%)":
                    f1_score(
                        y_test,
                        pred,
                        zero_division=0
                    ) * 100,

                "ROC-AUC (%)":
                    roc_auc_score(
                        y_test,
                        prob
                    ) * 100,

                "PR-AUC (%)":
                    average_precision_score(
                        y_test,
                        prob
                    ) * 100,

                "Balanced Accuracy (%)":
                    balanced_accuracy_score(
                        y_test,
                        pred
                    ) * 100
            })

            fitted[name] = model

            predictions[name] = {
                "predicted": pred,
                "probability": prob
            }

        except Exception as exc:

            st.warning(
                f"مدل {name} اجرا نشد: {exc}"
            )

        progress.progress(
            (i + 1) / len(models)
        )

    progress.empty()
    status.empty()

    if not results:

        raise RuntimeError(
            "هیچ الگوریتمی با موفقیت اجرا نشد."
        )

    result_df = pd.DataFrame(
        results
    ).sort_values(
        "F1 (%)",
        ascending=False
    ).reset_index(drop=True)

    return (
        result_df,
        fitted,
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
        تحلیل داده‌های سلامت و مقایسه الگوریتم‌های یادگیری ماشین
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 11. انتخاب دیتاست
# ==================================================

with st.sidebar:

    st.markdown("### تنظیمات تحلیل")

    dataset_name = st.radio(
        "انتخاب دیتاست",
        ["Kaggle", "CDC"]
    )

    mode = "همه ویژگی‌ها"

    if dataset_name == "Kaggle":

        mode = st.radio(
            "نوع تحلیل",
            [
                "غربالگری بدون آزمایش",
                "همه ویژگی‌ها"
            ],
            help="در حالت اول، HbA1c و قند خون وارد مدل نمی‌شوند."
        )

    st.divider()

    st.caption(
        "نسخه پژوهشی و آموزشی"
    )


# ==================================================
# 12. دریافت دیتاست
# ==================================================

if dataset_name == "Kaggle":

    df = read_csv(KAGGLE_FILE)

    if df is None:

        st.error(
            "فایل دیتاست Kaggle پیدا نشد."
        )

        st.info(
            "فایل را داخل پوشه dataset قرار بده."
        )

        st.stop()

else:

    try:

        if CDC_FILE.exists():

            df = read_csv(CDC_FILE)

        else:

            with st.spinner(
                "در حال دریافت دیتاست CDC..."
            ):

                df = load_cdc_from_uci()

    except Exception as exc:

        st.error(
            "دریافت دیتاست CDC ناموفق بود."
        )

        st.code(str(exc))

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

except Exception as exc:

    st.error(
        "آماده‌سازی دیتاست با خطا مواجه شد."
    )

    st.code(str(exc))

    st.stop()


# پاک‌کردن نتایج قدیمی در صورت تغییر دیتاست یا حالت تحلیل
active_key = f"{dataset_name}_{mode}"

if st.session_state.get("active_key") != active_key:

    for key in [
        "results",
        "fitted_models",
        "predictions",
        "X_test",
        "y_test",
        "best_model_name"
    ]:

        st.session_state.pop(
            key,
            None
        )

    st.session_state["active_key"] = active_key


# ==================================================
# 14. خلاصه دیتاست
# ==================================================

st.markdown(
    '<div class="section-title">خلاصه دیتاست</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

stats = [
    ("تعداد نمونه‌ها", f"{len(X):,}"),
    ("تعداد ویژگی‌ها", str(len(used_features))),
    ("نمونه‌های مثبت", f"{int((y == 1).sum()):,}")
]

for col, (label, value) in zip(
    [c1, c2, c3],
    stats
):

    with col:

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

if dataset_name == "CDC":

    st.caption(
        "در دیتاست CDC، کلاس مثبت شامل پیش‌دیابت یا دیابت است."
    )

else:

    st.caption(
        "منشأ داده و نحوه تولید برچسب Kaggle باید در پژوهش بررسی شود."
    )


# ==================================================
# 15. فرم ورود اطلاعات
# ==================================================

st.divider()

st.markdown(
    '<div class="section-title">بررسی اطلاعات فرد</div>',
    unsafe_allow_html=True
)

st.caption(
    "اطلاعات زیر را مطابق ویژگی‌های دیتاست وارد کن."
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

        binary_labels = {

            "HighBP": "فشار خون بالا",
            "HighChol": "کلسترول بالا",
            "CholCheck": "آزمایش کلسترول در ۵ سال اخیر",
            "Smoker": "سابقه مصرف حداقل ۱۰۰ نخ سیگار",
            "Stroke": "سابقه سکته مغزی",
            "HeartDiseaseorAttack": "سابقه بیماری قلبی",
            "PhysActivity": "فعالیت بدنی",
            "Fruits": "مصرف روزانه میوه",
            "Veggies": "مصرف روزانه سبزیجات",
            "HvyAlcoholConsump": "مصرف زیاد الکل",
            "AnyHealthcare": "دسترسی به خدمات درمانی",
            "NoDocbcCost": "عدم مراجعه به پزشک به دلیل هزینه",
            "DiffWalk": "مشکل در راه رفتن"
        }

        left, right = st.columns(2)

        for i, (feature, label) in enumerate(
            binary_labels.items()
        ):

            with (
                left if i % 2 == 0 else right
            ):

                values[feature] = yes_no_input(
                    label,
                    f"cdc_{feature}"
                )

        left, right = st.columns(2)

        with left:

            values["BMI"] = st.number_input(
                "شاخص توده بدنی (BMI)",
                12,
                90,
                27
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
                0,
                30,
                0
            )

            values["PhysHlth"] = st.number_input(
                "روزهای سلامت جسمانی نامناسب در ماه",
                0,
                30,
                0
            )

        with right:

            values["Sex"] = st.selectbox(
                "جنسیت",
                [0, 1],
                format_func=lambda x:
                    "زن" if x == 0 else "مرد"
            )

            values["Age"] = st.selectbox(
                "گروه سنی",
                list(range(1, 14)),
                format_func=lambda x:
                    f"گروه {x}"
            )

            values["Education"] = st.selectbox(
                "سطح تحصیلات",
                list(range(1, 7))
            )

            values["Income"] = st.selectbox(
                "گروه درآمدی",
                list(range(1, 9))
            )

    else:

        left, right = st.columns(2)

        with left:

            values["gender"] = st.selectbox(
                "جنسیت",
                ["Female", "Male", "Other"]
            )

            values["age"] = st.number_input(
                "سن",
                1,
                100,
                35
            )

            values["hypertension"] = yes_no_input(
                "فشار خون بالا",
                "kg_hypertension"
            )

            values["heart_disease"] = yes_no_input(
                "سابقه بیماری قلبی",
                "kg_heart"
            )

        with right:

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
                10.0,
                80.0,
                27.0,
                step=0.1
            )

            if mode == "همه ویژگی‌ها":

                values["HbA1c_level"] = st.number_input(
                    "مقدار HbA1c",
                    3.0,
                    20.0,
                    5.5,
                    step=0.1
                )

                values["blood_glucose_level"] = st.number_input(
                    "قند خون (mg/dL)",
                    50,
                    500,
                    100
                )

    return values


with st.form("patient_form"):

    user_values = make_user_input()

    submitted = st.form_submit_button(
        "بررسی اطلاعات",
        use_container_width=True
    )


# ==================================================
# 16. نتیجه پیش‌بینی
# ==================================================

if submitted:

    if "fitted_models" not in st.session_state:

        st.warning(
            "ابتدا مدل‌ها را در بخش آموزش الگوریتم‌ها آموزش بده."
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

        prediction = int(score >= 0.5)

        st.divider()

        st.markdown(
            '<div class="section-title">نتیجه بررسی مدل</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "امتیاز مدل برای کلاس مثبت",
            f"{score * 100:.1f}%"
        )

        c2.metric(
            "خروجی مدل",
            "کلاس مثبت" if prediction else "کلاس منفی"
        )

        if prediction:

            st.warning(
                "مدل این ورودی را در کلاس مثبت قرار داده است. "
                "برای ارزیابی وضعیت سلامت، آزمایش معتبر و نظر پزشک ضروری است."
            )

        else:

            st.success(
                "مدل این ورودی را در کلاس منفی قرار داده است؛ "
                "این نتیجه بیماری را رد نمی‌کند."
            )

        st.caption(
            "این عدد امتیاز خام مدل است، نه احتمال پزشکی کالیبره‌شده."
        )


# ==================================================
# 17. آموزش الگوریتم‌ها
# ==================================================

st.divider()

st.markdown(
    '<div class="section-title">آموزش و مقایسه الگوریتم‌ها</div>',
    unsafe_allow_html=True
)

st.caption(
    "تمام الگوریتم‌ها با مجموعه آموزش و آزمون یکسان ارزیابی می‌شوند."
)

if st.button(
    "آموزش و ارزیابی الگوریتم‌ها",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "در حال آموزش مدل‌ها؛ لطفاً منتظر بمان..."
        ):

            (
                results,
                fitted_models,
                predictions,
                X_test,
                y_test
            ) = train_and_evaluate(X, y)

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

        st.success(
            f"آموزش کامل شد. مدل برتر: {best_name}"
        )

    except Exception as exc:

        st.error(
            "آموزش الگوریتم‌ها با خطا مواجه شد."
        )

        st.exception(exc)


# ==================================================
# 18. نمایش نتایج
# ==================================================

if (
    "results" in st.session_state
    and st.session_state.get("active_key") == active_key
):

    results = st.session_state["results"]

    st.divider()

    st.markdown(
        '<div class="section-title">نتایج ارزیابی مدل‌ها</div>',
        unsafe_allow_html=True
    )

    best = results.iloc[0]

    a, b, c = st.columns(3)

    a.metric(
        "مدل برتر",
        best["Algorithm"]
    )

    b.metric(
        "درصد صحت",
        f'{best["Accuracy (%)"]:.2f}%'
    )

    c.metric(
        "امتیاز F1",
        f'{best["F1 (%)"]:.2f}%'
    )

    display_results = results.rename(
        columns={
            "Algorithm": "الگوریتم",
            "Accuracy (%)": "صحت",
            "Error (%)": "خطا",
            "Precision (%)": "Precision",
            "Recall (%)": "Recall",
            "F1 (%)": "F1",
            "ROC-AUC (%)": "ROC-AUC",
            "PR-AUC (%)": "PR-AUC",
            "Balanced Accuracy (%)": "صحت متوازن"
        }
    )

    st.dataframe(
        display_results.style.format({
            col: "{:.2f}%"
            for col in display_results.columns
            if col != "الگوریتم"
        }),
        use_container_width=True,
        hide_index=True
    )

    with st.expander(
        "نمودار مقایسه صحت و خطا"
    ):

        chart = results.set_index(
            "Algorithm"
        )[[
            "Accuracy (%)",
            "Error (%)"
        ]]

        st.bar_chart(chart)

    with st.expander(
        "جزئیات خطاهای مدل"
    ):

        selected = st.selectbox(
            "انتخاب الگوریتم",
            list(
                st.session_state[
                    "fitted_models"
                ].keys()
            )
        )

        y_test = st.session_state["y_test"]

        predicted = st.session_state[
            "predictions"
        ][selected]["predicted"]

        cm = confusion_matrix(
            y_test,
            predicted,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        fig, ax = plt.subplots(
            figsize=(5.5, 3.6)
        )

        ax.imshow(
            cm,
            cmap="Blues"
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticks(
            [0, 1],
            ["Negative", "Positive"]
        )

        ax.set_yticks(
            [0, 1],
            ["Negative", "Positive"]
        )

        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="black"
                )

        st.pyplot(fig)

        plt.close(fig)

        q1, q2, q3, q4 = st.columns(4)

        q1.metric("مثبت درست (TP)", f"{tp:,}")
        q2.metric("منفی درست (TN)", f"{tn:,}")
        q3.metric("مثبت اشتباه (FP)", f"{fp:,}")
        q4.metric("منفی اشتباه (FN)", f"{fn:,}")

    with st.expander(
        "راهنمای معیارهای ارزیابی"
    ):

        st.markdown("""
        - **Accuracy:** درصد کل پیش‌بینی‌های درست.
        - **Error:** درصد کل پیش‌بینی‌های نادرست.
        - **Precision:** دقت پیش‌بینی‌های مثبت.
        - **Recall:** میزان شناسایی نمونه‌های مثبت واقعی.
        - **F1:** توازن میان Precision و Recall.
        - **ROC-AUC:** توان تفکیک دو کلاس.
        - **PR-AUC:** معیار مهم برای داده‌های نامتوازن.
        - **Balanced Accuracy:** میانگین عملکرد در دو کلاس.
        """)


# ==================================================
# 19. توضیحات پژوهشی
# ==================================================

with st.expander(
    "درباره دیتاست و محدودیت‌های پژوهش"
):

    if dataset_name == "CDC":

        st.write(
            "در نسخه دودویی دیتاست CDC، کلاس مثبت شامل پیش‌دیابت یا دیابت است؛ "
            "بنابراین خروجی مدل نباید تشخیص قطعی دیابت معرفی شود."
        )

    else:

        st.write(
            "دیتاست Kaggle شامل متغیرهای بالینی و آزمایشگاهی است. "
            "منشأ داده و نحوه تولید برچسب باید پیش از نتیجه‌گیری علمی بررسی شود."
        )

    st.write("""
    این نسخه یک نمونه اولیه پژوهشی است.

    برای تبدیل آن به پژوهش قابل انتشار، مراحل زیر نیز لازم هستند:

    - اعتبارسنجی متقاطع
    - محاسبه فاصله اطمینان
    - بررسی کالیبراسیون مدل
    - تحلیل اهمیت ویژگی‌ها با SHAP
    - بررسی داده‌های تکراری و نشت اطلاعات
    - اعتبارسنجی خارجی روی داده مستقل
    """)


# ==================================================
# 20. پانوشت
# ==================================================

st.markdown("""
<div style="
    text-align:center;
    color:#71858C;
    padding:25px 0 5px 0;
    font-size:12px;
">
سامانه هوشمند غربالگری دیابت
<br>
پروژه پژوهشی یادگیری ماشین
</div>
""", unsafe_allow_html=True)
