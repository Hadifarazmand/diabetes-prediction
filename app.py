
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.set_page_config(page_title="پیش‌بینی دیابت", page_icon="🩺", layout="wide")
st.title("🩺 سامانه پژوهشی پیش‌بینی دیابت")
st.caption("مقایسه الگوریتم‌های یادگیری ماشین")

st.warning("این سامانه آموزشی و پژوهشی است و جایگزین تشخیص پزشک نیست.")

DATA_PATH = Path(__file__).parent / "dataset" / "diabetes.csv"

FEATURES = [
    "Age", "Gender", "Polyuria", "Polydipsia",
    "sudden weight loss", "weakness", "Polyphagia",
    "visual blurring", "Itching", "delayed healing"
]

CATEGORICAL = FEATURES[1:]

@st.cache_resource
def train_models():
    df = pd.read_csv(DATA_PATH)

    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip().str.title()

    df["class"] = df["class"].astype(str).str.strip().str.title()

    X = df[FEATURES].copy()
    X["Age"] = pd.to_numeric(X["Age"], errors="coerce")
    y = df["class"]

    valid = X["Age"].notna() & y.isin(["Positive", "Negative"])
    X, y = X.loc[valid], y.loc[valid]

    y = y.map({"Negative": 0, "Positive": 1})

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("age", StandardScaler(), ["Age"])
    ])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = {}
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", classifier)
        ])

        scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )

        results[name] = {
            metric: scores["test_" + metric].mean()
            for metric in scoring
        }

    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])

    final_model.fit(X, y)

    return results, final_model

if not DATA_PATH.exists():
    st.error("فایل dataset/diabetes.csv پیدا نشد.")
    st.stop()

try:
    results, model = train_models()
except Exception as e:
    st.error(f"خطا در آموزش مدل: {e}")
    st.stop()

st.subheader("نتایج مقایسه الگوریتم‌ها")

result_df = pd.DataFrame(results).T * 100
result_df.columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
st.dataframe(result_df.round(2), use_container_width=True)

best_model = result_df["F1-score"].idxmax()
st.success(f"بهترین الگوریتم بر اساس F1-score: {best_model}")

st.info("مدل نهایی فعلاً Random Forest است؛ انتخاب خودکار بهترین مدل در مرحله بعد اضافه می‌شود.")
