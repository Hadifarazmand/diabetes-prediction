
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="پیش‌بینی دیابت", page_icon="🩺", layout="centered")

st.title("🩺 سامانه پیش‌بینی اولیه دیابت")
st.write("اطلاعات زیر را وارد کنید تا مدل، نتیجه تخمینی خود را نمایش دهد.")
st.warning("این ابزار صرفاً آموزشی است و جایگزین تشخیص پزشک نیست.")

DATA_PATH = Path(__file__).parent / "dataset" / "diabetes.csv"

FEATURES = [
    "Age", "Gender", "Polyuria", "Polydipsia",
    "sudden weight loss", "weakness", "Polyphagia",
    "visual blurring", "Itching", "delayed healing"
]
CATEGORICAL = FEATURES[1:]

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip().str.title()

    X = df[FEATURES].copy()
    X["Age"] = pd.to_numeric(X["Age"], errors="coerce")
    df["class"] = df["class"].astype(str).str.strip().str.title()
    valid = X["Age"].notna() & df["class"].isin(["Positive", "Negative"])
    X = X.loc[valid]
    y = df.loc[valid, "class"]

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("age", "passthrough", ["Age"])
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    model.fit(X, y)
    return model, accuracy

if not DATA_PATH.exists():
    st.error("فایل دیتاست پیدا نشد. باید dataset/diabetes.csv را به مخزن اضافه کنید.")
    st.stop()

model, accuracy = train_model()

st.caption(f"دقت مدل روی داده‌های آزمون: {accuracy * 100:.1f}%")

with st.form("diabetes_form"):
    age = st.number_input("سن", min_value=1, max_value=120, value=35)
    gender = st.selectbox("جنسیت", ["Male", "Female"], format_func=lambda x: "مرد" if x == "Male" else "زن")

    labels = {
        "Polyuria": "ادرار بیش از حد",
        "Polydipsia": "تشنگی بیش از حد",
        "sudden weight loss": "کاهش وزن ناگهانی",
        "weakness": "ضعف عمومی",
        "Polyphagia": "افزایش اشتها",
        "visual blurring": "تاری دید",
        "Itching": "خارش",
        "delayed healing": "تأخیر در بهبود زخم"
    }

    answers = {}
    for feature, label in labels.items():
        answers[feature] = st.radio(
            label, ["No", "Yes"],
            format_func=lambda x: "خیر" if x == "No" else "بله",
            horizontal=True
        )

    submitted = st.form_submit_button("بررسی اطلاعات")

if submitted:
    row = {"Age": age, "Gender": gender, **answers}
    prediction_input = pd.DataFrame([row], columns=FEATURES)
    probabilities = model.predict_proba(prediction_input)[0]
    classes = list(model.named_steps["classifier"].classes_)
    positive_probability = float(probabilities[classes.index("Positive")])
    prediction = model.predict(prediction_input)[0]

    st.subheader("نتیجه مدل")
    if prediction == "Positive":
        st.error("مدل، الگوی مشابه موارد مثبت در دیتاست شناسایی کرده است.")
    else:
        st.success("مدل، الگوی مشابه موارد منفی در دیتاست شناسایی کرده است.")

    st.metric("احتمال تخمینی خروجی مثبت مدل", f"{positive_probability * 100:.1f}%")
    st.progress(positive_probability)
    st.caption("این درصد، احتمال خروجی مدل است و به معنی احتمال واقعی ابتلا به دیابت نیست.")

