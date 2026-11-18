
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

# مسیر اصلی پروژه
ROOT = Path(__file__).resolve().parents[1]

# مسیر ذخیره دیتاست
DATA_DIR = ROOT / "dataset"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "cdc_diabetes_health_indicators.csv"

print("Downloading CDC Diabetes Health Indicators...")

# دریافت دیتاست از UCI
dataset = fetch_ucirepo(id=891)

X = dataset.data.features
y = dataset.data.targets

# ترکیب ویژگی‌ها و ستون هدف
df = pd.concat(
    [
        X.reset_index(drop=True),
        y.reset_index(drop=True)
    ],
    axis=1
)

# حذف شناسه در صورت وجود
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

df.to_csv(OUTPUT_FILE, index=False)

print("Download completed.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved to:", OUTPUT_FILE)

print(df.head())
