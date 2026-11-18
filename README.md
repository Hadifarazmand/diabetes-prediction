
# Diabetes Prediction Research Project

## Project Overview

A machine learning research project for diabetes-related
classification using two independent datasets.

The project compares multiple machine learning algorithms
and evaluates their performance using a common test set.

## Datasets

### 1. CDC Diabetes Health Indicators

Source:
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

- 253,680 records
- 21 predictor features in the commonly used binary version
- Target: Diabetes_binary
- 0 = No diabetes
- 1 = Prediabetes or diabetes

DOI:
https://doi.org/10.24432/C53919

### 2. Kaggle Diabetes Prediction Dataset

Source:
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Expected columns:

- gender
- age
- hypertension
- heart_disease
- smoking_history
- bmi
- HbA1c_level
- blood_glucose_level
- diabetes

The data provenance and target-generation process
must be investigated before making clinical claims.

## Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Hist Gradient Boosting
- XGBoost (optional)

## Evaluation Metrics

- Accuracy
- Error Rate
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Balanced Accuracy
- Confusion Matrix

All algorithms are evaluated using the same stratified
80/20 train-test split.

## Important Research Notes

The two datasets are trained and evaluated separately.
They are not merged because their target definitions
and feature sets are different.

The CDC binary target represents prediabetes or diabetes,
not diabetes alone.

The Kaggle dataset includes HbA1c and blood glucose.
A second experiment excludes these laboratory variables
to investigate their influence and potential target leakage.

The displayed predict_proba score is not a clinically
calibrated probability.

This application is for educational and research purposes
and is not a medical diagnostic tool.

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py
