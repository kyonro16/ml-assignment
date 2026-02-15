# ML Assignment 2 – Classification Models & Streamlit App

## a. Problem statement

This project implements an end-to-end machine learning workflow for **multi-class classification** of **Credit Score** (Good / Standard / Poor) using the **Score.csv** dataset. Six classification models are trained and evaluated, and an interactive Streamlit app allows users to upload test data (CSV), select a model, and view evaluation metrics along with a confusion matrix and classification report.

---

## b. Dataset description

- **Source:** Score.csv (credit-related features and Credit_Score target).
- **Task:** Multi-class classification — predicting **Credit Score** as **Good**, **Standard**, or **Poor**.
- **Features:** 20 input columns:
  - **Numeric:** Delay_from_due_date, Num_of_Delayed_Payment, Num_Credit_Inquiries, Credit_Utilization_Ratio, Credit_History_Age, Amount_invested_monthly, Monthly_Balance, Age, Annual_Income, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan, Monthly_Inhand_Salary, Changed_Credit_Limit, Outstanding_Debt, Total_EMI_per_month.
  - **Categorical:** Payment_of_Min_Amount (No/Yes/NM), Credit_Mix (Good/Standard), Payment_Behaviour (e.g. High_spent_Medium_value_payments, Low_spent_Large_value_payments).
- **Target:** Credit_Score (Good, Standard, Poor).
- **Instances:** 99,960 samples (meets minimum 500).
- **Preprocessing:** Train–test split (75%–25%, stratified). Numeric features are scaled (StandardScaler); categorical features are one-hot encoded (OneHotEncoder with handle_unknown='ignore').

---

## c. Models used

Comparison table of evaluation metrics for all 6 models (on the same dataset, 25% held-out test set):

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression      | 0.6626   | 0.8   | 0.6674    | 0.6626 | 0.662 | 0.443 |
| Decision Tree            | 0.7409   | 0.776 | 0.7408    | 0.7409 | 0.7408| 0.5688|
| kNN                      | 0.7581   | 0.8731| 0.7601    | 0.7581 | 0.7588| 0.6016|
| Naive Bayes              | 0.6458   | 0.7493| 0.7055    | 0.6458 | 0.6505| 0.4844|
| Random Forest (Ensemble) | 0.813    | 0.91  | 0.8135    | 0.813  | 0.813 | 0.6911|
| XGBoost (Ensemble)       | 0.7725   | 0.8811| 0.7736    | 0.7725 | 0.7729| 0.6234|



### Observations on model performance

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Provides a strong, interpretable baseline for multi-class credit score. Benefits from scaled and one-hot-encoded features; coefficients can be inspected to see which features (e.g. delayed payments, utilization) drive Good vs Poor. Accuracy and AUC reflect how well linear decision boundaries separate the three classes. |
| Decision Tree            | Interpretable and does not require feature scaling. Can capture non-linear rules (e.g. “if Num_of_Delayed_Payment > X then Poor”). May overfit compared to ensemble methods; pruning or depth limits can improve generalization. Useful for explaining individual predictions. |
| kNN                      | Performance depends on scaling (applied in preprocessing) and choice of k. Can model local patterns in the feature space (e.g. similar payment behaviour leading to similar credit score). May be slower at prediction on large test sets and sensitive to irrelevant or noisy features. |
| Naive Bayes              | Fast to train and robust to irrelevant features under the independence assumption. Often competitive on credit-style data where many features (income, delays, utilization) contribute. May show slightly lower recall on minority classes (e.g. Poor) depending on class balance. |
| Random Forest (Ensemble) | Averages many trees to reduce overfitting and variance. Typically more stable and accurate than a single decision tree on this dataset. Handles mixed numeric and one-hot features well; feature importance can highlight which credit factors matter most overall. |
| XGBoost (Ensemble)       | Gradient boosting often achieves among the best accuracy and AUC by sequentially correcting errors. Tends to excel when there are many informative features and sufficient data, as in this credit score setting. May require more tuning (e.g. learning rate, depth) than Random Forest for optimal results. |

---

## Repository structure

```
project-folder/
├── Score.csv              # Dataset (required for training)
├── app.py                 # Streamlit app
├── requirements.txt
├── README.md
├── generate_sample_test_data.py   # Optional: creates sample_test_data.csv from Score.csv
└── model/
    ├── train.py           # Trains all 6 models using Score.csv
    └── saved/             # Preprocessor, label encoder, saved models (*.joblib), metrics.csv
```

---

## How to run

1. **Place Score.csv** in the project root (same folder as `app.py`).

2. **Install dependencies:**  
   `pip install -r requirements.txt`

3. **Train models (required before using the app):**  
   `python model/train.py`  
   This creates `model/saved/` with preprocessor, label encoder, `.joblib` models, and `metrics.csv`.

4. **Run the Streamlit app:**  
   `streamlit run app.py`

5. **Use the app:**  
   Upload a test CSV with the **same 20 feature columns** as Score.csv (and a `Credit_Score` column for evaluation). Select a model from the dropdown to see metrics, confusion matrix, and classification report.

6. **Optional – generate sample test CSV:**  
   `python generate_sample_test_data.py`  
   Creates `sample_test_data.csv` from the same split as training for quick testing.

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (include `Score.csv` if allowed, or ensure `model/saved/` is committed after running `model/train.py` locally).
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud), sign in with GitHub.
3. New App → select this repository, branch `main`, main file `app.py` → Deploy.
4. The deployed app needs the files in `model/saved/` (trained models and preprocessor) to run.
