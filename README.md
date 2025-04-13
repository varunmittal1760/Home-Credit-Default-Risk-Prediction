# Home-Credit-Default-Risk-Prediction

## 📌 Problem Statement
Predict whether a loan applicant will default using historical data provided by Home Credit. This is a supervised binary classification problem.

## 📂 Dataset
Sourced from [Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data).  
Includes:
- `application_train.csv`, `application_test.csv` – core datasets.
- Supporting datasets: `bureau.csv`, `previous_application.csv`, `credit_card_balance.csv`, etc.

## 🧠 Key EDA Insights
- **Class imbalance**: Only 8.1% are defaulters.
- **Top factors**: EXT_SOURCE_2, EXT_SOURCE_3, DAYS_BIRTH.
- **Higher default risk**: Younger age, more children, certain occupations.
- **Missing values**: Present in ~50% of features; handled via imputation or model-native methods.

## 🛠️ Preprocessing & Feature Engineering
- Label encoding + one-hot encoding.
- New features (e.g., `INCOME_CREDIT_PERC`, `DAYS_EMPLOYED_PERC`, `PAYMENT_PERC`).
- Aggregations from auxiliary datasets (e.g., `bureau_balance`, `POS_CASH_balance`).

## 🔍 Models Explored
| Model               | AUC    | True Positives | Accuracy | Strength               |
|--------------------|--------|----------------|----------|------------------------|
| Logistic Regression| 0.7664 | 437            | ~92%     | Simplicity, interpretability |
| Random Forest       | 0.7125 | 45             | ~92%     | Ensemble stability     |
| LightGBM            | 0.7858 | 4526           | ~92%     | Best recall, class handling |
| XGBoost             | 0.7564 | 499            | ~92%     | Handles categorical splits well |

✅ **Best Overall**: LightGBM (strong AUC + best recall)  
📊 **Top Features Across Models**: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `AMT_CREDIT`, `INCOME_CREDIT_PERC`.

## 🧪 Next Steps
- Hyperparameter tuning (e.g., Bayesian Optimization).
- Advanced model stacking/ensembling.
- Explore deep learning methods for non-linear patterns.

## 📚 References
- [Kaggle Competition Overview](https://www.kaggle.com/competitions/home-credit-default-risk/overview)
- [Quantifying Credit Risk – Investopedia](https://www.investopedia.com/ask/answers/022415/what-factors-are-taken-account-quantify-credit-risk.asp)
