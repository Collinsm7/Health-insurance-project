# Health Insurance Response Prediction

A machine learning project to predict customer interest in a new insurance product based on their profile data. This project uses data preprocessing, exploratory analysis, feature engineering, and classification models to solve a real-world business problem in the insurance domain.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Tools and Libraries](#tools-and-libraries)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Model Performance](#model-performance)
- [Results and Insights](#results-and-insights)
- [Future Work](#future-work)
- [Author](#author)

---

## Project Overview

The goal is to predict whether a health insurance policyholder will be interested (`Response = 1`) or not interested (`Response = 0`) in a new offering. This kind of prediction helps insurers target potential customers more efficiently and improve campaign ROI.

---

## Dataset

- **Source**: [Kaggle - Health Insurance Cross-Sell Prediction](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction)
- **Rows**: ~381,000
- **Features**: 12 including `Age`, `Gender`, `Vehicle_Age`, `Policy_Sales_Channel`, `Previously_Insured`, etc.

---

## Tools and Libraries

- **Python 3**
- **Pandas** – Data manipulation
- **Matplotlib & Seaborn** – Visualization
- **Scikit-learn** – Modeling and evaluation
- **Joblib** – Model saving

---

## Project Structure

```bash
.
├── 01_load_and_explore.py         # Data loading and EDA
├── 02_clean_data.py               # Data cleaning
├── 03_encode_data.py              # One-hot encoding
├── 04_split_data.py               # Train-test split
├── 05_logistic_regression.py      # Logistic Regression training
├── 06_feature_importance.py       # Feature importance with Random Forest
├── 07_random_forest.py            # Random Forest classifier
├── 08_model_evaluation.py         # Metrics and evaluation
├── 09_summary.md                  # Key insights and findings
├── 10_save_model.py               # Export final model using joblib
├── train.csv                      # Raw dataset
├── final_model.pkl                # Saved model file
└── README.md                      # Project description


---

How to Run the Project

1. Clone this repository:

git clone https://github.com/yourusername/insurance-prediction.git
cd insurance-prediction


2. Install the required libraries:

pip install -r requirements.txt


3. Run the scripts in order:

python 01_load_and_explore.py
...
python 10_save_model.py




---

Model Performance

Note: These values may vary slightly depending on data split and seed.


---

Results and Insights

Previously_Insured and Vehicle_Age were among the top predictors.

Customers not previously insured showed higher interest.

Random Forest outperformed Logistic Regression slightly in recall and F1.



---

Future Work

Try XGBoost or LightGBM for better accuracy.

Build a Streamlit or Flask web app to deploy the model.

Perform hyperparameter tuning using GridSearchCV.

Apply SMOTE for class imbalance handling.



---

Author

Collins Macharia
Certified Data Scientist | Actuarial Science Graduate
Passionate about data, technology, and solving business problems with AI.

GitHub: Collinsm7 

LinkedIn: https://www.linkedin.com/in/collins-macharia-5ba365244

Email: comathe17@gmail.com
