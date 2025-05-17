# Health Insurance Response Prediction Project

## Objective
To predict whether a health insurance policyholder is likely to show interest in a new offering (`Response = 1`) or not (`Response = 0`) using machine learning.

---

## Steps Taken

1. **Data Loading & Inspection**  
   - Loaded data from CSV  
   - Inspected structure, types, and missing values

2. **Exploratory Data Analysis (EDA)**  
   - Univariate: Distribution of age, BMI, etc.  
   - Bivariate: Relationship between features and `Response`

3. **Data Cleaning & Encoding**  
   - Handled missing values  
   - Converted categorical features using One-Hot Encoding

4. **Model Building**  
   - Baseline: Logistic Regression  
   - Final: Random Forest Classifier  
   - Achieved ~87.5% accuracy on test data

5. **Feature Importance**  
   - Identified top predictors: Age, Previously_Insured, Policy_Sales_Channel, etc.

6. **Model Saving**  
   - Saved the trained model using `joblib` for future use

---

## Tools Used
- Python  
- Pandas, Scikit-learn  

---

## Next Steps
- Improve model with hyperparameter tuning  
- Try cross-validation  
- Deploy as a REST API (future plan)

---

## Conclusion
Successfully built a predictive model to classify health insurance responses using machine learning. Ready for deployment or integration into health policy campaigns.