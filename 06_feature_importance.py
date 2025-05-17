import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
file_path = r'/storage/emulated/0/pythonproject2/health-insurance-project/train.csv'
df = pd.read_csv(file_path)

# Convert categorical variables to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df_encoded.drop(columns=['Response'])
y = df_encoded['Response']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model for feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importance = rf.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# Save and display feature importance chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.savefig('/storage/emulated/0/pythonproject2/health-insurance-project/feature_importance.png')
plt.show()

# Save feature importance data
feature_imp_df.to_csv('/storage/emulated/0/pythonproject2/health-insurance-project/feature_importance.csv', index=False)