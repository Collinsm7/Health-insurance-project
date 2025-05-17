import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
file_path = r'/storage/emulated/0/pythonproject2/health-insurance-project/train.csv'
df = pd.read_csv(file_path)
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=['Response'])
y = df_encoded['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the best model again
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Save the model
model_path = r'/storage/emulated/0/pythonproject2/health-insurance-project/rf_model.pkl'
joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")