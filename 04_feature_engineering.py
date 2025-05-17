import pandas as pd

# Load the cleaned dataset
file_path = r'/storage/emulated/0/pythonproject2/health-insurance-project/train.csv'
df = pd.read_csv(file_path)

# Drop unique ID column
df.drop(columns=['id'], inplace=True)

# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Vehicle_Age'] = df['Vehicle_Age'].map({
    '> 2 Years': 2,
    '1-2 Year': 1,
    '< 1 Year': 0
})
df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

# Confirm transformation
print(df.head())
print(df.info())