import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
file_path = r'/storage/emulated/0/pythonproject2/health-insurance-project/train.csv'
df = pd.read_csv(file_path)

# 5.1 Visualize numerical distributions
numerical_cols = df.select_dtypes(include='number').columns.tolist()

df[numerical_cols].hist(bins=30, figsize=(15, 10), color='skyblue')
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()

# 5.2 Visualize categorical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# 5.3 Optional: Explore target distribution across categories
for col in categorical_cols:
    print(f'\n{col} vs Response (%):\n')
    print(pd.crosstab(df[col], df['Response'], normalize='index') * 100)