import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = r'/storage/emulated/0/pythonproject2/health-insurance-project/train.csv'
df = pd.read_csv(file_path)

# 6.1 Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))
numerical_df = df.select_dtypes(include='number')
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 6.2 Boxplots of numerical features vs Response
numerical_cols = df.select_dtypes(include='number').columns.tolist()
numerical_cols.remove('Response')  # Exclude target

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Response', y=col, data=df, palette='Set2')
    plt.title(f'{col} vs Response')
    plt.tight_layout()
    plt.show()

# 6.3 Countplots of categorical features vs Response
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='Response', palette='Set1')
    plt.title(f'{col} vs Response')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()