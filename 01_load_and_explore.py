import pandas as pd
import os

# Define the file path
file_path = r'/storage/emulated/0/pythonproject2/health-insurance-project/train.csv'

# Check if the file exists
if os.path.exists(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    # Quick overview
    print(df.shape)
    print(df.columns)
    print(df.head())
else:
    print(f"File not found at {file_path}")
    # Check data types and non-null values
print(df.info())

# Check missing values in each column
print(df.isnull().sum())
print(df.describe())
print(df['Response'].value_counts(normalize=True))