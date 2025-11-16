# Cell 1: Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

print("✅ Libraries imported successfully!")

# Cell 2: ZIP File Extraction (Assumes a ZIP file named 'student_data.zip' was uploaded)
import zipfile
import os

# Replace filename with your uploaded zip file name
zip_path = '/content/student_data.zip'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')
print("✅ ZIP file extracted successfully!")

# List extracted files
os.listdir('/content')

# Cell 3: Load Data
import pandas as pd

df = pd.read_csv('/content/Titanic-Dataset.csv')  # change name if different
df.head()

# Cell 4: Initial Data Check
df.info()
print(df.head())

# Cell 5: Data Exploration
# STEP 5 – Explore the data
print("✅ BASIC DATA INFORMATION\n")

# Show general information
df.info()

# Count of missing values in each column
print("\n🧩 Missing Values:\n", df.isnull().sum())

# Basic numerical statistics (mean, std, etc.)
print("\n📊 Statistical Summary:\n", df.describe())

# Show first few rows
print("\n👀 Sample Rows:")
display(df.head())

# Cell 6: Handle Missing Values
# STEP 6 – Handle Missing Values
print("🎯 Imputing Missing Values...")

# 1. Impute 'Age' with the median
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# 2. Impute 'Embarked' with the mode
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

# 3. Drop 'Cabin' column (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

print("✅ Missing values handled successfully!")
print("\nUpdated Missing Values:\n", df.isnull().sum())
df.head()

# Cell 7: Feature Engineering (Extract Title)
# STEP 7 – Feature Engineering (Extract Title from Name)
print("🛠️ Extracting Features...")

df['Title'] = df['Name'].apply(lambda x: x.split(', ')[1].split('.')[0].strip())
df.drop('Name', axis=1, inplace=True)

print("Unique Titles:", df['Title'].unique())

# Group rare titles into 'Other'
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Other')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

print("\nSimplified Unique Titles:", df['Title'].unique())
df.head()

# Cell 8: Label Encoding for Categorical Columns
from sklearn.preprocessing import LabelEncoder

print("🎯 Encoding Categorical Columns...")

# Identify categorical columns again (in case you changed df)
categorical_cols = df.select_dtypes(exclude=['number']).columns

# Apply Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("✅ All categorical columns encoded successfully!")
print("\nEncoded columns:", categorical_cols)
df.head()

# Cell 9: Outlier Visualization (Boxplots)
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 Visualizing Outliers...")

numeric_cols = df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Cell 10: Outlier Treatment (IQR Method)
# STEP 8 – Outlier Treatment (Using IQR Method)
print("🧹 Removing Outliers...")

# List of columns to check for outliers (excluding ID and binary columns)
outlier_cols = ['Age', 'Fare', 'SibSp', 'Parch']

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("✅ Outliers handled successfully!")
print("New shape of dataset:", df.shape)

# Cell 11: Data Scaling (Standardization)
# STEP 9 – Data Scaling (Standardization)
print("⚖️ Scaling Numerical Features...")

scaler = StandardScaler()

# Columns to scale (excluding ID and target)
cols_to_scale = ['Age', 'Fare', 'SibSp', 'Parch']

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("✅ Features scaled successfully!")
df.head()

# Cell 12: Save the cleaned dataset
# STEP 10 – Save the cleaned dataset
output_path = '/content/cleaned_dataset.csv'

# Save as CSV file
df.to_csv(output_path, index=False)

print("🎉 Cleaned dataset saved successfully as cleaned_dataset.csv!")
