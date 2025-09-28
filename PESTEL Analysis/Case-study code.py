# 📦 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.graphics.tukeyplot import results
from statsmodels.tsa.api import ExponentialSmoothing

# 🔄 Step 1: Load the Data
df = pd.read_csv("healthcare_dataset.csv")  # Full path to be provided if file is not in project folder

# 1. Dataset Overview
# -----------------------------
# Preview structure and data quality
print("📋 Dataset Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())
print("-" * 40)

# Data Cleaning
# -----------------------------
# 1. Standardize Text Fields
# -----------------------------
text_columns = ['Gender', 'Medical Condition', 'Medication', 'Doctor', 'Hospital']
for col in text_columns:
    df[col] = df[col].astype(str).str.strip().str.title()

# -----------------------------
# 2. Date Conversion & Validation
# -----------------------------
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')

# -----------------------------
# 3. Missing Values Check
# -----------------------------
missing_summary = df.isnull().sum()
print("🔍 Missing values summary:\n", missing_summary)
print("-" * 40)

# -----------------------------
# 4. Outlier Detection
# -----------------------------

# Select numeric columns only
numeric_cols = df.select_dtypes(include=['number']).columns

# Function to detect outliers using IQR
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Dictionary to hold outlier summary
outlier_summary = []

# Analyze each numeric column
for col in numeric_cols:
    outlier_mask = detect_outliers_iqr(df[col])
    num_outliers = outlier_mask.sum()
    perc_outliers = (num_outliers / len(df)) * 100
    outlier_summary.append({
        'Column': col,
        'Outliers Count': num_outliers,
        'Percentage': round(perc_outliers, 2)
    })

# Convert summary to DataFrame and display
outlier_summary_df = pd.DataFrame(outlier_summary)
print("📊 Outlier Summary (IQR Method):")
print(outlier_summary_df)
print("-" * 40)

# -----------------------------
# 5. Feature Engineering
# -----------------------------
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Billing Per Day'] = df['Billing Amount'] / df['Length of Stay'].replace(0, np.nan)
df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 120], labels=['<30', '30-50', '50+'])
df['Admission Month'] = df['Date of Admission'].dt.to_period('M').astype(str)
df['Admission Weekday'] = df['Date of Admission'].dt.day_name()
df['Month'] = df['Date of Admission'].dt.to_period('M')

# Dictionary to map Medical conditions to numbers
condition_map = {
    'Arthritis': 1,
    'Asthma': 2,
    'Cancer':3,
    'Diabetes':4,
    'Hypertension':5,
    'Obesity':6
    # Add more conditions as needed
}
# Create new column with numeric codes
df['Condition Code'] = df['Medical Condition'].map(condition_map)

# Dictionary to map Insurance Provider to numbers
condition_map = {
    'Aetna': 1,
    'Blue Cross':2,
    'Cigna':3,
    'Medicare':4,
    'UnitedHealthcare':5
    # Add more conditions as needed
}
# Create new column with numeric codes
df['Insurance Code'] = df['Insurance Provider'].map(condition_map)

# Drop invalid dates
df = df.dropna(subset=['Date of Admission', 'Discharge Date'])
df = df[df['Discharge Date'] >= df['Date of Admission']]

# -----------------------------
# 6. Descriptive Statistics
# -----------------------------
print("\n📊 Descriptive Statistics for Numerical Features:")
print(df[['Age', 'Billing Amount', 'Length of Stay', 'Billing Per Day']].describe())

print("\n🔢 Value Counts for Categorical Features:")
print("Gender:\n", df['Gender'].value_counts())
print("\nAge Group:\n", df['Age Group'].value_counts())
print("\nMedical Condition:\n", df['Medical Condition'].value_counts())
print("-" * 40)

# -----------------------------
# 7. Visualizations
# -----------------------------

# Histogram: Age distribution
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Barplot: Count by Gender
sns.countplot(x='Gender', data=df)
plt.title('Patient Gender Distribution')
plt.tight_layout()
plt.show()

# Boxplot: Billing Amount by Age Group
sns.boxplot(x='Age Group', y='Billing Amount', data=df)
plt.title('Billing Amount by Age Group')
plt.tight_layout()
plt.show()

# Lineplot: Monthly Admission Trends
monthly_admissions = df['Admission Month'].value_counts().sort_index()
monthly_admissions.plot(kind='line', marker='o', title='Monthly Admission Trends')
plt.xlabel('Month')
plt.ylabel('Admissions')
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Save Cleaned Data
# -----------------------------
df.to_csv('cleaned_healthcare_dataset.csv', index=False)
print("✅ Cleaned and enriched dataset saved.")
print("-" * 40)

# -----------------------------
# 9. Statistical Analysis
# -----------------------------

# --- Descriptive Statistics ---
print("Descriptive Statistics:\n", df.describe(include='all'))
print("Mode:\n", df.mode().iloc[0])
print("Median:\n", df.median(numeric_only=True))
print("Standard Deviation:\n", df.std(numeric_only=True))
print("Range:\n", df.max(numeric_only=True) - df.min(numeric_only=True))
print("-" * 40)

# --- Trend Analysis ---
monthly_trend = df.groupby('Month')['Billing Amount'].sum()
print("Monthly Billing Trend:\n", monthly_trend)
print("-" * 40)

# --- Correlation ---
correlation = df[['Age', 'Billing Amount', 'Length of Stay']].corr()
print("Correlation Matrix:\n", correlation)
print("-" * 40)

# --- Regression Analysis ---
X = pd.get_dummies(df[['Age', 'Condition Code', 'Insurance Code']], drop_first=True)
X = sm.add_constant(X)
y = df['Billing Amount']
model = sm.OLS(y, X).fit()
print(model.summary())
print("-" * 40)

# --- Chi-Square Test (Gender vs Medical Condition) ---
contingency = pd.crosstab(df['Gender'], df['Medical Condition'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi-Square Test:\nChi2 = {chi2}, p-value = {p}, DOF = {dof}")
print("-" * 40)

# -----------------------------
# 10. Visualizations
# -----------------------------

# Trend Analysis: Monthly Billing
# -----------------------------
# Ensure correct plotting format for month
monthly_trend.index = monthly_trend.index.astype(str)

plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_trend.index, y=monthly_trend.values, marker='o')
plt.title("Monthly Billing Trend")
plt.xlabel("Month")
plt.ylabel("Total Billing Amount")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Age, Billing, Length of Stay)")
plt.tight_layout()
plt.show()

# Regression Residual Plot
# -----------------------------
# Predicted vs Actual Billing Amount
predicted = model.predict(X)
residuals = y - predicted

plt.figure(figsize=(8, 5))
sns.residplot(x=predicted, y=residuals, lowess=True, color="g", line_kws={'color': 'red'})
plt.title("Residuals vs Fitted Values")
plt.xlabel("Predicted Billing Amount")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# Chi-Square
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Gender vs Medical Condition (Counts)")
plt.xlabel("Medical Condition")
plt.ylabel("Gender")
plt.tight_layout()
plt.show()
