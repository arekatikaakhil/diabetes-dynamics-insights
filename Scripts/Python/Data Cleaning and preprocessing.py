import pandas as pd

# Load the dataset
file_path = r"C:\Users\raghu\Desktop\CS504\CS PROJECT\diabetes_health_indicators.csv"
data = pd.read_csv(file_path)

# Print the first few rows of the dataset
print("Head of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Impute missing values (example: filling missing numeric values with mean)
data.fillna(data.mean(), inplace=True)

# Identify outliers using z-score method for numerical columns
z_scores = (data - data.mean()) / data.std()
outliers = (z_scores > 3) | (z_scores < -3)
outlier_count = outliers.sum()
print("\nOutliers (count per column):")
print(outlier_count)

# Print summary statistics
print("\nSummary statistics:")
print(data.describe())

# Print the shape of the dataset
print("\nShape of the dataset:", data.shape)

# Save the cleaned dataset
data.to_csv("cleaned_diabetes_dataset.csv", index=False)
