import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

# Load the dataset
file_path = r"C:\Users\raghu\Desktop\cleaned_diabetes_dataset.csv"
data = pd.read_csv(file_path)

### Exploratory Data Analysis (EDA)

# Diabetes Status in the Dataset
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
sns.countplot(x='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of Diabetes Prevalence')
plt.xlabel('Diabetes Status')
plt.ylabel('Count')
plt.show()

# Create the count plot
plt.figure(figsize=(8, 6))
sns.countplot(x='Diabetes_012', hue='HeartDiseaseorAttack', data=data, palette=colors)
plt.title('Count Plot of Diabetes vs Heart Disease or Attack')
plt.xlabel('Diabetes')
plt.ylabel('Count')
plt.show()

# Create a cross-tabulation table
cross_tab = pd.crosstab(data['Diabetes_012'], data['HeartDiseaseorAttack'])

# Print the cross-tabulation table
print("Cross-tabulation table:")
print(cross_tab)

# Calculate the correlation coefficient
correlation = cross_tab.corr().iloc[0, 1]
print("\nCorrelation coefficient between Diabetes and Heart Disease or Attack:", correlation)

# Select columns of interest
columns_of_interest = ["Smoker", "Age", "Sex", "Education", "Fruits", "Veggies", "BMI", "Stroke", "Diabetes_012"]
comparison_data = data[columns_of_interest]

# Create the boxplot with custom colors
plt.figure(figsize=(12, 8))
sns.boxplot(data=comparison_data, x="Diabetes_012", y="Age", palette=colors)
plt.title('Boxplot of Age grouped by Diabetes')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=comparison_data, x="Diabetes_012", y="BMI", palette=colors)
plt.title('Boxplot of BMI grouped by Diabetes')
plt.show()

# Countplot comparison for categorical variables
categorical_columns = ["Smoker", "Sex", "Education", "Stroke"]
for column in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=comparison_data, x=column, hue="Diabetes_012", palette=colors)
    plt.title(f'Countplot of {column} grouped by Diabetes')
    plt.xticks(rotation=0)
    plt.show()

    # Calculate correlation for each categorical variable
for column in categorical_columns:
    # Encode the categorical variable as binary
    encoded_column = pd.get_dummies(comparison_data[column], drop_first=True).iloc[:, 0]  # Selecting the first column

    # Calculate correlation with Diabetes_012
    correlation = pointbiserialr(encoded_column, comparison_data['Diabetes_012'])

    print(f"Correlation between {column} and Diabetes_012:", correlation.correlation)
    print("p-value:", correlation.pvalue)

# Bar Plot of High Blood Pressure (HighBP) by Diabetes Status
plt.figure(figsize=(8, 6))
sns.countplot(x='HighBP', hue='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of HighBP by Diabetes Status')
plt.xlabel('High Blood Pressure')
plt.ylabel('Count')
plt.show()

# Bar Plot of High Cholesterol (HighChol) by Diabetes Status
plt.figure(figsize=(8, 6))
sns.countplot(x='HighChol', hue='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of HighChol by Diabetes Status')
plt.xlabel('High Cholesterol')
plt.ylabel('Count')
plt.show()

# Calculate correlation for HighBP
encoded_high_bp = pd.get_dummies(data['HighBP'], drop_first=True).iloc[:, 0]  # Selecting the first column
correlation_high_bp = pointbiserialr(encoded_high_bp, data['Diabetes_012'])
print("Correlation between High Blood Pressure and Diabetes", correlation_high_bp.correlation)
print("p-value:", correlation_high_bp.pvalue)

# Calculate correlation for HighChol
encoded_high_chol = pd.get_dummies(data['HighChol'], drop_first=True).iloc[:, 0]  # Selecting the first column
correlation_high_chol = pointbiserialr(encoded_high_chol, data['Diabetes_012'])
print("Correlation between High Cholesterol and Diabetes", correlation_high_chol.correlation)
print("p-value:", correlation_high_chol.pvalue)

# Bar Plot of Physically Active (PhysActivity) by Diabetes Status
plt.figure(figsize=(8, 6))
sns.countplot(x='PhysActivity', hue='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of PhysActivity by Diabetes Status')
plt.xlabel('Physical Activity')
plt.ylabel('Count')
plt.show()

# Bar Plot of Fruits Consumption by Diabetes Status
plt.figure(figsize=(8, 6))
sns.countplot(x='Fruits', hue='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of Fruits Consumption by Diabetes Status')
plt.xlabel('Fruits Consumption')
plt.ylabel('Count')
plt.show()

# Bar Plot of Vegetable Consumption by Diabetes Status
plt.figure(figsize=(8, 6))
sns.countplot(x='Veggies', hue='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of Vegetable Consumption by Diabetes Status')
plt.xlabel('Vegetable Consumption')
plt.ylabel('Count')
plt.show()

# Bar Plot of General Health by Diabetes Status
plt.figure(figsize=(8, 6))
sns.countplot(x='GenHlth', hue='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of General Health by Diabetes Status')
plt.xlabel('General Health')

# Bar Plot of Income Level by Diabetes Status
plt.figure(figsize=(8, 6))
sns.countplot(x='Income', hue='Diabetes_012', data=data, palette=colors)
plt.title('Bar Plot of Income Level by Diabetes Status')
plt.xlabel('Income Level')
plt.ylabel('Count')
plt.show()

# Calculate correlation for HighBP
encoded_high_bp = pd.get_dummies(data['HighBP'], drop_first=True).iloc[:, 0]  # Selecting the first column
correlation_high_bp = pointbiserialr(encoded_high_bp, data['Diabetes_012'])

# Calculate correlation for HighChol
encoded_high_chol = pd.get_dummies(data['HighChol'], drop_first=True).iloc[:, 0]  # Selecting the first column
correlation_high_chol = pointbiserialr(encoded_high_chol, data['Diabetes_012'])

# Create a DataFrame to store correlation coefficients
correlation_data = pd.DataFrame({
    'High Blood Pressure': [correlation_high_bp.correlation],
    'High Cholesterol': [correlation_high_chol.correlation]
}, index=['Correlation with Diabetes'])

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between High Blood Pressure/High Cholesterol and Diabetes')
plt.show()
