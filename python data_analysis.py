
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'your_dataset.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display summary statistics
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Handling missing values (example: filling with mean)
data.fillna(data.mean(), inplace=True)

# Correlation matrix
print("\nCorrelation matrix:")
print(data.corr())

# Visualizations
# Histogram of all numerical features
data.hist(figsize=(10, 10))
plt.show()

# Pairplot
sns.pairplot(data)
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Save cleaned dataset
cleaned_file_path = 'cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")
# Histogram of all numerical features
data.hist(figsize=(10, 10))
plt.savefig('histograms.png')

# Pairplot
sns.pairplot(data)
plt.savefig('pairplot.png')

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.savefig('heatmap.png')
