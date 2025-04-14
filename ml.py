# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Step 1: Load the dataset
df = pd.read_csv('AmesHousing.csv')

# Step 2: Initial Exploration
# Show the first few rows of the dataset
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3: Visualize Data
# Plot the distribution of 'SalePrice'
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.show()

# Correlation heatmap to understand relationships between numerical variables
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Data Cleaning and Feature Engineering
# Handle missing values by filling with median for numerical columns
df.fillna(df.median(), inplace=True)

# Create a new feature 'Age' (current year - YearBuilt)
df['Age'] = 2023 - df['YearBuilt']

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

# Step 5: Key Findings and Insights
# Explore relationship between 'OverallQual' and 'SalePrice'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='OverallQual', y='SalePrice', data=df)
plt.title('Overall Quality vs Sale Price')
plt.show()

# Correlation between 'OverallQual' and 'SalePrice'
correlation = df[['OverallQual', 'SalePrice']].corr()
print("\nCorrelation between 'OverallQual' and 'SalePrice':")
print(correlation)

# Step 6: Formulate Hypotheses
# Hypothesis 1: Homes with higher overall quality (OverallQual) will have higher sale prices.
# Hypothesis 2: Larger homes (LotArea) will have a higher sale price.
# Hypothesis 3: Homes built after 2000 will have a higher sale price than those built earlier.

# Step 7: Test Hypothesis 1 (Correlation Test between 'OverallQual' and 'SalePrice')
corr, p_value = pearsonr(df['OverallQual'], df['SalePrice'])
print(f"\nPearson Correlation Test for 'OverallQual' and 'SalePrice':")
print(f"Correlation: {corr}, p-value: {p_value}")

# Interpretation of p-value
if p_value < 0.05:
    print("There is a significant correlation between 'OverallQual' and 'SalePrice'.")
else:
    print("No significant correlation between 'OverallQual' and 'SalePrice'.")

# Step 8: Suggestions for Next Steps
print("\nNext Steps:")
print("- Build machine learning models such as Linear Regression, Random Forest, etc., to predict 'SalePrice'.")
print("- Segment the data by features such as 'Neighborhood' or 'BldgType' for more targeted insights.")
print("- Explore the impact of other factors like 'LotFrontage', 'GarageCars', and 'TotRmsAbvGrd' on 'SalePrice'.")

# Step 9: Quality of Data Set and Additional Data
print("\nData Quality Summary:")
print("The Ames Housing dataset is fairly clean with only a few missing values which were imputed using the median. "
      "The features are mostly well-defined. However, the dataset could benefit from additional information on the "
      "geographical location of homes and the economic conditions of the area. Acquiring more data on home renovations "
      "or neighborhood-level income could provide further insights into house prices.")
