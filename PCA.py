import numpy as np  # For numerical operations
import pandas as pd  # For handling data in tabular format
from sklearn.decomposition import PCA  # For performing Principal Component Analysis
from sklearn.preprocessing import StandardScaler  # For standardizing features

# Load the dataset from a CSV file.
data = pd.read_csv("preprocessed_CIC-NEW.csv")

# Separate the independent variables (features) from the dependent variable (label/target).
# Drop the "label" column to extract features and keep "label" as the target variable.
features = data.drop(columns=["label"], axis=1)  # Features (X)
labels = data["label"]  # Labels (y)

# Standardize the features to have zero mean and unit variance.
# This step ensures all features contribute equally to the PCA.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Initialize PCA without specifying the number of components to retain all initially.
pca = PCA(n_components=None)
pca.fit(features_scaled)  # Fit PCA to the standardized feature data

# Extract the explained variance ratio for each principal component.
# The explained variance ratio indicates the proportion of the dataset's variance captured by each component.
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate the cumulative variance ratio, which shows how much total variance is retained as components are added.
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Print the variance explained by each component
print("Explained Variance Ratio:")
print(explained_variance_ratio)

# Print the cumulative variance explained as components are added.
print("\nCumulative Variance Ratio:")
print(cumulative_variance_ratio)

# Determine the number of components required to retain a desired level of variance (e.g., 95%).
threshold = 0.95  # Retain 95% of the dataset's variance
num_components = np.argmax(cumulative_variance_ratio >= threshold) + 1  # Find the index where cumulative variance >= threshold
print(f"\nNumber of components needed to retain {threshold*100:.1f}% variance: {num_components}")

# Perform PCA again, now with the optimal number of components identified in the previous step.
pca = PCA(n_components=num_components)
features_reduced = pca.fit_transform(features_scaled)  # Reduce the dataset dimensions

# Create a new DataFrame with the reduced features (principal components).
# Each column represents one principal component (PC), and the labels are appended as a separate column.
reduced_data = pd.DataFrame(features_reduced, columns=[f"PC_{i}" for i in range(1, num_components + 1)])
reduced_data["label"] = labels.values  # Add labels back to the reduced dataset

# Print the first few rows of the reduced dataset for verification.
print("\nReduced Dataset (First 5 Rows):")
print(reduced_data.head().to_string())

"""# Save the reduced dataset to a CSV file for further use.
reduced_data.to_csv("PCA-SelectedFeatures.csv", index=False)"""

