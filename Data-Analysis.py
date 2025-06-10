import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, normaltest, probplot
from sklearn import preprocessing
import pandas as pd

# Data
data = pd.read_csv("CICIoT2023_small.csv")
data.drop( "Unnamed: 0",axis=1,inplace=True)

#Encoding non-numeric values to numeric
labelEncoder=preprocessing.LabelEncoder()
data["label"]=labelEncoder.fit_transform(data["label"])



# Plot histograms for numerical columns in a grid
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
n_cols = 4  # Number of histograms per row
n_rows = -(-len(numerical_columns) // n_cols)  # Calculate rows needed (ceil division)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, n_rows * 4))

for i, column in enumerate(numerical_columns):
    ax = axes[i // n_cols, i % n_cols]
    sns.histplot(data[column], kde=True, bins=50, ax=ax)
    ax.set_title(column)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
