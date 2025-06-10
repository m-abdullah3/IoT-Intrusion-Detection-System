from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split

#Loading the dataset as a pandas frame
dataFrame=pd.read_csv("preprocessed_CIC-13-XXS.csv")

#Printing 1st five rows
print(dataFrame.head())

#seperating the features and the labels
X=dataFrame.drop("label",axis=1)
Y=dataFrame["label"]

#spliting the dataset into 10% validation set
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.10)

#Step 1: Define the SVM model
svm = SVC()

# Step 2: Define the hyperparameter search space
param_dist = {
    'C': [0.1, 1, 10, 100, 1000],                 # Regularization parameter
    'kernel': ['linear', 'poly', 'sigmoid'],  # Kernel types
    'degree': [2, 3, 4],                          # Degree for polynomial kernel
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1], # Kernel coefficient for poly', and 'sigmoid'
    'class_weight': [None, 'balanced']            # Handle class imbalance
}

# Step 3: Perform Randomized Search
random_search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_dist,
    n_iter=50,                 # Number of random combinations to evaluate
    cv=5,                      # 5-fold cross-validation
    scoring='accuracy',        # Evaluation metric
    n_jobs=2,                 # Use all available CPUs
    random_state=42            # For reproducibility
)

# Step 4: Fit the RandomizedSearchCV on the dataset
random_search.fit(X_train, y_train)

# Step 5: Get the best parameters and evaluate performance
print("Best Parameters:", random_search.best_params_)