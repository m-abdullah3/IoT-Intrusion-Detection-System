from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from random import randint

#Loading the dataset as a pandas frame
dataFrame=pd.read_csv("preprocessed_CIC-NEW.csv")

#Printing 1st five rows
print(dataFrame.head())

#seperating the features and the labels
X=dataFrame.drop("label",axis=1)
Y=dataFrame["label"]

#spliting the dataset into 10% validation set
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.10)

#Define the SVM model
model = KNeighborsClassifier()

#Define the hyperparameter search space
param_dist = {
    'n_neighbors': [5,10,15],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute nearest neighbors
    'leaf_size': [10,20,40],  # Leaf size passed to BallTree or KDTree
}


#Perform Randomized Search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,                 # Number of random combinations to evaluate
    cv=5,                      # 5-fold cross-validation
    scoring='accuracy',        # Evaluation metric
    n_jobs=2,                 # Use 2  CPUs
    random_state=42            # For reproducibility
    ,verbose=2
)

#Fitting the random search to find the best result through cross validation
#k=5 folds
random_search.fit(X_train, y_train)

#Get the best parameters
print("Best Parameters:", random_search.best_params_)