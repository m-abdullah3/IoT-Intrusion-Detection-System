from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split


#Loading the dataset as a pandas frame
dataFrame=pd.read_csv("preprocessed_CIC-NEW.csv")

#Printing 1st five rows
print(dataFrame.head())

#seperating the features and the labels
X=dataFrame.drop("label",axis=1)
Y=dataFrame["label"]
#spliting the dataset into 10% validation set
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.10)



#Define the Random Forest model
rf = RandomForestClassifier()

#Define the hyperparameter space for Random Search
param_dist = {
    'n_estimators': [100, 200, 300],       # Number of trees
    'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required at each leaf node
    'max_features': ['sqrt', 'log2', None],   # Number of features to consider for the best split
}

#Perform Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,                 # Number of parameter settings to sample
    cv=5,                      # 5-fold cross-validation
    scoring='accuracy',        # Evaluation metric
    n_jobs=2,                 # Use 2 CPUs
    random_state=42            # For reproducibility
    ,verbose=2
)

#Fitting the random search to find the best result through cross validation
#k=5 folds
random_search.fit(X_train, y_train)

# Step 6: Get the best parameters
print("Best Parameters:", random_search.best_params_)


