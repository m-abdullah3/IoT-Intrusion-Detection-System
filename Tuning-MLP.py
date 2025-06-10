from sklearn.neural_network import MLPClassifier
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


#Define the MLPClassifier model
mlp = MLPClassifier()

#Define the hyperparameter search space
param_dist = {
    'hidden_layer_sizes': [(100,100),(200,200), (300,300,300)],  # Number of neurons in hidden layers
    'activation': ['relu', 'tanh',"logistic"],                             # Activation functions
    'solver': ['adam', 'sgd'],                                              # Optimization algorithms
    'alpha': [0.0001, 0.001, 0.01, 0.1],                                    # L2 regularization term
    'learning_rate': ['constant', 'adaptive'],# Learning rate schedule
'learning_rate_init': [0.001, 0.01, 0.1],

}

#Perform Randomized Search
random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_dist,
    n_iter=50,                 # Number of random combinations to evaluate
    cv=5,                      # 5-fold cross-validation
    scoring='accuracy',        # Evaluation metric
    n_jobs=2,                 # Use 2 CPUs
    random_state=42            # For reproducibility
    ,verbose=2
)

#Fitting the random search to find the best result through cross validation
#k=5 folds
random_search.fit(X_train, y_train)

#Get the best parameters
print("Best Parameters:", random_search.best_params_)

