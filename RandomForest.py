import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import seaborn as sns


# Loading the dataset as a pandas frame
dataFrame = pd.read_csv("preprocessed_CIC-NEW.csv")

# Printing 1st five rows
print(dataFrame.head().to_string())

# Separating the features and the labels
#Selecting Features based on heatmpa analysis
X = dataFrame.drop([ 'label',"Max",'Telnet','SMTP', 'IRC','Rate','Srate', 'Drate','SSH','flow_duration','ece_flag_number', 'cwr_flag_number','LLC','urg_count','syn_count',"DHCP","ARP","IPv"],axis=1)
Y = dataFrame["label"]

# Splitting the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10,train_size=0.10)

#Adding name of the model to a variable for later use
name_Model="Random Forest"

# Creating and fitting the random forest model
model = RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=1,max_features=None,n_jobs=2)
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Calculating the performance metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_score=metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)

#Making a list of encoded labels
labels=list(range(34))
#Calculting the confusion matrix
confusionMatrix = metrics.confusion_matrix(y_test, y_pred,labels=labels)

# Plot the confusion matrix
sns.heatmap(confusionMatrix, annot=True,fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
plt.ylabel('Prediction',fontsize=11)
plt.xlabel('Actual',fontsize=11)
plt.title(f'Confusion Matrix for {name_Model}',fontsize=14)
plt.show()


#Initialize dictionaries to store TruePositive, FalsePositive, TrueNegative, FalseNegatives for each class
TruePositive = {}
FalsePositive = {}
TrueNegative = {}
FalseNegatives = {}

#Calculate TruePositive, FalsePositive, TrueNegative, FalseNegatives for each class
num_classes = len(np.unique(y_test))  # Total number of classes (e.g., 8)
for i in range(num_classes):
    TruePositive[i] = confusionMatrix[i, i]  # True positives are the diagonal elements
    FalsePositive[i] = np.sum(confusionMatrix[:, i]) - confusionMatrix[i, i]  # Sum of column minus the diagonal (false positives)
    FalseNegatives[i] = np.sum(confusionMatrix[i, :]) - confusionMatrix[i, i]  # Sum of row minus the diagonal (false negatives)
    TrueNegative[i] = np.sum(confusionMatrix) - (TruePositive[i] + FalsePositive[i] + FalseNegatives[i])  # Total sum minus TP, FP, FN (true negatives)

#Sum the TruePositive, FalsePositive, TrueNegative, FalseNegatives across all classes
total_TruePositive = sum(TruePositive.values())
total_FalsePositive = sum(FalsePositive.values())
total_TrueNegative = sum(TrueNegative.values())
total_FalseNegatives = sum(FalseNegatives.values())



# Storing the performance metrics in a dictionary
results = [{
    'Model': name_Model,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score':f1_score,
    'True Negative': total_TrueNegative,
    'True Positive': total_TruePositive,
    'False Positive': total_FalsePositive,
    'False Negative': total_FalseNegatives,
}]

# Converting the results list into a DataFrame
resultsDataFrame = pd.DataFrame(results)

print("-----------------------------------------------------------------------------")
# Printing the DataFrame containing the results by converting into a string
print(resultsDataFrame.to_string(index=True, max_rows=None, max_cols=None, line_width=None, float_format='{:,.4f}'.format, header=True))

# Setting the x values for the accuracy bar
xValues = [name_Model]
yValues = [accuracy]

# Plotting the accuracy bar
plt.bar(xValues, yValues)

# Setting up x and y labels
plt.xlabel("Model")
plt.ylabel("Accuracy")

# Setting up the title
plt.title(f'Accuracy for {name_Model}')
plt.show()

# Getting the need information to plot ROC Curves and the AUC for each class
# Binarize the labels for each class using one-vs-rest
y_test_bin = label_binarize(y_test, classes=labels)

# Get the probabilities predicted for each class
y_probs = model.predict_proba(X_test)

#Initialize dictionaries to store the information need to plot the ROC curve
#False positive rate
fpr = dict()
#True Positive rate
tpr = dict()
#The area under the curve for the ROC curve
roc_auc = dict()

#Compute the ROC curve for each class and their resulting AUC
for i in range(len(labels)):  # Loop through each class
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Settup the figure for the ROC curve for each class
plt.figure(figsize=(20, 15))

# Plot all ROC curves
for i in range(len(labels)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plot a diagonal line representing the dummy classifier
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')


#adding the scale for the x axis
plt.xlim([0.0, 1.0])
#adding the scale for y axis
plt.ylim([0.0, 1.05])
#adding label for the x axis
plt.xlabel('False Positive Rate')
#adding label for the y axis
plt.ylabel('True Positive Rate')
#Setting the title of the ROC plot
plt.title(f'Receiver Operating Characteristic (ROC) Curve - {name_Model}')
#setting the location for the legend of the ROC curve
plt.legend(loc="lower right",fontsize=6.8)

plt.show()