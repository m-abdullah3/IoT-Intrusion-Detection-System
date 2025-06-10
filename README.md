IoT Intrusion Detection System
A comprehensive machine learning-based intrusion detection system designed specifically for IoT network environments. This project implements and compares multiple ML algorithms to detect various cyber attacks including DDoS, MITM, and malware injection using the CIC IoT Dataset 2023.
Project Overview
This project develops a multi-class classification system capable of identifying various attack types in IoT network traffic. By leveraging machine learning algorithms including Random Forest, Multi-Layer Perceptron (MLP), K-Nearest Neighbors (KNN), and Support Vector Machine (SVM), the system provides robust intrusion detection capabilities tailored for IoT environments.
The implementation includes comprehensive data preprocessing, feature selection, hyperparameter tuning, and performance evaluation to ensure optimal detection accuracy and reliability.
Features

Multi-Algorithm Implementation: Random Forest, MLP, KNN, and SVM classifiers
Comprehensive Data Preprocessing: Duplicate removal, categorical encoding, and feature scaling
Feature Selection: Correlation-based feature selection to reduce noise and improve efficiency
Hyperparameter Optimization: RandomizedSearchCV with cross-validation for optimal performance
Principal Component Analysis (PCA): Dimensionality reduction for improved computational efficiency
Extensive Evaluation Metrics: Accuracy, precision, recall, F1-score, ROC curves, and AUC scores
Visualization: Confusion matrices, ROC curves, and performance comparison charts

Project Structure
iot-intrusion-detection-ml/

├── Data-Analysis.py          # Exploratory data analysis and visualization

├── Preprocessing-Data.py     # Data cleaning and preprocessing pipeline

├── FeatureSelection.py       # Correlation analysis and feature selection

├── PCA.py                   # Principal Component Analysis implementation

├── KNN.py                   # K-Nearest Neighbors classifier

├── MLP.py                   # Multi-Layer Perceptron classifier

├── RandomForest.py          # Random Forest classifier

├── svm.py                   # Support Vector Machine classifier

├── Tune-KNN.py              # KNN hyperparameter tuning

├── Tuning-MLP.py            # MLP hyperparameter tuning
├── Tunning-RF.py            # Random Forest hyperparameter tuning

├── Tunning-SVM.py           # SVM hyperparameter tuning

└── README.md

Installation & Setup Instructions
Prerequisites

Python 3.7+
pip package manager

Dependencies Installation
bashpip install pandas numpy matplotlib seaborn scikit-learn scipy
Dataset Setup

Download the CIC IoT Dataset 2023 from the Canadian Institute for Cybersecurity
Place the dataset file as CICIoT2023_small.csv in the project directory
Update file paths in the preprocessing script as needed

Running the Project

Data Preprocessing:
bashpython Preprocessing-Data.py

Feature Selection:
bashpython FeatureSelection.py

Model Training and Evaluation:
bashpython RandomForest.py
python MLP.py
python KNN.py
python svm.py

Hyperparameter Tuning (Optional):
bashpython Tunning-RF.py
python Tuning-MLP.py
python Tune-KNN.py


Usage Examples
Basic Model Training
python# Load preprocessed data
dataFrame = pd.read_csv("preprocessed_CIC-NEW.csv")

# Feature selection based on correlation analysis
X = dataFrame.drop(['label', 'Max', 'Telnet', 'SMTP', 'IRC', 
                   'Rate', 'Srate', 'Drate', 'SSH'], axis=1)
y = dataFrame["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, min_samples_split=2)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
Hyperparameter Tuning
python# Define parameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy'
)

Key Findings

Random Forest achieved the highest overall performance with 99.2% accuracy
MLP demonstrated strong performance with excellent generalization capabilities
Feature selection reduced dimensionality from 47 to 29 features while maintaining accuracy
PCA analysis showed that 95% of variance could be retained with reduced dimensions

Attack Detection Capabilities
The system successfully detects multiple attack types including:

DDoS attacks
Man-in-the-Middle (MITM) attacks
Malware injection
Various IoT-specific vulnerabilities

Tech Stack / Libraries Used

Python 3.7+: Core programming language
Pandas: Data manipulation and analysis
NumPy: Numerical computing
Scikit-learn: Machine learning algorithms and utilities
Matplotlib & Seaborn: Data visualization
SciPy: Statistical analysis

Machine Learning Algorithms

Random Forest Classifier
Multi-Layer Perceptron (Neural Network)
K-Nearest Neighbors
Support Vector Machine
Principal Component Analysis

License
This project is licensed under the MIT License - see the LICENSE file for details.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
Acknowledgments

Canadian Institute for Cybersecurity (CIC) for the CIC IoT Dataset 2023
The open-source community for the excellent machine learning libraries


Note: This project is designed for educational and research purposes. For production deployment, additional security considerations and real-time processing capabilities should be implemented.
