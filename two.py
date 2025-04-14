# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the dataset
data = pd.read_csv('customer_churn.csv')

# Data Exploration
print(data.describe())

# Handling missing values
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = data['TotalCharges'].astype(float)
data = data.dropna()

# Feature Engineering
data['TotalCharges'] = data['MonthlyCharges'] * data['Tenure']

# One-Hot Encoding for categorical variables
data = pd.get_dummies(data, columns=['ServicePlan'], drop_first=True)

# Splitting the dataset into training and testing sets
X = data.drop(columns=['Churn', 'CustomerID'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)

# Model 2: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# Model 3: Support Vector Machine
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

# Evaluation Metrics
print('Logistic Regression Accuracy:', logreg.score(X_test_scaled, y_test))
print('Random Forest Accuracy:', rf.score(X_test_scaled, y_test))
print('SVM Accuracy:', svm.score(X_test_scaled, y_test))

# Confusion Matrix and Classification Report for Random Forest
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ROC-AUC Score for Random Forest
print('Random Forest ROC-AUC:', roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1]))
