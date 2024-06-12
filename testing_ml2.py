import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('sorted_tested_molecules.csv')

# Drop rows with NaN values
data = data.dropna()
# computing number of rows
rows = len(data.axes[0])

# computing number of columns
cols = len(data.axes[1])
print(f"rows:{rows}, cols:{cols}")

# Split the data into features and targets
X = data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
y_PKM2 = data['PKM2_inhibition']
y_ERK2 = data['ERK2_inhibition']

# Split into training and test sets
X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = train_test_split(X, y_PKM2, test_size=0.2, stratify=y_PKM2, random_state=42)
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = train_test_split(X, y_ERK2, test_size=0.2, stratify=y_ERK2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_PKM2, y_train_PKM2 = smote.fit_resample(X_train_PKM2, y_train_PKM2)
X_train_ERK2, y_train_ERK2 = smote.fit_resample(X_train_ERK2, y_train_ERK2)

# Define a Random Forest model
rf_model_PKM2 = RandomForestClassifier(n_estimators=30,random_state=42)
rf_model_ERK2 = RandomForestClassifier(n_estimators=30,random_state=42)

# Train the model for PKM2
rf_model_PKM2.fit(X_train_PKM2, y_train_PKM2)

# Train the model for ERK2
rf_model_ERK2.fit(X_train_ERK2, y_train_ERK2)

# Evaluate the model for PKM2
y_pred_PKM2 = rf_model_PKM2.predict(X_test_PKM2)
print("PKM2 Classification Report")
print(classification_report(y_test_PKM2, y_pred_PKM2))
print("PKM2 ROC AUC Score:", roc_auc_score(y_test_PKM2, y_pred_PKM2))

# Evaluate the model for ERK2
y_pred_ERK2 = rf_model_ERK2.predict(X_test_ERK2)
print("ERK2 Classification Report")
print(classification_report(y_test_ERK2, y_pred_ERK2))
print("ERK2 ROC AUC Score:", roc_auc_score(y_test_ERK2, y_pred_ERK2))

# Load untested molecules
untested_molecules = pd.read_csv('untested_molecules-3.csv')
X_untested = untested_molecules.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])

# Make predictions on untested molecules
untested_molecules['PKM2_inhibition_predict'] = rf_model_PKM2.predict(X_untested)
untested_molecules['ERK2_inhibition_predict'] = rf_model_ERK2.predict(X_untested)

# Save predictions to a CSV file
untested_molecules.to_csv('untested_molecules-5.csv', index=False)
