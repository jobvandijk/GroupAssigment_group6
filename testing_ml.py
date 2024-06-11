import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('/mnt/data/tested_molecular_desc.csv')

# Split the data into features and targets
X = data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
y_PKM2 = data['PKM2_inhibition']
y_ERK2 = data['ERK2_inhibition']

# Split into training and test sets
X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = train_test_split(X, y_PKM2, test_size=0.2, stratify=y_PKM2, random_state=42)
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = train_test_split(X, y_ERK2, test_size=0.2, stratify=y_ERK2, random_state=42)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_PKM2, y_train_PKM2 = smote.fit_resample(X_train_PKM2, y_train_PKM2)
X_train_ERK2, y_train_ERK2 = smote.fit_resample(X_train_ERK2, y_train_ERK2)

# Define a model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search_PKM2 = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
grid_search_PKM2.fit(X_train_PKM2, y_train_PKM2)
best_model_PKM2 = grid_search_PKM2.best_estimator_

grid_search_ERK2 = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1)
grid_search_ERK2.fit(X_train_ERK2, y_train_ERK2)
best_model_ERK2 = grid_search_ERK2.best_estimator_

# Evaluate the model
y_pred_PKM2 = best_model_PKM2.predict(X_test_PKM2)
y_pred_ERK2 = best_model_ERK2.predict(X_test_ERK2)

print("PKM2 Classification Report")
print(classification_report(y_test_PKM2, y_pred_PKM2))
print("PKM2 ROC AUC Score:", roc_auc_score(y_test_PKM2, y_pred_PKM2))

print("ERK2 Classification Report")
print(classification_report(y_test_ERK2, y_pred_ERK2))
print("ERK2 ROC AUC Score:", roc_auc_score(y_test_ERK2, y_pred_ERK2))

# Load untested molecules
untested_molecules = pd.read_csv('/mnt/data/untested_molecules.csv')
X_untested = untested_molecules.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])

# Make predictions on untested molecules
untested_molecules['PKM2_inhibition'] = best_model_PKM2.predict(X_untested)
untested_molecules['ERK2_inhibition'] = best_model_ERK2.predict(X_untested)

# Save predictions to a CSV file
untested_molecules.to_csv('/mnt/data/untested_molecules_predictions.csv', index=False)

