import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
tested_data = pd.read_csv('sorted_tested_molecules.csv')

# Drop rows with NaN values
tested_data = tested_data.dropna()

# Split the data into features and targets for PKM2
X = tested_data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
y_PKM2 = tested_data['PKM2_inhibition']
y_ERK2 = tested_data['ERK2_inhibition']

# Split into training and test sets for PKM2
X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = train_test_split(X, y_PKM2, test_size=0.2, random_state=42)

# Split into training and test sets for ERK2
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = train_test_split(X, y_ERK2, test_size=0.2, random_state=42)

# Train a linear regression model for PKM2
lr_model_PKM2 = LinearRegression()
lr_model_PKM2.fit(X_train_PKM2, y_train_PKM2)

# Train a linear regression model for ERK2
lr_model_ERK2 = LinearRegression()
lr_model_ERK2.fit(X_train_ERK2, y_train_ERK2)

# Evaluate the model for PKM2
y_pred_PKM2 = lr_model_PKM2.predict(X_test_PKM2)
print("PKM2 Linear Regression Metrics")
print("Mean Squared Error:", mean_squared_error(y_test_PKM2, y_pred_PKM2))
print("R2 Score:", r2_score(y_test_PKM2, y_pred_PKM2))

# Evaluate the model for ERK2
y_pred_ERK2 = lr_model_ERK2.predict(X_test_ERK2)
print("ERK2 Linear Regression Metrics")
print("Mean Squared Error:", mean_squared_error(y_test_ERK2, y_pred_ERK2))
print("R2 Score:", r2_score(y_test_ERK2, y_pred_ERK2))


"""
# Prepare the untested data for prediction
untested_data = pd.read_csv('/mnt/data/untested_molecules-3.csv')
X_untested = untested_data.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])

# Make predictions on untested molecules
untested_data['PKM2_inhibition'] = lr_model_PKM2.predict(X_untested)
untested_data['ERK2_inhibition'] = lr_model_ERK2.predict(X_untested)

# Save predictions to a CSV file
untested_data.to_csv('/mnt/data/untested_molecules_predictions.csv', index=False)
"""
