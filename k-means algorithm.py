import pandas as pd
from sklearn.datasets    import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split


# Load the data
file_path = r"C:\Users\20223095\OneDrive - TU Eindhoven\Documents\GitHub\GroupAssigment_group6\tested_molecular_desc.csv"
df = pd.read_csv(file_path)
print (df.isna().any(axis=1))
df1 = df[df.isna().any(axis=1)]
print (df1)


# Define the features and target variables
X = df.drop(columns=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])
y = df[['PKM2_inhibition', 'ERK2_inhibition']]

# Convert data to float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

kmeans = KMeans(n_clusters=20,random_state=0)
y_pred = kmeans.fit_predict(X_train)

results_df = pd.DataFrame({
    'Molecule': df.loc[y_pred.index, 'SMILES'],
    'PKM2_actual': df['PKM2_inhibition'].values,
    'ERK2_actual': df['ERK2_inhibition'].values,
    'Cluster': y_pred
})
df['Cluster'] = y_pred
print(results_df.head())


#voorbeeld dataframe
X, _ = make_blobs(n_samples=10, centers=3, n_features=4)
df = pd.DataFrame(X, columns=['Feat_1', 'Feat_2', 'Feat_3', 'Feat_4'])

kmeans = KMeans(n_clusters=3,random_state=0)

y = kmeans.fit_predict(df[['Feat_1', 'Feat_2', 'Feat_3', 'Feat_4']])
df['Cluster'] = y


y_other = kmeans.fit_predict(df.iloc[:,:4])


df['Other'] = y_other

print(df.head())

