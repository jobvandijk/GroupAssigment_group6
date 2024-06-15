import pandas as pd
from sklearn.datasets    import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the data
file_path = r"sorted_tested_molecules.csv"
df = pd.read_csv(file_path)
df_smiles = pd.read_csv(r"tested_molecules_original.csv")
df["SMILES"] = df_smiles["SMILES"]
#print(df)
#print (df.isna().any(axis=1))
df1 = df[df.isna().any(axis=1)]
#print (df1)


# Define the features and target variables
X = df.drop(columns=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])
y = df[['PKM2_inhibition', 'ERK2_inhibition']]

# Convert data to float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

kmeans = KMeans(n_clusters=20,random_state=0)
y_pred = kmeans.fit_predict(X)

df['Cluster'] = y_pred


results_df = pd.DataFrame({
    'Molecule': df['SMILES'],
    'PKM2_actual': df['PKM2_inhibition'].values,
    'ERK2_actual': df['ERK2_inhibition'].values,
    'Cluster': y_pred
})
print(results_df)
print(df)

plt.scatter(results_df['PKM2_actual'], results_df['ERK2_actual'], c=results_df['Cluster'], cmap='viridis')
plt.xlabel('PKM2 Inhibition')
plt.ylabel('ERK2 Inhibition')
plt.title('K-means Clustering of Molecules based on Kinase Inhibition')
plt.show()




