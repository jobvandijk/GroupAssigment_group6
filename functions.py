import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



df = pd.read_csv('tested_molecules.csv')
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = [Descriptors.descList[i][1](mol) for i in range(len(Descriptors.descList))]
    return descriptors

# Compute descriptor values for each SMILES string and add as new columns
descriptor_columns = [descriptor[0] for descriptor in Descriptors.descList]
df[descriptor_columns] = df['SMILES'].apply(lambda x: pd.Series(compute_descriptors(x)))

# Load your dataset
# Replace 'your_dataset.csv' with the actual path to your dataset


# Separate features (descriptors) and target (PKM2_inhibition)
X = df.drop(columns=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])
y = df['PKM2_inhibition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)