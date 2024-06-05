import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

dframe = pd.read_csv('tested_molecules.csv')
Melanoma = np.array(dframe['melanoma'])
outfile = 'group2022_08_scores.csv'
dframe = pd.read_csv(outfile)

featuresAsymmetry = np.array(dframe['asymmetry'])
featureCompactness = np.array(dframe['compactness'])
featureTexture = np.array(dframe['texture'])

axs = util.scatter_data(featureTexture, featuresAsymmetry, Melanoma)
axs.set_xlabel('X1 = Asymmetry')
axs.set_ylabel('X2 = Compactness')
axs.legend()
# define K's that are tested on the validation set and the number of the current fold
Validation_K = [3]
curr_fold = 0
# load features
X = dframe.iloc[:, 1:].to_numpy()
# load labels
y = Melanoma

# split dataset into 5 different dataset folds for cross-validation
kf = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=1)
# predict labels for each fold using the KNN algorithm
specifity = []
sensitivity = []
for train_index, test_val_index in kf.split(X, y):
    # define accuracy score and predictions for test set
    Acc_Score = 0
    y_pred_test = 0
    # split dataset into a train, validation and test dataset
    test_index, val_index = np.split(test_val_index, 2)
    X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
    # generate predictions using knn_classifier for every K
    for K in Validation_K:
        y_pred_val, y_pred_test_curr = util.knn_classifier(
            X_train, y_train, X_val, X_test, K)
        Curr_Acc = accuracy_score(y_val, y_pred_val)
        # if accuracy of the predictions on the validation set is larger than the current accuracy, save predictions
        # for test set
        if Curr_Acc > Acc_Score:
            Acc_Score = Curr_Acc
            y_pred_test = y_pred_test_curr
    # add 1 to the number of the current fold and print the accuracy on the test set for the current fold
    curr_fold += 1
    test_acc = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    specifity.append((confusion[0][0] / (confusion[0][0] + confusion[0][1])))
    sensitivity.append((confusion[1][1] / (confusion[1][1] + confusion[1][0])))
    print('accuracy of predictions on test set of fold ' +
          str(curr_fold) + ': ' + str(test_acc))
    print(confusion)
    print("specificiteit:",
          confusion[0][0] / (confusion[0][0] + confusion[0][1]),
          "sensitiviteit:",
          confusion[1][1] / (confusion[1][1] + confusion[1][0]), '\n')

average_specifity = sum(specifity) / len(specifity)
average_sensitivity = sum(sensitivity) / len(sensitivity)
print('the average specifity is', average_specifity)
print('the average sensitivity is', average_sensitivity)


def knn_classifier(X_train, y_train, X_validation, X_test, k):
    # returns the labels for test_data, predicted by the k-NN clasifier trained on X_train and y_train
    # input:
    # X_train - num_train x num_features matrix with features for the training data
    # y_train - num_train x 1 vector with labels for the training data
    # X_validation - num_test x num_features matrix with features for the validation data
    # X_test - num_test x num_features matrix with features for the test data
    # k - Number of neighbors to take into account
    # output:
    # y_pred_validation - num_test x 1 predicted vector with labels for the validation data
    # y_pred_test - num_test x 1 predicted vector with labels for the test data
    # bron: https://github.com/tueimage/8qa01/blob/master/code/class2022_group00_functions.py

    X_test_val = np.vstack((X_validation, X_test))

    # compute standardized euclidian distance of validation and test points to the other points
    D = cdist(X_test_val, X_train, metric='seuclidean')

    # sort distances per row and return array of indices from low to high
    sort_ix = np.argsort(D, axis=1)

    # get the k smallest distances
    sort_ix_k = sort_ix[:, :k]
    predicted_labels = y_train[sort_ix_k]

    # predictions for each point is the mode of the K labels closest to the point
    predicted_labels = mode(predicted_labels, axis=1)[0]
    y_pred_validation = predicted_labels[:len(X_validation)]
    y_pred_test = predicted_labels[len(X_validation):]

    return y_pred_validation, y_pred_test