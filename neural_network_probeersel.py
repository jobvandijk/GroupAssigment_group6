import pandas as pd
import numpy as np
import rdkit

import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import Precision, Recall
import logging

tf.get_logger().setLevel(logging.ERROR)


def read_file(file_dir):

    input_df = pd.read_csv(file_dir)

    return input_df


def split_data(df):

    output_cols = ['PKM2_inhibition', 'ERK2_inhibition']
    mol_col = ['SMILES']

    descriptor_array = df.drop(columns=mol_col + output_cols).to_numpy()

    outputs_array = df[output_cols].to_numpy()

    return descriptor_array, outputs_array


# def train_classifier_network(descriptor_array, outputs_array):

#     y_PKM2 = outputs_array[:,0]
#     y_ERK2 = outputs_array[:,1]

#     network = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam', max_iter=500, random_state=10, 
#                             early_stopping=True, n_iter_no_change=25, momentum=0.9, class_weight='balanced_subsample')

#     kfold = StratifiedKFold(n_splits=10)

#     acc_PKM2 = []
#     acc_ERK2 = []
#     prec_PKM2 = []
#     prec_ERK2 = []
#     recal_PKM2 = []
#     recal_ERK2 = []
#     f1_PKM2 = []
#     f1_ERK2 = []

#     for train_index, val_index in kfold.split(descriptor_array, y_PKM2):
        
#         # First binary descision
#         descr_train, descr_val = descriptor_array[train_index], descriptor_array[val_index]
#         PKM2_train, PKM2_val = y_PKM2[train_index], y_PKM2[val_index]

#         network.fit(descr_train, PKM2_train)

#         predict_PKM2 = network.predict(descr_val)

#         acc_PKM2_prediction = accuracy_score(PKM2_val, predict_PKM2)
#         acc_PKM2.append(acc_PKM2_prediction)

#         prec_PKM2_prediction = precision_score(PKM2_val, predict_PKM2)
#         prec_PKM2.append(prec_PKM2_prediction)

#         recall_PKM2_prediction = recall_score(PKM2_val, predict_PKM2)
#         recal_PKM2.append(recall_PKM2_prediction)

#         f1_PKM2_prediction = f1_score(PKM2_val, predict_PKM2)
#         f1_PKM2.append(f1_PKM2_prediction)

#         # Second binary decision
#         descr_train, descr_val = descriptor_array[train_index], descriptor_array[val_index]
#         ERK2_train, ERK2_val = y_ERK2[train_index], y_ERK2[val_index]

#         network.fit(descr_train, ERK2_train)

#         predict_ERK2 = network.predict(descr_val)

#         acc_ERK2_prediction = accuracy_score(ERK2_val, predict_ERK2)
#         acc_ERK2.append(acc_ERK2_prediction)

#         prec_ERK2_prediction = precision_score(ERK2_val, predict_ERK2)
#         prec_ERK2.append(prec_ERK2_prediction)

#         recall_ERK2_prediction = recall_score(ERK2_val, predict_ERK2)
#         recal_ERK2.append(recall_ERK2_prediction)

#         f1_ERK2_prediction = f1_score(ERK2_val, predict_ERK2)
#         f1_ERK2.append(f1_ERK2_prediction)

#     mean_acc_PKM2 = np.mean(acc_PKM2)
#     mean_prec_PKM2 = np.mean(prec_PKM2)
#     mean_recal_PKM2 = np.mean(recal_PKM2)
#     mean_f1_PKM2 = np.mean(f1_PKM2)

#     mean_acc_ERK2 = np.mean(acc_ERK2)
#     mean_prec_ERK2 = np.mean(prec_ERK2)
#     mean_recal_ERK2 = np.mean(recal_ERK2)
#     mean_f1_ERK2 = np.mean(f1_ERK2)    

#     print("Metrics PRK2:")
#     print("Mean accuracy: ", mean_acc_PKM2)
#     print("Mean precision: ", mean_prec_PKM2)
#     print("Mean recall: ", mean_recal_PKM2)
#     print("Mean f1 score: ", mean_f1_PKM2)

#     print("Metrics ERK2:")
#     print("Mean accuracy: ", mean_acc_ERK2)
#     print("Mean precision: ", mean_prec_ERK2)
#     print("Mean recall: ", mean_recal_ERK2)
#     print("Mean f1 score: ", mean_f1_ERK2)    


def tensorflow_model(input_shape):

    input_shape = input_shape[1]

    input_layer = Input(shape=input_shape)
    
    first_dense = Dense(64, activation='relu')(input_layer)
    first_drop = Dropout(0.3)(first_dense)
    
    second_dense = Dense(32, activation='relu')(first_drop)
    second_drop = Dropout(0.3)(second_dense)
    
    output1 = Dense(1, activation='sigmoid', name='out1')(second_drop)
    output2 = Dense(1, activation='sigmoid', name='out2')(second_drop)

    model = Model(inputs=input_layer, outputs=[output1, output2])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', Precision(), Recall()])

    return model


def train_model(descriptor_array, outputs_array):

    model = tensorflow_model(descriptor_array.shape)

    descr_train, descr_val, outp_train, outp_val = train_test_split(descriptor_array, outputs_array, test_size=0.2)

    outp_train_split = [outp_train[:,0], outp_train[:,1]]
    outp_val_split = [outp_val[:,0], outp_val[:,1]]

    training_history = model.fit(descr_train, outp_train_split, epochs=25, batch_size=16, validation_data=(descr_val, outp_val_split), verbose=0)
        
    evaluation = model.evaluate(descr_val, outp_val_split)

    #print("Validation loss: ", loss)
    # print("Binary validation accuracy: ", bin_acc)
    # print("Validation precision: ", prec)
    # print("Validation recall: ", recal)

    print("\n", evaluation)
    print("\n", "Number of 0 in PKM2: ", np.count_nonzero(outputs_array[:,0] == 0), " so percentage of 0 in PKM2 is: ", np.count_nonzero(outputs_array[:,0] == 0)/outputs_array.shape[0])
    print("Number of 1 in PKM2: ", np.count_nonzero(outputs_array[:,0] == 1), " so percentage of 1 in PKM2 is: ", np.count_nonzero(outputs_array[:,0] == 1)/outputs_array.shape[0])
    print("Number of 0 in ERK2: ", np.count_nonzero(outputs_array[:,1] == 0), " so percentage of 0 in ERK2 is: ", np.count_nonzero(outputs_array[:,1] == 0)/outputs_array.shape[0])
    print("Number of 1 in ERK2: ", np.count_nonzero(outputs_array[:,1] == 1), " so percentage of 1 in ERK2 is: ", np.count_nonzero(outputs_array[:,1] == 1)/outputs_array.shape[0])


def main_struct(input_file):

    df = read_file(input_file)
    
    descr_arr, outp_arr = split_data(df)

    #find_string_columns(outp_arr)

    train_model(descr_arr, outp_arr)


input_file_loc = "tested_molecules.csv"
descriptor_file = r""
main_struct(input_file_loc)