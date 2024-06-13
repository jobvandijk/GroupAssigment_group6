import pandas as pd
import numpy as np
import rdkit
import random as rnd

import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import EarlyStopping
import logging

tf.get_logger().setLevel(logging.ERROR)


def read_file(file_dir):

    input_df = pd.read_csv(file_dir)

    return input_df


def write_file(data, output_file):

    data.to_csv(output_file, index=False)


def format_output(predictions, df, output_cols=[], mol_col=[]):

    smiles_and_vals = df[mol_col+output_cols].copy()

    predict_PKM2 = predictions[0]
    predict_ERK2 = predictions[1]

    prediction_array = np.column_stack((predict_PKM2, predict_ERK2))

    predictions_df = pd.DataFrame(prediction_array, columns=output_cols)

    output_df = pd.concat([smiles_and_vals, predictions_df], axis=1)

    return output_df


def split_data(df, output_cols=[], mol_col=[]):

    descriptor_array = df.drop(columns=mol_col + output_cols).to_numpy()

    outputs_array = df[output_cols].to_numpy()

    return descriptor_array, outputs_array


def line_multiplication_class_balancing(df, dupe_amount):

    ones_filter = (df['PKM2_inhibition'] == 1) | (df['ERK2_inhibition'] == 1)

    filtered_df = df[ones_filter]

    random_selection = np.random.choice(filtered_df.index, size=dupe_amount, replace=True)
    duped_rows = df.loc[random_selection]

    # "balanced"
    balanced_df = pd.concat([df,duped_rows], ignore_index=True)

    return balanced_df


def tensorflow_model(input_shape):

    input_shape = input_shape[1]

    input_layer = Input(shape=input_shape)
    
    first_dense = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
    first_drop = Dropout(0.3)(first_dense)
    
    second_dense = Dense(32, activation='relu', kernel_initializer='he_normal')(first_drop)
    second_drop = Dropout(0.3)(second_dense)
    
    output1 = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='out1')(second_drop)
    output2 = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='out2')(second_drop)

    model = Model(inputs=input_layer, outputs=[output1, output2])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', Precision(), Recall()])

    return model


#def tensorflow_model_v2


def train_model(descriptor_array, outputs_array):

    model = tensorflow_model(descriptor_array.shape)

    descr_train, descr_val, outp_train, outp_val = train_test_split(descriptor_array, outputs_array, test_size=0.2)

    outp_train_split = [outp_train[:,0], outp_train[:,1]]
    outp_val_split = [outp_val[:,0], outp_val[:,1]]

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    training_history = model.fit(descr_train, outp_train_split, epochs=100, batch_size=32, validation_data=(descr_val, outp_val_split),
                                verbose=0, callbacks=[early_stopping])
        
    evaluation = model.evaluate(descr_val, outp_val_split)

    predictions = model.predict(descriptor_array)  

    f1_score_PKM2 = f1_score(outputs_array[:,0], np.round(predictions[0]).astype(int))
    f1_score_ERK2 = f1_score(outputs_array[:,1], np.round(predictions[1]).astype(int))

    print("\n", evaluation, "\n")

    print("Accuracy PKM2: ", evaluation[3])
    print("Val loss PKM2:", evaluation[1])
    print("Precision PKM2: ", evaluation[4])
    print("Recall PKM2: ", evaluation[5])
    print("F1 score PKM2: ", f1_score_PKM2)
    print("Number of 0 in PKM2: ", np.count_nonzero(outputs_array[:,0] == 0), " so percentage of 0 in PKM2 is: ", np.count_nonzero(outputs_array[:,0] == 0)/outputs_array.shape[0])
    print("Number of 1 in PKM2: ", np.count_nonzero(outputs_array[:,0] == 1), " so percentage of 1 in PKM2 is: ", np.count_nonzero(outputs_array[:,0] == 1)/outputs_array.shape[0])
    print("Number of predicted PKM2 1s: ", np.count_nonzero(predictions[0] == 1), "\n")

    print("Accuracy ERK2: ", evaluation[6])
    print("Val loss ERK2: ", evaluation[2])
    print("Precision ERK2: ", evaluation[7])
    print("Recall ERK2: ", evaluation[8])
    print("F1 score ERK2: ", f1_score_ERK2)
    print("Number of 0 in ERK2: ", np.count_nonzero(outputs_array[:,1] == 0), " so percentage of 0 in ERK2 is: ", np.count_nonzero(outputs_array[:,1] == 0)/outputs_array.shape[0])
    print("Number of 1 in ERK2: ", np.count_nonzero(outputs_array[:,1] == 1), " so percentage of 1 in ERK2 is: ", np.count_nonzero(outputs_array[:,1] == 1)/outputs_array.shape[0])
    print("Number of predicted PKM2 1s: ", np.count_nonzero(predictions[1] == 1), "\n")

    print("Number of epochs before early stopping: ", early_stopping.stopped_epoch -early_stopping.patience)

    return predictions 


def main_struct(input_file, output_file):

    df = read_file(input_file)

    dupe_amount = 2000
    output_cols = ['PKM2_inhibition', 'ERK2_inhibition']
    mol_col = ['SMILES']

    balanced_df = line_multiplication_class_balancing(df, dupe_amount)

    # print(balanced_df.head())

    ### For full descriptor file, so with SMILES as column
    # descr_arr, outp_arr = split_data(balanced_df, output_cols, mol_col)
    # predictions = train_model(descr_arr, outp_arr)
    # output_data = format_output(predictions, balanced_df, output_cols)
    # write_file(output_data, output_file)

    ### For data analysis file, so without SMILES as column
    descr_arr, outp_arr = split_data(balanced_df, output_cols)
    predictions = train_model(descr_arr, outp_arr)
    output_data = format_output(predictions, balanced_df, output_cols)
    write_file(output_data, output_file)    


input_file_loc = "tested_molecules.csv"
data_analysis_file = "data_analysis_tested_molecules.csv"
output_file_loc = "neural_network_predictions.csv"
main_struct(data_analysis_file, output_file_loc)