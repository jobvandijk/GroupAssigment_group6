import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdkit
import random as rnd
import ast

import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import logging

tf.get_logger().setLevel(logging.ERROR)


def read_file(file_dir):

    input_df = pd.read_csv(file_dir)

    return input_df


def write_file(data, output_file):

    data.to_csv(output_file, index=False)


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




def tensorflow_model_double_output(input_shape):
    '''Creates model for one input layer two output layers'''

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

def train_model_double_output(descriptor_array, outputs_array):
    '''Trains model for one input layer two output layers'''

    model = tensorflow_model_double_output(descriptor_array.shape)

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

    print("\nFor model with single input double output: \n")

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

def format_output(predictions, df, output_cols=[], mol_col=[]):

    smiles_and_vals = df[mol_col+output_cols].copy()

    predict_PKM2 = predictions[0]
    predict_ERK2 = predictions[1]

    prediction_array = np.column_stack((predict_PKM2, predict_ERK2))

    pred_col_names = [name+"prediction" for name in output_cols]

    predictions_df = pd.DataFrame(prediction_array, columns=pred_col_names)

    output_df = pd.concat([smiles_and_vals, predictions_df], axis=1)

    return output_df

def main_struct_v1(input_file, output_file, data_file=False):
    '''Structure for the model with 1 input layer and 2 output layers. Still does have differences based on which file is used.'''

    df = read_file(input_file)

    dupe_amount = 2000
    output_cols = ['PKM2_inhibition', 'ERK2_inhibition']
    mol_col = ['SMILES']

    balanced_df = line_multiplication_class_balancing(df, dupe_amount)

    # print(balanced_df.head())

    if data_file==False:
        ## For full descriptor file, so with SMILES as column
        descr_arr, outp_arr = split_data(balanced_df, output_cols, mol_col)
    elif data_file==True:
        ## For data analysis file, so without SMILES as column
        descr_arr, outp_arr = split_data(balanced_df, output_cols)

    predictions = train_model_double_output(descr_arr, outp_arr)

    if data_file==False:
        output_data = format_output(predictions, balanced_df, output_cols, mol_col)
    elif data_file==True:
        output_data = format_output(predictions, balanced_df, output_cols)

    write_file(output_data, output_file)     



def tensorflow_model_individual(input_shape):
    '''Creates model for single input layer single output layer'''

    input_shape = input_shape[1]

    input_layer = Input(shape=input_shape)

    first_dense = Dense(64, activation='relu', kernel_initializer='he_normal')(input_layer)
    first_drop = Dropout(0.3)(first_dense)
    
    second_dense = Dense(32, activation='relu', kernel_initializer='he_normal')(first_drop)
    second_drop = Dropout(0.3)(second_dense)

    output_layer = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='out1')(second_drop)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', Precision(), Recall()])

    return model

def train_model_individual(descriptor_array, outputs_array, mol_name):
    '''Trains model for single input layer single output layer'''    

    model = tensorflow_model_individual(descriptor_array.shape)

    descr_train, descr_val, outp_train, outp_val = train_test_split(descriptor_array, outputs_array, test_size=0.2)    

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    balancing_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(outputs_array), y=outputs_array)
    balancing_weights[1] = balancing_weights[1]
    balancing_weights_dict = dict(enumerate(balancing_weights))

    training_history = model.fit(descr_train, outp_train, epochs=100, class_weight=balancing_weights_dict, batch_size=16,
                                validation_data=(descr_val, outp_val), verbose=0, callbacks=[early_stopping])    

    evaluation = model.evaluate(descr_val, outp_val)

    predictions = model.predict(descr_val) 
    #print(predictions)
    predictions = np.round(predictions).astype(int)
    #print(predictions)

    f1_score_val = f1_score(outp_val, np.round(predictions).astype(int))

    print("\nBalancing weights:", balancing_weights_dict, "\n")

    print("For model of", mol_name, " with single input single output: \n")

    print(evaluation, "\n")

    print("Accuracy", mol_name, ": ", evaluation[1])
    print("Val loss", mol_name, ": ", evaluation[0])
    print("Precision", mol_name, ": ", evaluation[2])
    print("Recall", mol_name, ": ", evaluation[3])
    print("F1 score", mol_name, ": ", f1_score_val)
    print("Number of 0 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 0), " so percentage of 0 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 0)/outputs_array.shape[0])
    print("Number of 1 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 1), " so percentage of 1 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 1)/outputs_array.shape[0])
    print("Number of predicted 1s for ", mol_name, ": ", np.count_nonzero(predictions == 1), "\n") 

    print("Number of epochs before early stopping: ", early_stopping.stopped_epoch - early_stopping.patience)

    return predictions       
  
def format_output_v2(predictions_pkm2, predictions_erk2, df, output_cols=[], mol_col=[]):

    smiles_and_vals = df[mol_col+output_cols].copy()

    prediction_array = np.column_stack((predictions_pkm2, predictions_erk2))

    pred_col_names = [name+"prediction" for name in output_cols]

    predictions_df = pd.DataFrame(prediction_array, columns=pred_col_names)

    output_df = pd.concat([smiles_and_vals, predictions_df], axis=1)

    return output_df

def main_struct_v2(input_file, output_file, data_file=False):
    balanced = True

    df = read_file(input_file)

    output_cols = ['PKM2_inhibition', 'ERK2_inhibition']
    mol_col = ['SMILES']

    if balanced == True:
        dupe_amount = 200
        balanced_df = line_multiplication_class_balancing(df, dupe_amount)    
    else:
        balanced_df = df

    if data_file==True:
        descr_arr, outp_arr = split_data(balanced_df, output_cols)  
    else:
        descr_arr, outp_arr = split_data(balanced_df, output_cols, mol_col)

    pkm2_output = outp_arr[:,0]
    erk2_output = outp_arr[:,1]

    predictions_PKM2 = train_model_individual(descr_arr, pkm2_output, 'PKM2')
    predictions_ERK2 = train_model_individual(descr_arr, erk2_output, 'ERK2')

    output_data = format_output_v2(predictions_PKM2, predictions_ERK2, balanced_df, output_cols)
    write_file(output_data, output_file)



def convert_two_to_four_classes(data):

    four_classing = np.zeros(data.shape[0])

    for combi in range(data.shape[0]):
        
        output_row = data[combi]

        if np.array_equal(output_row, [0,0]):
            four_classing[combi] = 0
        elif np.array_equal(output_row, [1,0]):
            four_classing[combi] = 1
        elif np.array_equal(output_row, [0,1]):
            four_classing[combi] = 2
        elif np.array_equal(output_row, [1,1]):
            four_classing[combi] = 3

    return four_classing

def convert_four_to_two_classes(data):

    two_classing = np.zeros((data.shape[0], 2))

    for row in range(data.shape[0]):
        
        output_value = data[row]

        if output_value == 0:
            two_classing[row,:] = [0,0]
        elif output_value == 1:          
            two_classing[row,:] = [1,0]
        elif output_value == 2:         
            two_classing[row,:] = [0,1]
        elif output_value == 3:         
            two_classing[row,:] = [1,1]      

    return two_classing

def tensorflow_model_four_classes(input_shape, n_classes):

    model = Sequential()

    model.add(Dense(64, input_dim=input_shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

    return model

def plot_loss_functions(train_history):

    plt.plot(train_history.history['loss'], label="Train Loss")
    plt.plot(train_history.history['val_loss'], label="Val Loss")

    plt.title("Training and validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model_four_classes(descriptor_array, outputs_array):

    n_classes = 4
    outputs_array_cat = to_categorical(outputs_array, num_classes=n_classes)

    model = tensorflow_model_four_classes(descriptor_array.shape, n_classes)

    descr_train, descr_val, outp_train, outp_val = train_test_split(descriptor_array, outputs_array_cat, test_size=0.2)    

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    balancing_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(outputs_array), y=outputs_array)
    balancing_weights_dict = dict(enumerate(balancing_weights))

    training_history = model.fit(descr_train, outp_train, epochs=200, class_weight=balancing_weights_dict, batch_size=16,
                                validation_data=(descr_val, outp_val), verbose=0, callbacks=[early_stopping])  # epochs=100,  

    evaluation = model.evaluate(descr_val, outp_val)

    predictions = model.predict(descriptor_array) 

    class_predictions = np.argmax(predictions, axis=1)

    print("\nBalancing weights:", balancing_weights, "\n")
    print("\nBalancing weights dict: ", balancing_weights_dict, "\n")

    if early_stopping.stopped_epoch == None or early_stopping.stopped_epoch == 0:
        stopped_at = 120
        print("Wrong stopped: ", early_stopping.stopped_epoch, "\n")
    else:
        stopped_at = early_stopping.stopped_epoch

    #f1_score_val = f1_score(outputs_array, np.round(predictions).astype(int))
    f1_score_val = (2*evaluation[2]*evaluation[3])/(evaluation[2]+evaluation[3])

    print("\nEvaluation: ", evaluation, "\n")

    print("Accuracy: ", evaluation[1])
    print("Val loss: ", evaluation[0])
    print("Precision: ", evaluation[2])
    print("Recall: ", evaluation[3])
    print("F1 score: ", f1_score_val)
    print("Number of 0 in: ", np.count_nonzero(outputs_array[:] == 0), " so percentage of 0: ", np.count_nonzero(outputs_array[:] == 0)/outputs_array.shape[0])
    print("Number of 1 in: ", np.count_nonzero(outputs_array[:] == 1), " so percentage of 1: ", np.count_nonzero(outputs_array[:] == 1)/outputs_array.shape[0])
    print("Number of 2 in: ", np.count_nonzero(outputs_array[:] == 2), " so percentage of 2: ", np.count_nonzero(outputs_array[:] == 2)/outputs_array.shape[0])
    print("Number of 3 in: ", np.count_nonzero(outputs_array[:] == 3), " so percentage of 3: ", np.count_nonzero(outputs_array[:] == 3)/outputs_array.shape[0], "\n")
    print("Number of predicted 1s for: ", np.count_nonzero(class_predictions == 1)) 
    print("Number of predicted 2s for: ", np.count_nonzero(class_predictions == 2)) 
    print("Number of predicted 3s for: ", np.count_nonzero(class_predictions == 3), "\n")             

    print("Number of epochs before early stopping: ", stopped_at - early_stopping.patience)

    return class_predictions

def main_struct_four_class(input_file, output_file, data_file=False, balance_weights=False):

    df = read_file(input_file)

    output_cols = ['PKM2_inhibition', 'ERK2_inhibition']
    mol_col = ['SMILES']

    if balance_weights==True:
        dupe_amount = 1000
        balanced_df = line_multiplication_class_balancing(df, dupe_amount)    
    elif balance_weights==False:
        balanced_df = df

    if data_file==True:
        descr_arr, outp_arr = split_data(balanced_df, output_cols)  
    
    elif data_file==False:
        descr_arr, outp_arr = split_data(balanced_df, output_cols, mol_col)

    four_class_data = convert_two_to_four_classes(outp_arr)

    predictions = train_model_four_classes(descr_arr, four_class_data)

    two_class_data = convert_four_to_two_classes(predictions)




def format_output_v4(df, predictions, mol_col=[], output_cols=[]):
    
    smiles_and_vals = df[mol_col+output_cols].copy()

    pred_col_names = [name+"prediction" for name in output_cols]

    predictions_df = pd.DataFrame(predictions, columns=pred_col_names)

    output_df = pd.concat([smiles_and_vals, predictions_df], axis=1)  

    return output_df  

def main_struct_v4(input_file, output_file, four_class, balance, data_file=False):  

    df = read_file(input_file)

    maccs_keys_string_to_list = [ast.literal_eval(string_list) for string_list in df["maccs_keys_bitstring"]]
    descr_array = np.array(maccs_keys_string_to_list)
    outp_array = np.array(df[["PKM2_inhibition", "ERK2_inhibition"]].values.tolist())

    if balance == True:
        dupe_amount=200
        balanced_df = line_multiplication_class_balancing(df, dupe_amount)    

        balanced_list = [ast.literal_eval(string_list) for string_list in balanced_df["maccs_keys_bitstring"]]
        descr_array = np.array(balanced_list)
        outp_array = np.array(balanced_df[["PKM2_inhibition", "ERK2_inhibition"]].values.tolist())
    else:
        balanced_df = df

    if four_class==True:
        four_class_data = convert_two_to_four_classes(outp_array)
        predictions = train_model_four_classes(descr_array, four_class_data)
        two_class_data = convert_four_to_two_classes(predictions) 
        output_df = format_output_v4(balanced_df, two_class_data, ["SMILES"], ["PKM2_inhibition", "ERK2_inhibition"])
        write_file(output_df, output_file)

        return two_class_data, balanced_df

    else:
        pkm2_output = outp_array[:,0]
        erk2_output = outp_array[:,1]

        predictions_PKM2 = train_model_individual(descr_array, pkm2_output, 'PKM2')
        predictions_ERK2 = train_model_individual(descr_array, erk2_output, 'ERK2')

        prediction_array = np.column_stack((predictions_PKM2, predictions_ERK2))

        return prediction_array, balanced_df





input_file_loc = "tested_molecules.csv"
data_analysis_file = "tested_molecular_desc.csv"
test_set_macc_keys = "Test set with MACCS keys.csv"
train_set_macc_keys = "Train set with MACCS keys.csv"
output_file_loc = "neural_network_predictions.csv"

#main_struct_v2(data_analysis_file, output_file_loc)

#main_struct_v4(train_set_macc_keys, output_file_loc, four_class=False, balance=False)

main_struct_v4(train_set_macc_keys, output_file_loc, four_class=True, balance=False)