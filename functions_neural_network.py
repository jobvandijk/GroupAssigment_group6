import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
import ast

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import logging

tf.get_logger().setLevel(logging.ERROR)


def read_file(file_dir):

    input_df = pd.read_csv(file_dir)

    return input_df


def format_output(predictions, df, output_cols=[], mol_col=[]):
    # Formats the outputs
    smiles_and_vals = df[mol_col+output_cols].copy()

    pred_col_names = [name+"prediction" for name in output_cols]
    predictions_df = pd.DataFrame(predictions, columns=pred_col_names)

    output_df = pd.concat([smiles_and_vals, predictions_df], axis=1)

    return output_df


def write_file(data, output_file):

    data.to_csv(output_file, index=False)


def split_data(df, output_cols=[], mol_col=[]):
    # Splits the data into a dataframe with only descriptors and a dataframe with only outputs.

    descriptor_array = df.drop(columns=mol_col + output_cols).to_numpy()

    outputs_array = df[output_cols].to_numpy()

    return descriptor_array, outputs_array


def line_multiplication_class_balancing(df, dupe_amount):
    # Randomly selects rows containting at least one 1 value in the PKM2_inhibition and/or the
    # ERK2_inhibition row and duplicates these to add to the database. This artificially increases
    # the number of occurances of 1 in the dataframe.

    ones_filter = (df['PKM2_inhibition'] == 1) | (df['ERK2_inhibition'] == 1)

    filtered_df = df[ones_filter]

    random_selection = np.random.choice(filtered_df.index, size=dupe_amount, replace=True)
    duped_rows = df.loc[random_selection]

    # "balanced"
    balanced_df = pd.concat([df,duped_rows], ignore_index=True)

    return balanced_df


def plot_network_loss_progression(train_history):

    plt.plot(train_history.history['loss'], label="Train Loss")
    plt.plot(train_history.history['val_loss'], label="Val Loss")

    plt.title("Training and validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def balanced_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    sensitivities = []
 
    for i in range(num_classes):
        true_positives = cm[i, i]
        actual_positives = cm[i, :].sum()
        sensitivity = true_positives / actual_positives
        sensitivities.append(sensitivity)
 
    balanced_accuracy = sum(sensitivities) / num_classes
    return balanced_accuracy
 
 
def cm_plot(y_true, y_pred, class_name):
    # Confusion Matrix for PKM2 inhibition
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix for "+ class_name + " inhibition")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(predictions, outp_val, mol_name='', single=False):

    if single==True:
        y_pred = predictions
        y_true = outp_val
        print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
        print("Confusion Matrix:", cm_plot(y_true, y_pred, class_name=mol_name))

    elif single==False:        
        y_pred_PKM2 = predictions[:, 0]
        y_pred_ERK2 = predictions[:, 1]
        
        y_true_PKM2 = outp_val[:,0]
        y_true_ERK2 = outp_val[:,1]  

        print("Balanced Accuracy:", balanced_accuracy_score(y_true_PKM2, y_pred_PKM2))
        print("Balanced Accuracy:", balanced_accuracy_score(y_true_ERK2, y_pred_ERK2))    
        
        
        print("Confusion Matrix:", cm_plot(y_true_PKM2, y_pred_PKM2, class_name='PKM2'))
        print("Confusion Matrix:", cm_plot(y_true_ERK2, y_pred_ERK2, class_name="ERK2"))


def tensorflow_model_individual(input_shape):
    # Creates the model architecture for the method of creating an individual model for
    # each of the 2 outputs.

    model = Sequential()

    model.add(Dense(64, input_dim=input_shape[1], activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

    return model


def train_model_individual(descriptor_array, outputs_array, untested_data, mol_name='', balance_weights=[], plot_loss=False, plot_conf_mat=False):
    # Trains and evaluates the model for the method of creating separate networks for both outputs.    

    model = tensorflow_model_individual(descriptor_array.shape)

    descr_train, descr_val, outp_train, outp_val = train_test_split(descriptor_array, outputs_array, test_size=0.25)    

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    balancing_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(outputs_array), y=outputs_array)
    balancing_weights_dict = dict(enumerate(balancing_weights))

    if balance_weights == True:
        training_history = model.fit(descr_train, outp_train, epochs=100, class_weight=balancing_weights_dict, batch_size=16,
                                    validation_data=(descr_val, outp_val), verbose=0, callbacks=[early_stopping])    

    elif balance_weights == False:
        training_history = model.fit(descr_train, outp_train, epochs=200, batch_size=16, validation_data=(descr_val, outp_val), 
                                    verbose=0, callbacks=[early_stopping])         

    evaluation = model.evaluate(descr_val, outp_val)

    predictions = model.predict(descr_val) 
    predictions = np.round(predictions).astype(int)  

    untested_predictions = model.predict(untested_data)
    untested_predictions = np.round(predictions).astype(int)

    if early_stopping.stopped_epoch == None or early_stopping.stopped_epoch == 0:
        best_epoch = 200
    else:
        best_epoch = early_stopping.stopped_epoch - early_stopping.patience

    if plot_conf_mat == True:
        plot_confusion_matrix(predictions, outp_val, mol_name, single=True)   

    f1_score_val = f1_score(outp_val, np.round(predictions).astype(int))

    print("For model of", mol_name, " with single input single output: \n")

    print("Balancing weights:", balancing_weights_dict, "\n")

    print("Accuracy", mol_name, ": ", evaluation[1])
    print("Val loss", mol_name, ": ", evaluation[0])
    print("Precision", mol_name, ": ", evaluation[2])
    print("Recall", mol_name, ": ", evaluation[3])
    print("F1 score", mol_name, ": ", f1_score_val, "\n")

    print("Number of 0 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 0), " so percentage of 0 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 0)/outputs_array.shape[0])
    print("Number of 1 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 1), " so percentage of 1 in", mol_name, ": ", np.count_nonzero(outputs_array[:] == 1)/outputs_array.shape[0])
    print("Number of predicted 1s for ", mol_name, ": ", np.count_nonzero(predictions == 1), "\n") 

    print("Number of epochs before early stopping: ", best_epoch, "\n")

    if plot_loss == True:
        plot_network_loss_progression(training_history)

    return predictions, untested_predictions


def main_struct_individual(input_train_file, input_test_file, output_file, balance_weights=[], plot_loss=False, MACCS=False, write_output=False, plot_conf_mat=False, seed_val=10):
    # Main structure for the method of individual networks.

    np.random.seed(seed_val)
    tf.random.set_seed(seed_val)

    df = read_file(input_train_file)
    untested_df = read_file(input_test_file)

    output_cols = ['PKM2_inhibition', 'ERK2_inhibition']
    mol_col = ['SMILES']

    if balance_weights[0] == False:
        dupe_amount = balance_weights[1]
        balanced_df = line_multiplication_class_balancing(df, dupe_amount)    
    elif balance_weights[0] == True:
        balanced_df = df

    if MACCS == True:
        maccs_keys_string_to_list = [ast.literal_eval(string_list) for string_list in balanced_df["maccs_keys_bitstring"]]
        descr_arr = np.array(maccs_keys_string_to_list)
        outp_arr = np.array(balanced_df[["PKM2_inhibition", "ERK2_inhibition"]].values.tolist())

        untested_maccs_convert = [ast.literal_eval(string_list) for string_list in untested_df["maccs_keys_bitstring"]]
        untested_descr_arr = np.array(untested_maccs_convert)
        untested_outp_arr = np.array(untested_df[["PKM2_inhibition", "ERK2_inhibition"]].values.tolist())

    elif MACCS == False:
        descr_arr, outp_arr = split_data(balanced_df, output_cols, mol_col)
        untested_descr_arr, untested_outp_arr = split_data(untested_df, output_cols, mol_col)

    pkm2_output = outp_arr[:,0]
    erk2_output = outp_arr[:,1]

    if descr_arr.shape[1] < untested_descr_arr.shape[1]:
        untested_descr_arr = untested_descr_arr[:, :descr_arr.shape[1]]

    predictions_PKM2, untested_predictions_PKM2 = train_model_individual(descr_arr, pkm2_output, untested_descr_arr, mol_name='PKM2', balance_weights=balance_weights[0], plot_loss=plot_loss, plot_conf_mat=plot_conf_mat)
    predictions_ERK2, untested_predictions_ERK2 = train_model_individual(descr_arr, erk2_output, untested_descr_arr, mol_name='ERK2', balance_weights=balance_weights[0], plot_loss=plot_loss, plot_conf_mat=plot_conf_mat)

    predictions = np.column_stack((predictions_PKM2, predictions_ERK2))  
    untested_predictions = np.column_stack((untested_predictions_PKM2, untested_predictions_ERK2)) 

    print("\nNumber of untested PKM2 1 predictions: ", np.count_nonzero(untested_predictions[:,0] == 1))
    print("Number of untested ERK2 1 predictions: ", np.count_nonzero(untested_predictions[:,1] == 1)) 

    output_data = format_output(untested_predictions, untested_df, output_cols)

    if write_output==True:
        write_file(output_data, output_file)


def convert_two_to_four_classes(data):
    # Converts the two outputs of binary data into 4 classes.

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
    # Converts the 4 classes back into the two binary outputs.

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
    # Creates the network architecture for the method of 4 classes.

    model = Sequential()

    model.add(Dense(64, input_dim=input_shape[1], activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

    return model


def train_model_four_classes(descriptor_array, outputs_array, untested_descr, balance_weights=False, plot_loss=False, plot_conf_mat=False):

    n_classes = 4
    outputs_array_cat = to_categorical(outputs_array, num_classes=n_classes)

    model = tensorflow_model_four_classes(descriptor_array.shape, n_classes)

    descr_train, descr_val, outp_train, outp_val = train_test_split(descriptor_array, outputs_array_cat, test_size=0.25)    

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    balancing_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(outputs_array), y=outputs_array)
    balancing_weights_dict = dict(enumerate(balancing_weights))

    if balance_weights == True:
        training_history = model.fit(descr_train, outp_train, epochs=200, class_weight=balancing_weights_dict, batch_size=16,
                                    validation_data=(descr_val, outp_val), verbose=0, callbacks=[early_stopping])  # epochs=100,  
    
    elif balance_weights == False:
        training_history = model.fit(descr_train, outp_train, epochs=200, batch_size=16,
                                    validation_data=(descr_val, outp_val), verbose=0, callbacks=[early_stopping])    

    evaluation = model.evaluate(descr_val, outp_val)

    predictions = model.predict(descr_val)
    class_predictions = np.argmax(predictions, axis=1)

    untested_predictions = model.predict(untested_descr)
    untested_predictions = np.argmax(untested_predictions, axis=1)

    if plot_conf_mat == True:
        outp_val_four_class = np.argmax(outp_val, axis=1)

        two_class_outp_val = convert_four_to_two_classes(outp_val_four_class)
        two_class_pred = convert_four_to_two_classes(class_predictions)

        plot_confusion_matrix(two_class_pred, two_class_outp_val)

    if early_stopping.stopped_epoch == None or early_stopping.stopped_epoch == 0:
        best_epoch = 200
    else:
        best_epoch = early_stopping.stopped_epoch - early_stopping.patience

    f1_score_val = 0.0
    if evaluation[2]+evaluation[3] != 0.0:
        f1_score_val = (2*evaluation[2]*evaluation[3])/(evaluation[2]+evaluation[3])

    print("\nBalancing weights:", balancing_weights_dict, "\n")

    print("Accuracy: ", evaluation[1])
    print("Val loss: ", evaluation[0])
    print("Precision: ", evaluation[2])
    print("Recall: ", evaluation[3])
    print("F1 score: ", f1_score_val, "\n")

    print("Number of 0 in: ", np.count_nonzero(outputs_array[:] == 0), " so percentage of 0: ", np.count_nonzero(outputs_array[:] == 0)/outputs_array.shape[0])
    print("Number of 1 in: ", np.count_nonzero(outputs_array[:] == 1), " so percentage of 1: ", np.count_nonzero(outputs_array[:] == 1)/outputs_array.shape[0])
    print("Number of 2 in: ", np.count_nonzero(outputs_array[:] == 2), " so percentage of 2: ", np.count_nonzero(outputs_array[:] == 2)/outputs_array.shape[0])
    print("Number of 3 in: ", np.count_nonzero(outputs_array[:] == 3), " so percentage of 3: ", np.count_nonzero(outputs_array[:] == 3)/outputs_array.shape[0], "\n")

    print("Number of predicted 0s for: ", np.count_nonzero(class_predictions == 0)) 
    print("Number of predicted 1s for: ", np.count_nonzero(class_predictions == 1)) 
    print("Number of predicted 2s for: ", np.count_nonzero(class_predictions == 2)) 
    print("Number of predicted 3s for: ", np.count_nonzero(class_predictions == 3), "\n")             

    print("Number of epochs before early stopping: ", best_epoch, "\n")

    if plot_loss == True:
        plot_network_loss_progression(training_history)

    return class_predictions, untested_predictions


def main_struct_four_class(input_file, input_test_file, output_file,  balance_weights=[], plot_loss=False, MACCS=False, write_output=False, plot_conf_mat=False, seed_val=10):

    np.random.seed(seed_val)
    tf.random.set_seed(seed_val)

    df = read_file(input_file)
    untested_df = read_file(input_test_file)

    output_cols = ['PKM2_inhibition', 'ERK2_inhibition']
    mol_col = ['SMILES']

    if balance_weights[0]==False:
        dupe_amount = balance_weights[1]
        balanced_df = line_multiplication_class_balancing(df, dupe_amount)   
    elif balance_weights[0]==True:
        balanced_df = df

    if MACCS==True:
        maccs_keys_string_to_list = [ast.literal_eval(string_list) for string_list in df["maccs_keys_bitstring"]]
        descr_arr = np.array(maccs_keys_string_to_list)
        outp_arr = np.array(df[["PKM2_inhibition", "ERK2_inhibition"]].values.tolist())

        maccs_keys_untested = [ast.literal_eval(string_list) for string_list in untested_df["maccs_keys_bitstring"]]
        untested_descr_arr = np.array(maccs_keys_untested)        
    elif MACCS==False:
        descr_arr, outp_arr = split_data(balanced_df, output_cols, mol_col)

        untested_descr_arr, untested_outp = split_data(untested_df, output_cols, mol_col)

    four_class_data = convert_two_to_four_classes(outp_arr)

    if descr_arr.shape[1] < untested_descr_arr.shape[1]:
        untested_descr_arr = untested_descr_arr[:, :descr_arr.shape[1]]

    four_class_predictions, four_class_untested_predictions = train_model_four_classes(descr_arr, four_class_data, untested_descr_arr, balance_weights=balance_weights[0], plot_loss=plot_loss, plot_conf_mat=plot_conf_mat)

    two_class_predictions = convert_four_to_two_classes(four_class_predictions) 
    two_class_untested_pred = convert_four_to_two_classes(four_class_untested_predictions)

    print("\nNumber of untested PKM2 1 predictions: ", np.count_nonzero(two_class_untested_pred[:,0] == 1))
    print("Number of untested ERK2 1 predictions: ", np.count_nonzero(two_class_untested_pred[:,1] == 1))    

    output_data = format_output(two_class_untested_pred, untested_df, output_cols, mol_col)

    if write_output==True:
        write_file(output_data, output_file)



