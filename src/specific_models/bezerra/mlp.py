import functools
import time
import math
import tensorflow as tf
from tensorflow import keras
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

STATES = [0, 10, 100, 1000, 10000]


DATASET_DIR = '../../../datasets/Dataset-IoT/'
NETFLOW_DIRS = ['MC/NetFlow/', 'SC/NetFlow/', 'ST/NetFlow/']


# MC_I_FIRST: Has infected data by Hajime, Aidra and BashLite botnets'
# MC_I_SECOND: Has infected data from Mirai botnets
# MC_I_THIR: Has infected data from Mirai, Doflo, Tsunami and Wroba botnets
# MC_L: Has legitimate data, no infection


path_types = ['MC', 'SC', 'ST']
data_set_files = [ [r'MC_I{}.csv'.format(index) for index in range(1, 4)],
                   [r'SC_I{}.csv'.format(index) for index in range(1, 4)],
                   [r'ST_I{}.csv'.format(index) for index in range(1, 4)] ]

for path, files in zip(path_types, data_set_files):
    files.append(path + '_L.csv')

################
##reading data##
################

for n, (path, files) in enumerate(zip(NETFLOW_DIRS, data_set_files), start=1):
    for csvFile in files:
        if n == 1:
            df = pd.read_csv(DATASET_DIR + path + csvFile)
        else:
            aux_df = pd.read_csv(DATASET_DIR + path + csvFile)
            df = pd.concat([df, aux_df], ignore_index=True)

print ("Data Loaded")

#making the final DataFrame
#dropping the number of the rows column
df = df.sample (frac=1, replace=True, random_state=0)
df = df.drop(df.columns[0], axis=1)

#dropping bad columns
nUniques = df.nunique()
for column, nUnique in zip (df.columns, nUniques):
    if(nUnique == 1):
        df.drop(axis='columns', columns=column, inplace=True)

#dropping unrelated columns
df.drop(axis='columns', columns=['ts', 'te', 'sa', 'da'], inplace=True)
#counting different labels
neg, pos = np.bincount(df['Label'])
##################################
## encoding categorical columns ##
##################################

from sklearn import preprocessing

cat_cols, num_cols = df.columns[df.dtypes == 'O'], df.columns[df.dtypes != 'O']
num_cols = num_cols[1:]

categories = [df[column].unique() for column in df[cat_cols]]

categorical_encoder = preprocessing.OrdinalEncoder(categories=categories)
categorical_encoder.fit(df[cat_cols])
df[cat_cols] = categorical_encoder.transform(df[cat_cols])

#########################
## Splitting the Data  ##
#########################

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

for state in STATES:

    train, test = train_test_split (df, test_size=0.2, random_state=state)
    train, val = train_test_split (train, test_size=0.2, random_state=state)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    train_labels = np.array (train.pop('Label'))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val.pop('Label'))
    test_labels = np.array (test.pop('Label'))

    train_features = np.array(train)
    val_features = np.array(val)
    test_features = np.array(test)

    ###########################
    ## Normalizing the Data  ##
    ###########################

    #getting the index of the numerical columns
    index = [df.columns.get_loc(c)-1 for c in num_cols]
    index = np.array(index)

    cat_index = [df.columns.get_loc(c) for c in cat_cols]
    cat_index = np.array(index)

    scaler = StandardScaler()
    train_features[:, index] = scaler.fit_transform(train_features[:, index])

    val_features[:, index] = scaler.transform(val_features[:, index])

    test_features[:, index] = scaler.transform(test_features[:, index])

    train_features[:, index] = np.clip(train_features[:, index], -5, 5)
    val_features[:, index] = np.clip(val_features[:, index], -5, 5)
    test_features[:, index] = np.clip(test_features[:, index], -5, 5)

    ###########################
    ## Creating the Model    ##
    ###########################

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]


    def make_model(metrics = METRICS, output_bias=None, hidden_layer_size=32, lr=1e-3, dropout_rate=0.0):

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        model = keras.Sequential([
            keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(train_features.shape[-1],)),#, kernel_initializer=initializer),
            keras.layers.Dense(hidden_layer_size, activation='relu'),#, kernel_initializer=initializer),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(hidden_layer_size, activation='relu'),#, kernel_initializer=initializer),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)#, kernel_initializer=initializer)

        ])

        model.compile(
            optimizer=keras.optimizers.Adam(lr=lr),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

        return model

    #dealing with the imbalanced data through class weights

    total = pos + neg
    weight_for_0 = (1/neg)*(total)/2.0
    weight_for_1 = (1/pos)*(total)/2.0

    class_weight={0: weight_for_0, 1: weight_for_1}



    ###########################
    ## Running the Model     ##
    ###########################


    EPOCHS = 10
    #we need a bigger batch_size to reduce the effects of the imbalanced data
    BATCH_SIZE = 2048

    initial_bias = np.log([neg/pos])


    weighted_model = make_model(hidden_layer_size=64, lr=0.01, dropout_rate=0.2)
    startTime = time.time()
    weighted_history = weighted_model.fit(
        train_features,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_features, val_labels), verbose=2)
        #class_weight=class_weight, verbose=1)
    print ("{} s to train model".format(time.time() - startTime))
        ####################################################
        ## Analyze results                                ##
        ####################################################

    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    from sklearn.metrics import f1_score, classification_report, accuracy_score
    from sklearn.metrics import cohen_kappa_score
    y_pred = weighted_model.predict (test_features)
    y_pred = y_pred.round ()
    print (y_pred)
    TARGET = 'Label'

    print ('Confusion matrix:')
    print (confusion_matrix (test_labels, y_pred,
                            labels = [0, 1]))

    print ('Classification report:')
    print (classification_report (test_labels, y_pred,
                                labels = [0, 1],
                                digits = 3))

    print ('Accuracy:', accuracy_score (test_labels, y_pred))
    print ('Precision:', precision_score (test_labels, y_pred, average = 'macro'))
    print ('Recall:', recall_score (test_labels, y_pred, average = 'macro'))
    print ('F1:', f1_score (test_labels, y_pred, average = 'macro'))
    print ('Cohen Kappa:', cohen_kappa_score (test_labels, y_pred,
                            labels = [0, 1]))

