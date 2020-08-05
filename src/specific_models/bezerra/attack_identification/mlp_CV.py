# Author: Luiz Giserman
# github.com/LuizGiserman
# giserman AT gta DOT ufrj DOT br
import functools
import time
import sys
import math
import tensorflow as tf
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from unit import remove_columns_with_one_value
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score
from  unit import load_dataset, remove_columns_with_one_value, remove_nan_columns

state = 0
try:
  state = int (sys.argv [1])
except:
  pass
print ("STATE = ", state)

df = load_dataset ()
print ("Data Loaded")
remove_columns_with_one_value (df, verbose = False)
remove_nan_columns (df, 0.6, verbose = False)
#making the final DataFrame
#dropping the number of the rows column
df = df.drop (df.columns [0], axis = 1)

#dropping unrelated columns
df.drop (axis = 'columns', columns = ['ts', 'te', 'sa', 'da'], inplace = True)

#################################
## Encoding the data           ##
#################################

cat_cols, num_cols = df.columns [df.dtypes == 'O'], df.columns [df.dtypes != 'O']
num_cols = num_cols [1:]

categories = [df [column].unique () for column in df [cat_cols]]

categorical_encoder = preprocessing.OrdinalEncoder (categories = categories)
categorical_encoder.fit (df [cat_cols])
df [cat_cols] = categorical_encoder.transform (df [cat_cols])

############################################
## Split dataset into train and test sets ##
############################################
# for state in STATES:


TEST_SIZE = 0.3
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                                df.iloc [:, 1:],
                                                df.iloc [:, 0],
                                                test_size = TEST_SIZE,
                                                random_state = state)
print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)



cols = (len (num_cols) + len (cat_cols)) * [None]
cols [0:len (num_cols)] = num_cols
cols [len (num_cols):] = cat_cols

standard_scaler_features = cols
my_scaler = StandardScaler ()
steps = []
steps.append (('scaler', my_scaler))
standard_scaler_transformer = Pipeline (steps)

preprocessor = ColumnTransformer (transformers = [
               ('sca', standard_scaler_transformer, standard_scaler_features)])


METRICS = [
    keras.metrics.TruePositives (name = 'tp'),
    keras.metrics.FalsePositives (name = 'fp'),
    keras.metrics.TrueNegatives (name = 'tn'),
    keras.metrics.FalseNegatives (name = 'fn'),
    keras.metrics.BinaryAccuracy (name = 'accuracy'),
    keras.metrics.Precision (name = 'precision'),
    keras.metrics.Recall (name = 'recall'),
    keras.metrics.AUC (name = 'auc'),
]

def create_model (metrics = METRICS, output_bias = None, hidden_layer_size = 32, lr = 1e-3, dropout_rate = 0.0):

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant (output_bias)

    model = keras.Sequential ( [
        keras.layers.Dense (hidden_layer_size, activation = 'relu',
                           input_shape = (X_train_df.shape [-1],)),
        keras.layers.Dense (hidden_layer_size, activation = 'relu'),
        keras.layers.Dropout (dropout_rate),
        keras.layers.Dense (hidden_layer_size, activation = 'relu'),
        keras.layers.Dense (1, activation = 'sigmoid',
                           bias_initializer = output_bias)
    ])

    model.compile (
        optimizer = keras.optimizers.Adam (lr = lr),
        loss = keras.losses.BinaryCrossentropy (),
        metrics = metrics)

    return model

clf = KerasClassifier (build_fn = create_model, hidden_layer_size = 64,
                       batch_size = 1000, epochs = 17, verbose = 2)
clf = Pipeline (steps = [ ('preprocessor', preprocessor),
                      ('classifier', clf)], verbose = True)
startTime = time.time ()
clf.fit (X_train_df, y_train_df)
print (str (time.time () - startTime), 's to train model')


TARGET = 'Label'
print ('\nPerformance on TEST set:')
y_pred = clf.predict (X_test_df)
my_confusion_matrix = confusion_matrix (y_test_df, y_pred, labels = df [TARGET].unique ())
tn, fp, fn, tp = my_confusion_matrix.ravel ()
print ('Confusion matrix:')
print (my_confusion_matrix)
print ('Accuracy:', accuracy_score (y_test_df, y_pred))
print ('Precision:', precision_score (y_test_df, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_test_df, y_pred, average = 'macro'))
print ('F1:', f1_score (y_test_df, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_test_df, y_pred,
                        labels = df [TARGET].unique ()))
print ('TP:', tp)
print ('TN:', tn)
print ('FP:', fp)
print ('FN:', fn)


##### GRID SEARCH #######

#from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

#param_grid = {'classifier__epochs': [1, 3, 5], 'classifier__hidden_layer_size': [16, 32, 64],
#              'classifier__dropout_rate': [0.0, 0.1, 0.2, 0.3], 'classifier__batch_size': [1000, 2048, 3000]}
#cv = RepeatedStratifiedKFold (n_splits = 5, n_repeats = 1, random_state = 0)
#grid = GridSearchCV (estimator = clf, param_grid = param_grid, scoring = 'f1', verbose = 1, n_jobs = -1, cv = cv)
#grid_result = grid.fit (X_train_df, y_train_df)
#
#print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_ ['mean_test_score']
#stds = grid_result.cv_results_ ['std_test_score']
#params = grid_result.cv_results_ ['params']
#for mean, stdev, param in zip (means, stds, params):
#  print ("%f (%f) with: %r" % (mean, stdev, param))
