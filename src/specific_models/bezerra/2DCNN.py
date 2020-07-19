import functools
import time
import math
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import datetime
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

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

  ########################
  ## Reshaping the Data ##
  ########################
  SAMPLE_2D_SIZE = 3 # 3x3

  ## zero padding and reshaping

  train_features.resize((train_features.shape[0], SAMPLE_2D_SIZE, SAMPLE_2D_SIZE))
  train_features = train_features.reshape((train_features.shape[0], 3, 3, 1))
  val_features.resize((val_features.shape[0], SAMPLE_2D_SIZE, SAMPLE_2D_SIZE))
  val_features = val_features.reshape((val_features.shape[0], 3, 3, 1))
  test_features.resize((test_features.shape[0], SAMPLE_2D_SIZE, SAMPLE_2D_SIZE))
  test_features = test_features.reshape((test_features.shape[0], 3, 3, 1))
  print (train_features.shape)

  ########################
  ## Building the Model ##
  ########################

  FILTERS = 4

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

  def create_model (lr=1e-1, dropout_rate=0.0):
      initializer = tf.initializers.VarianceScaling(scale=2.0)

      model = models.Sequential()
      model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=(3, 3, 1), kernel_initializer=initializer))
      model.add(layers.Flatten())
      model.add(layers.Dense(64, activation='relu', kernel_initializer=initializer))
      model.add(layers.Dropout(dropout_rate))
      model.add(layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))
      # model.summary()
      model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=['binary_accuracy'])

      return model
#   ####GRID SEARCH
#   from sklearn.model_selection import GridSearchCV
#   from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#   from sklearn.model_selection import PredefinedSplit

#   model = KerasClassifier (build_fn=create_model, verbose=2)

#   batch_size = [1000, 2048, 3200]
#   epochs = [3, 5, 10]
#   lr = [1e-3, 1e-2, 1e-1, 2e-1]
#   dropout_rate = [0.0, 0.2, 0.3]

#   test_fold = np.repeat ([-1, 0] , [train_features.shape[0], val_features.shape[0]])
#   myPreSplit = PredefinedSplit (test_fold)

#   param_grid = dict(batch_size = batch_size, epochs=epochs, lr=lr, dropout_rate=dropout_rate)
#   grid = GridSearchCV (estimator = model, cv=myPreSplit, param_grid=param_grid, scoring= 'f1_weighted', verbose=2, n_jobs=-1)

#   grid_result = grid.fit( np.concatenate ((train_features, val_features), axis=0,),
#                           np.concatenate ((train_labels, val_labels), axis=0))


#   print (grid_result.best_params_)
#   print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#   means = grid_result.cv_results_['mean_test_score']
#   stds = grid_result.cv_results_['std_test_score']
#   params = grid_result.cv_results_['params']
#   for mean, stdev, param in zip (means, stds, params):
#    print ("%f (%f) with: %r" % (mean, stdev, param))
#   sys.exit ()


  model = create_model(lr=1e-3)
  startTime = time.time()
  history = model.fit (train_features, train_labels,
  epochs=50, validation_data=(val_features, val_labels), batch_size=100, verbose=2)
  print("{} s to train model".format(time.time()-startTime))

  from sklearn.metrics import confusion_matrix, precision_score, recall_score
  from sklearn.metrics import f1_score, classification_report, accuracy_score
  from sklearn.metrics import cohen_kappa_score
  y_pred = model.predict (test_features)
  y_pred = y_pred.round ()
  # print (y_pred)
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
