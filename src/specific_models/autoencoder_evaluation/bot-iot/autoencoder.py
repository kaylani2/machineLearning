#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Autoencoder
import sys
import time
import pandas as pd
import os
import math
sys.path.insert(1, '../')
import numpy as np
from numpy import mean, std
from unit import remove_columns_with_one_value, remove_nan_columns, load_dataset
from unit import display_general_information, display_feature_distribution
from collections import Counter
#from imblearn.over_sampling import RandomOverSampler, RandomUnderSampler
import sklearn
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import keras.utils
from keras import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from keras.optimizers import RMSprop, Adam
from keras.constraints import maxnorm


# In[2]:


###############################################################################
## Define constants
###############################################################################
pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 5)
BOT_IOT_DIRECTORY = '../../../../../datasets/bot-iot/'
BOT_IOT_FEATURE_NAMES = 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv'
BOT_IOT_FILE_5_PERCENT_SCHEMA = 'UNSW_2018_IoT_Botnet_Full5pc_{}.csv' # 1 - 4
FIVE_PERCENT_FILES = 4
BOT_IOT_FILE_FULL_SCHEMA = 'UNSW_2018_IoT_Botnet_Dataset_{}.csv' # 1 - 74
FULL_FILES = 74
FILE_NAME = BOT_IOT_DIRECTORY + BOT_IOT_FILE_5_PERCENT_SCHEMA
FEATURES = BOT_IOT_DIRECTORY + BOT_IOT_FEATURE_NAMES
NAN_VALUES = ['?', '.']
TARGET = 'attack'
INDEX_COLUMN = 'pkSeqID'
LABELS = ['attack', 'category', 'subcategory']
STATE = 0
try:
  STATE = int (sys.argv [1])
except:
  pass
#for STATE in [1, 2, 3, 4, 5]:
np.random.seed (STATE)
print ('STATE:', STATE)


# In[3]:


###############################################################################
## Load dataset
###############################################################################
df = load_dataset (FILE_NAME, FIVE_PERCENT_FILES, INDEX_COLUMN, NAN_VALUES)


# In[4]:


###############################################################################
## Clean dataset
###############################################################################
###############################################################################
### Remove columns with only one value
df, log = remove_columns_with_one_value (df, verbose = False)
print (log)


###############################################################################
### Remove redundant columns, useless columns and unused targets
### K: _number columns are numerical representations of other existing columns.
### K: category and subcategory are other labels.
### K: saddr and daddr may specialize the model to a single network
redundant_columns = ['state_number', 'proto_number', 'flgs_number']
other_targets = ['category', 'subcategory']
misc_columns = ['saddr', 'daddr']
print ('Removing redundant columns:', redundant_columns)
print ('Removing useless targets:', other_targets)
print ('Removing misc columns:', misc_columns)
columns_to_remove = redundant_columns + other_targets + misc_columns
df.drop (axis = 'columns', columns = columns_to_remove, inplace = True)

###############################################################################
### Remove NaN columns (with a lot of NaN values)
df, log = remove_nan_columns (df, 1/2, verbose = False)
print (log)

###############################################################################
### Encode categorical features
print ('Encoding categorical features (ordinal encoding).')
my_encoder = OrdinalEncoder ()
df ['flgs'] = my_encoder.fit_transform (df ['flgs'].values.reshape (-1, 1))
df ['proto'] = my_encoder.fit_transform (df ['proto'].values.reshape (-1, 1))
df ['sport'] = my_encoder.fit_transform (df ['sport'].astype (str).values.reshape (-1, 1))
df ['dport'] = my_encoder.fit_transform (df ['dport'].astype (str).values.reshape (-1, 1))
df ['state'] = my_encoder.fit_transform (df ['state'].values.reshape (-1, 1))
print ('Objects:', list (df.select_dtypes ( ['object']).columns))


# In[5]:


###############################################################################
## Quick sanity check
###############################################################################
display_general_information (df)


# In[6]:


###############################################################################
## Split dataset
###############################################################################
## K: Dataset is too big? Drop.
drop_indices = np.random.choice (df.index, int (df.shape [0] * 0.5),
                                 replace = False)


# In[7]:


### Isolate attack and normal samples
df = df.drop (drop_indices)
mask = df [TARGET] == 0
# 0 == normal
df_normal = df [mask]
# 1 == attack
df_attack = df [~mask]

print ('Attack set:')
print (df_attack [TARGET].value_counts ())
print ('Normal set:')
print (df_normal [TARGET].value_counts ())


# In[8]:


### Sample and drop random attacks
df_random_attacks = df_attack.sample (n = df_normal.shape [0], random_state = STATE)
df_attack = df_attack.drop (df_random_attacks.index)

### Assemble train set (only attacks)
X_train = df_attack.loc [:, df.columns != TARGET]
y_train = df_attack [TARGET]
print ('Train set:')
print (df_attack [TARGET].value_counts ())

### Assemble test set (50/50 attacks and non-attacks)
df_test = pd.DataFrame ()
df_test = pd.concat ( [df_test, df_normal])
df_test = pd.concat ( [df_test, df_random_attacks])
print ('Test set:')
print (df_test [TARGET].value_counts ())
X_test = df_test.loc [:, df.columns != TARGET]
y_test = df_test [TARGET]
### K: y_test is required to plot the roc curve in the end

# df_train = df_attack
VALIDATION_SIZE = 1/4
print ('\nSplitting dataset (validation/train):', VALIDATION_SIZE)
X_train, X_val, y_train, y_val = train_test_split (
                                             X_train,
                                             y_train,
                                             test_size = VALIDATION_SIZE,
                                             random_state = STATE,)


# In[9]:


print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_val shape:', X_val.shape)
print ('y_val shape:', y_val.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)


# In[10]:


# ###############################################################################
# ## Convert dataframe to a numpy array
# ###############################################################################
# print ('\nConverting dataframe to numpy array.')
# X_train = X_train.values
# y_train = y_train.values
# X_val = X_val.values
# y_val = y_val.values
# X_test = X_test.values
# y_test = y_test.values
# print ('X_train shape:', X_train.shape)
# print ('y_train shape:', y_train.shape)
# print ('X_val shape:', X_val.shape)
# print ('y_val shape:', y_val.shape)
# print ('X_test shape:', X_test.shape)
# print ('y_test shape:', y_test.shape)


# In[11]:


###############################################################################
## Apply normalization
###############################################################################
### K: NOTE: Only use derived information from the train set to avoid leakage.
print ('\nApplying normalization.')
startTime = time.time ()
scaler = StandardScaler ()
scaler.fit (X_train)
X_train = scaler.transform (X_train)
X_val = scaler.transform (X_val)
X_test = scaler.transform (X_test)
print (str (time.time () - startTime), 'to normalize data.')


# In[12]:


print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_val shape:', X_val.shape)
print ('y_val shape:', y_val.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)
print (type (X_train))


# In[20]:


###############################################################################
## Perform feature selection
###############################################################################
### K: TODO: Adjust NUMBER_OF_FEATURES
NUMBER_OF_FEATURES = 15 #X_train.shape [-1]
print ('\nSelecting top', NUMBER_OF_FEATURES, 'features.')
fs = PCA (n_components = NUMBER_OF_FEATURES)
fs.fit (X_train)
print (fs.explained_variance_ratio_)
print (fs.singular_values_)
X_train = fs.transform (X_train)
X_val = fs.transform (X_val)
X_test = fs.transform (X_test)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_val shape:', X_val.shape)
print ('y_val shape:', y_val.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)


# In[63]:


###############################################################################
## Create learning model (Autoencoder) and tune hyperparameters
###############################################################################
###############################################################################
# Hyperparameter tuning
test_fold = np.repeat ([-1, 0], [X_train.shape [0], X_val.shape [0]])
myPreSplit = PredefinedSplit (test_fold)
def create_model (learn_rate = 0.01, dropout_rate = 0.0, weight_constraint = 0,
                  metrics = ['mse'], input_layer_neurons,
                  neurons_on_first_layer = 32,
                  second_layer_boolean = False,
                  neurons_on_chokehold_layer = 8):
  model = Sequential ()
  model.add (Dense (input_layer_neurons, activation = 'relu',
                   input_shape = (input_layer_neurons, )))
  if (second_layer_boolean):
    model.add (Dense (neurons_on_first_layer/2, activation = 'relu'))
  model.add (Dense (neurons_on_second_layer, activation = 'relu'))
  model.add (Dense (neurons_on_chokehold_layer,  activation = 'relu'))
  if (second_layer_boolean):
    model.add (Dense (neurons_on_first_layer/2, activation = 'relu'))
  model.add (Dense (neurons_on_first_layer, activation = 'relu'))
  model.add (Dense (input_layer_neurons, activation = None))
  model.compile (loss = 'mean_squared_error',
                 optimizer = 'adam',
                 metrics = metrics)
  return model


model = KerasRegressor (build_fn = create_model, verbose = 2)
input_layer_neurons = X_train.shape [1]
batch_size = [5000, 10000]
epochs = [200]
learn_rate = [0.0001, 0.001]
dropout_rate = [0.0]
weight_constraint = [0]
neurons_on_first_layer = [32, 64]
neurons_on_chokehold_layer = [4, 8]
second_layer_boolean = [False, True] # Is there another layer?
param_grid = dict (batch_size = batch_size, epochs = epochs,
                   dropout_rate = dropout_rate, learn_rate = learn_rate,
                   weight_constraint = weight_constraint,
                   input_layer_neurons = input_layer_neurons,
                   neurons_on_first_layer = neurons_on_first_layer,
                   second_layer_boolean = second_layer_boolean,
                   neurons_on_chokehold_layer = neurons_on_chokehold_layer)
grid = GridSearchCV (estimator = model, param_grid = param_grid,
                    scoring = 'neg_mean_squared_error', cv = myPreSplit,
                    verbose = 2, n_jobs = 1)
startTime = time.time ()
grid_result = grid.fit (np.vstack ( (X_train, X_val)),#, axis = 1),
                       np.vstack ( (X_train, X_val)))#, axis = 1))
print (str (time.time () - startTime), 's to search grid.')
print (grid_result.best_params_)

print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_ ['mean_test_score']
stds = grid_result.cv_results_ ['std_test_score']
params = grid_result.cv_results_ ['params']
for mean, stdev, param in zip (means, stds, params):
  print ("%f (%f) with: %r" % (mean, stdev, param))

## 9 features:
## Best: -0.148847 using {'batch_size': 5000, 'dropout_rate': 0.0, 'epochs': 10, 'learn_rate': 0.001, 'weight_constraint': 0}

## All features (SEM PCA): (97s to search grid)
## Best: -0.159944 using {'batch_size': 10000, 'dropout_rate': 0.0, 'epochs': 10, 'learn_rate': 0.1, 'weight_constraint': 0}

## All features (COM PCA):  (100s to search grid)
## Best: -0.100227 using {'batch_size': 5000, 'dropout_rate': 0.0, 'epochs': 10, 'learn_rate': 0.001, 'weight_constraint': 0}

## 32 features: (88s to search grid)
## Best: -0.179934 using {'batch_size': 5000, 'dropout_rate': 0.0, 'epochs': 10, 'learn_rate': 0.001, 'weight_constraint': 0}

## 15 features (67s to search grid)
## Best: -0.235642 using {'batch_size': 10000, 'dropout_rate': 0.0, 'epochs': 10, 'learn_rate': 0.1, 'weight_constraint': 0}

## 15 features (60min to search grid)
## Best: -0.019674 using {'batch_size': 10000, 'dropout_rate': 0.0, 'epochs': 200, 'learn_rate': 0.0001, 'neurons_on_chokehold_layer': 8, 'neurons_on_first_layer': 64, 'weight_constraint': 0}


# In[38]:


###############################################################################
## Finished model
# Best: -0.015576 using {'batch_size': 5000, 'dropout_rate': 0.0, 'epochs': 10,
# learn_rate': 0.1, 'weight_constraint': 0}
METRICS = [keras.metrics.MeanSquaredError (name = 'MSE'),
           keras.metrics.RootMeanSquaredError (name = 'RMSE'),  
           keras.metrics.MeanAbsoluteError (name = 'MAE'),]
### K: learning rate foi alterado manualmente ao olhar os valores do erro na
### validacao ao longo das epochs...
NUMBER_OF_EPOCHS = 700
BATCH_SIZE = 10000
LEARNING_RATE = 0.0001

print ('\nCreating learning model.')
clf = Sequential ()
clf.add (Dense (X_train.shape [1], activation = 'relu',
                      input_shape = (X_train.shape [1], )))
clf.add (Dense (32, activation = 'relu'))
clf.add (Dense (8,  activation = 'relu'))
clf.add (Dense (32, activation = 'relu'))
clf.add (Dense (X_train.shape [1], activation = None))


###############################################################################
## Compile the network
###############################################################################
print ('\nCompiling the network.')
clf.compile (loss = 'mean_squared_error',
                   optimizer = Adam (lr = LEARNING_RATE),
                   metrics = METRICS)
print ('Model summary:')
clf.summary ()


# In[39]:


###############################################################################
## Fit the network
###############################################################################
print ('\nFitting the network.')
startTime = time.time ()
history = clf.fit (X_train, X_train,
                         batch_size = BATCH_SIZE,
                         epochs = NUMBER_OF_EPOCHS,
                         verbose = 2, #1 = progress bar, not useful for logging
                         workers = 0,
                         use_multiprocessing = True,
                         #class_weight = 'auto',
                         validation_data = (X_val, X_val))
print (str (time.time () - startTime), 's to train model.')


# In[ ]:


###############################################################################
## Analyze results
###############################################################################
X_val_pred   = clf.predict (X_val)
X_train_pred = clf.predict (X_train)
print ('Train error (MSE):'     , mean_squared_error (X_train_pred, X_train))
print ('Validation error (MSE):', mean_squared_error (X_val_pred, X_val))

#SAMPLES = 50
#print ('Error on first', SAMPLES, 'samples:')
#print ('MSE (pred, real)')
#for pred_sample, real_sample in zip (X_val_pred [:SAMPLES], X_val [:SAMPLES]):
#  print (mean_squared_error (pred_sample, real_sample))


train_mse_element_wise = np.mean (np.square (X_train_pred - X_train), axis = 1)
val_mse_element_wise = np.mean (np.square (X_val_pred - X_val), axis = 1)


# In[62]:


max_threshold_val = np.max (val_mse_element_wise)
val_mse_element_wise.sort ()
print ('Maximum validation error (MSE):', max_threshold_val)
print ('Bottom 20 validation error (MSE):', val_mse_element_wise [:20])
print ('Top 20 validation error (MSE):', val_mse_element_wise [::-1][:20])


### K: This looks like another hyperparameter to be adjusted by using a
### separate validation set that contains normal and anomaly samples.
### K: I've guessed 1%, this may be a future line of research.
THRESHOLD_SAMPLE_PERCENTAGE = 1/100
top_n_values_val = np.partition (-val_mse_element_wise,
                                 int (round (val_mse_element_wise.shape [0] *
                                             THRESHOLD_SAMPLE_PERCENTAGE)))

top_n_values_val = -top_n_values_val [: int (round (val_mse_element_wise.shape [0] *
                                                    THRESHOLD_SAMPLE_PERCENTAGE))]
print ('\nNumber of samples considered:', int (round (val_mse_element_wise.shape [0] *
                                        THRESHOLD_SAMPLE_PERCENTAGE)))
top_n_values_val.sort ()
print ('Bottom 20 on considered samples (MSE):', top_n_values_val [:20])
print ('Top 20 on considered samples (MSE):', top_n_values_val [::-1][:20])


### K: O limiar de classificacao sera a mediana dos N maiores custos obtidos
### ao validar a rede no conjunto de validacao. N e um hiperparametro que pode
### ser ajustado, mas e necessario um conjunto de validacao com amostras
### anomalas em adicao ao conjunto de validacao atual, que so tem amostras nao
### anomalas. @TODO: Desenvolver e validar o conjunto com esta nova tecnica.
threshold = np.median (top_n_values_val)
print ('Thresh val:', threshold)


# In[ ]:


### K: NOTE: Only look at test results when publishing...
sys.exit ()
X_test_pred = clf.predict (X_test)
print (X_test_pred.shape)
print ('Test error:', mean_squared_error (X_test_pred, X_test))


y_pred = np.mean (np.square (X_test_pred - X_test), axis = 1)
#y_pred = []
#for pred_sample, real_sample, label in zip (X_test_pred, X_test, y_test):
#  y_pred.append (mean_squared_error (pred_sample, real_sample))

#print ('\nLabel | MSE (pred, real)')
#for label, pred in zip (y_test, y_pred):
#  print (label, '|', pred)

y_test, y_pred = zip (*sorted (zip (y_test, y_pred)))
#print ('\nLabel | MSE (pred, real) (ordered)')
#for label, pred in zip (y_test, y_pred):
#  print (label, '|', pred)

# 0 == normal
# 1 == attack
print ('\nPerformance on TEST set:')
print ('\nMSE (pred, real) | Label (ordered)')
tp, tn, fp, fn = 0, 0, 0, 0
for label, pred in zip (y_test, y_pred):
#  if (pred >= threshold):
#    print ('Classified as anomaly     (NORMAL):', label)
#  else:
#    print ('Classified as not anomaly (ATTACK):', label)

  if ((pred >= threshold) and (label == 0)):
    print ('True negative.')
    tn += 1
  elif ((pred >= threshold) and (label == 1)):
    print ('False negative!')
    fn += 1
  elif ((pred < threshold) and (label == 1)):
    print ('True positive.')
    tp += 1
  elif ((pred < threshold) and (label == 0)):
    print ('False positive!')
    fp += 1

print ('Confusion matrix:')
print ('tp | fp')
print ('fn | tn')
print (tp, '|', fp)
print (fn, '|', tn)
print ('TP:', tp)
print ('TN:', tn)
print ('FP:', fp)
print ('FN:', fn)

