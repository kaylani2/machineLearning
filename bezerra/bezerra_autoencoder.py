#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Authors: Kaylani Bochie and Ernesto Rodríguez
# github.com/kaylani2
# github.com/ernestorodg

###############################################################################
## Analyse Bezerra's dataset for intrusion detection using Decision Trees
###############################################################################




# In[63]:


import pandas as pd
import numpy as np
import sys
from sklearn.svm import SVC

###############################################################################
## Define constants 
###############################################################################


# Random state for reproducibility
try: 
  # If defined at argv:
  STATE = int(sys.argv[1])
except:
  # If not defined, it will be 0
  STATE = 0
np.random.seed(10)
# List of available attacks on the dataset


# Set the name of the column that will be the y set
TARGET = 'Label'


# Especific to the repository 
DATASET_DIRECTORY = r'../datasets/Dataset-bezerra-IoT-20200528T203526Z-001/Dataset-IoT/'
NETFLOW_DIRECTORY = r'NetFlow/'


# There are different csv files on the Dataset, with different types of data:

# Some meanings:
# MC: Media Center
# I: One hour of legitimate and malicious NetFlow data from profile.
# L: One hour of legitimate NetFlow data from profile.

MC = r'MC/'
ST = r'ST/'
SC = r'SC/'


# MC_I_FIRST: Has infected data by Hajime, Aidra and BashLite botnets 
MC_I_FIRST = r'MC_I1.csv'

# MC_I_SECOND: Has infected data from Mirai botnets
MC_I_SECOND = r'MC_I2.csv'

# MC_I_THIR: Has infected data from Mirai, Doflo, Tsunami and Wroba botnets
MC_I_THIRD = r'MC_I3.csv'

# MC_L: Has legitimate data, no infection
MC_L = r'MC_L.csv'


# Constants for ST
ST_I_FIRST = r'ST_I1.csv'
ST_I_SECOND = r'ST_I2.csv'
ST_I_THIRD = r'ST_I3.csv'
ST_L = r'ST_L.csv'

# Constants for SC
SC_I_FIRST = r'SC_I1.csv'
SC_I_SECOND = r'SC_I2.csv'
SC_I_THIRD = r'SC_I3.csv'
SC_L = r'SC_L.csv'


# In[64]:


# In[12]:



###############################################################################
## Load dataset
###############################################################################

# For MC data:
df_mc_I_first = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_FIRST)
df_mc_I_second = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_SECOND)
df_mc_I_third = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_THIRD)

# Add legitimate rows from MC_L
legitimate_frame_mc = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_L)

###################

# For ST data:
df_st_I_first = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_FIRST)
df_st_I_second = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_SECOND)
df_st_I_third = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_THIRD)

# Add legitimate rows from SC_L
legitimate_frame_st = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_L)


###################

# For SC data:
df_sc_I_first = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_FIRST)
df_sc_I_second = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_SECOND)
df_sc_I_third = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_THIRD)

# Add legitimate rows from MC_L
legitimate_frame_sc = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_L)

dataframes_list = [df_mc_I_first,
                df_mc_I_second,
                df_mc_I_third,
                legitimate_frame_mc,
                df_st_I_first,
                df_st_I_second,
                df_st_I_third,
                legitimate_frame_st,
                df_sc_I_first,
                df_sc_I_second,
                df_sc_I_third,
                legitimate_frame_sc]

# Joining the differents DataFrames
prev_df = pd.concat(dataframes_list)


# In[65]:


# In[ ]:


###############################################################################
## Modify the DataFrame
###############################################################################


# Sample the dataset if necessary
df = prev_df.sample (frac = 1, replace = True, random_state = 0)

# We can see that this dataset has a temporal description.
# So it is not a good idea to randomly remove rows if using RNN

# In this case we drop the index column, since pandas library creates an index
# automatically. 
df = df.drop(df.columns[0], axis=1)

# Also drop columns that has no significant data
df = df.drop(df.columns[14:], axis=1)

# Initial and end time is not a good feature for svm model
df = df.drop(['ts', 'te'], axis=1)

# Trying another drops to see relation between features and results
df = df.drop(['fwd', 'stos', 'sa', 'da'], axis=1)
# 'sp', 'dp', 'sa',  'da',  

# Counting number of null data
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]

# Remove NaN and inf values
df.replace ('Infinity', np.nan, inplace = True) ## Or other text values
df.replace (np.inf, np.nan, inplace = True) ## Remove infinity
df.replace (np.nan, 0, inplace = True)


# In[ ]:



# ###############################################################################
# ## Create artificial non-attacks samples using Random Oversampling
# ###############################################################################

# from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE

# ros = RandomOverSampler(random_state=42)

# X, y = ros.fit_resample(X, y)

# print('Number of non-attacks: ', y.value_counts()[0])
# print('Number of attacks: ', y.value_counts()[1])


# In[68]:


###############################################################################
## Create artificial non-attacks samples using Random undersampling
###############################################################################

# from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE

# ros = RandomUnderSampler(random_state=42)

# X, y = ros.fit_resample(X, y)

# print('Number of non-attacks: ', y.value_counts()[0])
# print('Number of attacks: ', y.value_counts()[1])


# # In[69]:


# X


# In[70]:


# In[ ]:


####################################################################
# Treating categorical data before splitting the dataset into the differents sets
####################################################################
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from numpy import empty

cat_cols = df.columns[df.dtypes == 'O'] # Returns array with the columns that has Object types elements

# Check wether cat_cols is empty or not. If it is empty, do not do anything
if list(cat_cols):
    categories = [
        df[column].unique() for column in df[cat_cols]]

    for cat in categories:
        cat[cat == None] = 'missing'  # noqa

    # Replacing missing values
    categorical_imputer = SimpleImputer(missing_values=None, 
                                        strategy='constant', 
                                        fill_value='missing')

    df[cat_cols] = categorical_imputer.fit_transform(df[cat_cols])

    # Encoding the categorical data
    categorical_encoder = OrdinalEncoder(categories = categories)
    categorical_encoder.fit(df[cat_cols])
    df[cat_cols] = categorical_encoder.transform(df[cat_cols])


# In[ ]:


###############################################################################
## Split dataset into train, validation and test sets
###############################################################################
### Isolate attack and normal samples
mask = df[TARGET] == 0
# 0 == normal
df_normal = df[mask]
# 1 == attack
df_attack = df[~mask]

print ('Attack set:')
print (df_attack [TARGET].value_counts ())
print ('Normal set:')
print (df_normal [TARGET].value_counts ())


# In[ ]:


### Sample and drop random attacks
df_random_attacks = df_attack.sample (n = df_normal.shape [0], random_state = STATE)
df_attack = df_attack.drop (df_random_attacks.index)

### Assemble test set
df_test = pd.DataFrame ()
df_test = pd.concat ( [df_test, df_normal])
df_test = pd.concat ( [df_test, df_random_attacks])
print ('Test set:')
print (df_test [TARGET].value_counts ())


# In[ ]:



X_test_df = df_test.iloc [:, 1:]
y_test_df = df_test.iloc [:, 0]

### K: y_test is required to plot the roc curve in the end



df_train = df_attack
VALIDATION_SIZE = 1/4
print ('\nSplitting dataset (validation/train):', VALIDATION_SIZE)
X_train_df, X_val_df, y_train_df, y_val_df = train_test_split (
                                           df_train.iloc [:, :-1],
                                           df_train.iloc [:, -1],
                                           test_size = VALIDATION_SIZE,
                                           random_state = STATE,)


print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_val_df shape:', X_val_df.shape)
print ('y_val_df shape:', y_val_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)


# In[ ]:


###############################################################################
## Convert dataframe to a numpy array
###############################################################################
print ('\nConverting dataframe to numpy array.')
X_train = X_train_df.values
y_train = y_train_df.values
X_val = X_val_df.values
y_val = y_val_df.values
X_test = X_test_df.values
y_test = y_test_df.values
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_val shape:', X_val.shape)
print ('y_val shape:', y_val.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)



# In[ ]:


###############################################################################
## Apply normalization
###############################################################################
import time

print ('\nApplying normalization.')
startTime = time.time ()
scaler = StandardScaler ()
scaler.fit (X_train)
X_train = scaler.transform (X_train)
X_val = scaler.transform (X_val)
X_test = scaler.transform (X_test)
print (str (time.time () - startTime), 'to normalize data.')


# In[ ]:


###############################################################################
## Importing necessary libraries
###############################################################################

# For the autoencoder model
import keras.utils
from keras.utils import to_categorical
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras import metrics
from keras.constraints import maxnorm

# For the metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# In[ ]:


##############################################################################
# Create learning model (Autoencoder) and tune hyperparameters
##############################################################################

# Hyperparameter tuning

test_fold = np.repeat ( [-1, 0], [X_train.shape [0], X_val.shape [0]])
myPreSplit = PredefinedSplit (test_fold)
def create_model (learn_rate = 0.01, dropout_rate = 0.0, weight_constraint = 0):
    model = Sequential ()
    model.add (Dense (X_train.shape [1], activation = 'relu',
                   input_shape = (X_train.shape [1], )))
    model.add (Dense (32, activation = 'relu'))
    model.add (Dense (8,  activation = 'relu'))
    model.add (Dense (32, activation = 'relu'))
    model.add (Dense (X_train.shape [1], activation = None))
    model.compile (loss = 'mean_squared_error',
                optimizer = 'adam',
                metrics = ['mse'])
    return model

model = KerasRegressor (build_fn = create_model, verbose = 2)
batch_size = [30]#, 50]
epochs = [5]#, 5, 10]
learn_rate = [0.01, 0.1]#, 0.2, 0.3]
dropout_rate = [0.0, 0.2]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
weight_constraint = [0]#1, 2, 3, 4, 5]
param_grid = dict (batch_size = batch_size, epochs = epochs,
                  dropout_rate = dropout_rate, learn_rate = learn_rate,
                  weight_constraint = weight_constraint)
grid = GridSearchCV (estimator = model, param_grid = param_grid,
                    scoring = 'neg_mean_squared_error', cv = myPreSplit,
                    verbose = 2, n_jobs = 16)

grid_result = grid.fit (np.vstack ( (X_train, X_val)),#, axis = 1),
                       np.vstack ( (X_train, X_val)))#, axis = 1))
print (grid_result.best_params_)

print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_ ['mean_test_score']
stds = grid_result.cv_results_ ['std_test_score']
params = grid_result.cv_results_ ['params']
for mean, stdev, param in zip (means, stds, params):
    print ("%f (%f) with: %r" % (mean, stdev, param))

# Best: -0.129429 using {'batch_size': 30, 'dropout_rate': 0.0, 'epochs': 5, 'learn_rate': 0.1, 'weight_constraint': 0}


# In[ ]:


###############################################################################
## Finished model
NUMBER_OF_EPOCHS = 5
BATCH_SIZE = 30
LEARNING_RATE = 0.1

INPUT_SHAPE = (X_train.shape [1], )

print ('\nCreating learning model.')
bestModel = Sequential ()
bestModel.add (Dense (X_train.shape [1], activation = 'relu',
                    input_shape = (X_train.shape [1], )))
bestModel.add (Dense (32, activation = 'relu'))
bestModel.add (Dense (8,  activation = 'relu'))
bestModel.add (Dense (32, activation = 'relu'))
bestModel.add (Dense (X_train.shape [1], activation = None))


###############################################################################
## Compile the network
###############################################################################
print ('\nCompiling the network.')
bestModel.compile (loss = 'mean_squared_error',
                 optimizer = Adam (lr = LEARNING_RATE),
                 metrics = ['mse'])#,metrics.Precision ()])

print ('Model summary:')
bestModel.summary ()


# In[ ]:


###############################################################################
## Fit the network
###############################################################################
print ('\nFitting the network.')
startTime = time.time ()
history = bestModel.fit (X_train, X_train,
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
X_val_pred   = bestModel.predict (X_val)
X_train_pred = bestModel.predict (X_train)
print ('Train error:'     , mean_squared_error (X_train_pred, X_train))
print ('Validation error:', mean_squared_error (X_val_pred, X_val))

#SAMPLES = 50
#print ('Error on first', SAMPLES, 'samples:')
#print ('MSE (pred, real)')
#for pred_sample, real_sample in zip (X_val_pred [:SAMPLES], X_val [:SAMPLES]):
#  print (mean_squared_error (pred_sample, real_sample))

### K: This looks like another hyperparameter to be adjusted by using a
### separate validation set that contains normal and anomaly samples.
### K: I've guessed 1%, this may be a future line of research.
THRESHOLD_SAMPLE_PERCENTAGE = 1/100

train_mse_element_wise = np.mean (np.square (X_train_pred - X_train), axis = 1)
val_mse_element_wise = np.mean (np.square (X_val_pred - X_val), axis = 1)

max_threshold_val = np.max (val_mse_element_wise)
print ('max_Thresh val:', max_threshold_val)



print ('samples:')
print (int (round (val_mse_element_wise.shape [0] *
         THRESHOLD_SAMPLE_PERCENTAGE)))

top_n_values_val = np.partition (-val_mse_element_wise,
                               int (round (val_mse_element_wise.shape [0] *
                                           THRESHOLD_SAMPLE_PERCENTAGE)))

top_n_values_val = -top_n_values_val [: int (round (val_mse_element_wise.shape [0] *
                                                  THRESHOLD_SAMPLE_PERCENTAGE))]


### K: O limiar de classificacao sera a mediana dos N maiores custos obtidos
### ao validar a rede no conjunto de validacao. N e um hiperparametro que pode
### ser ajustado, mas e necessario um conjunto de validacao com amostras
### anomalas em adicao ao conjunto de validacao atual, que so tem amostras nao
### anomalas. @TODO: Desenvolver e validar o conjunto com esta nova tecnica.
threshold = np.median (top_n_values_val)
print ('Thresh val:', threshold)


# In[ ]:



### K: NOTE: Only look at test results when publishing...
#sys.exit ()
X_test_pred = bestModel.predict (X_test)
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
print ('\nMSE (pred, real) | Label (ordered)')
tp, tn, fp, fn = 0, 0, 0, 0
for label, pred in zip (y_test, y_pred):
#  if (pred >= threshold):
#    print ('Classified as anomaly     (NORMAL):', label)
#  else:
#    print ('Classified as not anomaly (ATTACK):', label)

    if ((pred >= threshold) and (label == 0)):
        tn += 1
    elif ((pred >= threshold) and (label == 1)):
        fn += 1
    elif ((pred < threshold) and (label == 1)):
        tp += 1
    elif ((pred < threshold) and (label == 0)):
        fp += 1

print ('Confusion matrix:')
print ('tp | fp')
print ('fn | tn\n\n')
print (tp, '|', fp)
print (fn, '|', tn)


# In[ ]:




