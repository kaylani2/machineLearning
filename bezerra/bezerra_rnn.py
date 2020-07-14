#!/usr/bin/env python
# coding: utf-8

# In[62]:


# Author: Ernesto Rodríguez
# github.com/ernestorodg

###############################################################################
## Analyse Bezerra's dataset for intrusion detection using Decision Trees
###############################################################################


# In[63]:


import pandas as pd
import numpy as np
import sys

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
STEPS = 10


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


###############################################################################
## Modify the DataFrame
###############################################################################


# Sample the dataset if necessary
# df = prev_df.sample (frac = 1, replace = True, random_state = 0)

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


# if (df.Label.value_counts()[1] < df.Label.value_counts()[0]):
#     remove_n =  df.Label.value_counts()[0] - df.Label.value_counts()[1]  # Number of rows to be removed   
#     print(remove_n)
#     df_to_be_dropped = df[df.Label == 0]
#     drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)
#     df = df.drop(drop_indices)
# else: 
#     remove_n =  df.Label.value_counts()[1] - df.Label.value_counts()[0]  # Number of rows to be removed   
#     print(remove_n)
#     df_to_be_dropped = df[df.Label == 1]
#     drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)
#     df = df.drop(drop_indices)


# In[66]:


###############################################################################
## Slice the dataframe (usually the last column is the target)
###############################################################################

X = pd.DataFrame(df.iloc [:, 1:])

# Selecting other columns
# X = pd.concat([X, df.iloc[:, 2]], axis=1)

y = df.iloc [:, 0]
print('Number of non-attacks: ', y.value_counts()[0])
print('Number of attacks: ', y.value_counts()[1])

# See Output, only available on jupyter-notebooks
# X



###############################################################################
## Modify number of samples
###############################################################################


# from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
# from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE

# if TRAFFIC == 0: 
#     # Oversampling
#     ros = RandomOverSampler(random_state=42)

#     X, y = ros.fit_resample(X, y)

#     print('Number of non-attacks: ', y.value_counts()[0])
#     print('Number of attacks: ', y.value_counts()[1])
# else: 
#     # Undersampling
#     ros = RandomUnderSampler(random_state=42)

#     X, y = ros.fit_resample(X, y)

#     print('Number of non-attacks: ', y.value_counts()[0])
#     print('Number of attacks: ', y.value_counts()[1])




####################################################################
# Treating categorical data before splitting the dataset into the differents sets
####################################################################
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

cat_cols = X.columns[X.dtypes == 'O'] # Returns array with the columns that has Object types elements

# Check wether cat_cols is empty or not. If it is empty, do not do anything
if list(cat_cols):
    categories = [
        X[column].unique() for column in X[cat_cols]]

    for cat in categories:
        cat[cat == None] = 'missing'  # noqa

    # Replacing missing values
    categorical_imputer = SimpleImputer(missing_values=None, 
                                        strategy='constant', 
                                        fill_value='missing')

    X[cat_cols] = categorical_imputer.fit_transform(X[cat_cols])

    # Encoding the categorical data
    categorical_encoder = OrdinalEncoder(categories = categories)
    categorical_encoder.fit(X[cat_cols])
    X[cat_cols] = categorical_encoder.transform(X[cat_cols])


# In[10]:


###############################################################################
## Split dataset into train and test sets if not using cross validation
###############################################################################
from sklearn.model_selection import train_test_split
TEST_SIZE = 1/5
VALIDATION_SIZE = 1/5


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = TEST_SIZE,
                                                     random_state = STATE)


print ('\nSplitting dataset (validation/train):', 1/5)
X_train_val, X_val, y_train_val, y_val = train_test_split (
                                             X_train,
                                             y_train,
                                             test_size = VALIDATION_SIZE,
                                             random_state = STATE)


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train_val = pd.DataFrame(X_train_val)
X_val = pd.DataFrame(X_val)


# In[11]:


####################################################################
# Treat numerical data 
####################################################################
from sklearn.preprocessing import StandardScaler


num_cols = X_train.columns[(X_train.dtypes == 'float64') | (X_train.dtypes == 'int64')] # Returns array with the columns that has float types elements

# Scaling numerical values

numerical_scaler = StandardScaler()
numerical_scaler.fit(X_train)
X_train = numerical_scaler.transform(X_train)

X_test = numerical_scaler.transform(X_test)


print('y_test values: ', y_test.value_counts())




# from numpy import array


# sets_list = [X_train, X_test]
# for index, data in enumerate(sets_list):
#     n = data.shape[0]
#     samples = []

#     # step over the X_train.shape[0] (samples) in jumps of 200 (time_steps)
#     for i in range(0,n,STEPS):
#         # grab from i to i + 200
#         sample = data[i:i+STEPS]
#         samples.append(sample)

#     # convert list of arrays into 2d array
#     new_data = list()
#     new_data = np.array(new_data)
#     for i in range(len(samples)):
#         new_data = np.append(new_data, samples[i])
        
#     sets_list[index] = new_data.reshape(len(samples), STEPS, data.shape[1])
     
    
# X_train = sets_list[0]
# X_test = sets_list[1]


# answer_list = [y_train, y_test]
# for index, answer in enumerate(answer_list):

#     new_answer = list()
#     new_answer = np.array(new_answer)
#     answer = np.array(answer)
#     for i in range (0, len(answer), STEPS):
#         new_answer = np.append(new_answer, answer[i])
#     answer_list[index] = new_answer


# def window_stack(a, stepsize=STEPS, width=3):
#     n = a.shape[0]
#     return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

# ## Preparing the test to see performance
# y_train = answer_list[0]
# y_test = answer_list[1]


# # In[17]:


# pd.Series(y_train).value_counts()

FEATURES = X_train.shape [1]
def window_stack (a, stride = 1, numberOfSteps = 3):
    return np.hstack ([ a [i:1+i-numberOfSteps or None:stride] for i in range (0,numberOfSteps) ])

X_train = window_stack (X_train, stride = 1, numberOfSteps = STEPS)
X_train = X_train.reshape (X_train.shape [0], STEPS, FEATURES)
X_test = window_stack (X_test, stride = 1, numberOfSteps = STEPS)
X_test = X_test.reshape (X_test.shape [0], STEPS, FEATURES)

y_train = y_train [ (STEPS - 1):]
y_test = y_test [ (STEPS - 1):]

print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)



# univariate lstm

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

###################

print ('\nCreating learning model.')
numberOfClasses = len (df [TARGET].unique ())
BATCH_SIZE = 30
NUMBER_OF_EPOCHS = 3
LEARNING_RATE = 0.001
WEIGHT_CONSTRAINT = 1
bestModel = Sequential ()
#bestModel.add (LSTM (100, activation = 'relu', return_sequences = True,
#               input_shape = (X_train.shape [1], X_train.shape [2])))
#bestModel.add (LSTM (100, activation = 'relu'))
bestModel.add (LSTM (50, activation= 'relu' , input_shape= (X_train.shape [1], X_train.shape [2])))
bestModel.add (Dense (1, activation = 'sigmoid'))

print ('Model summary:')
bestModel.summary ()

###############################################################################
## Compile the network
###############################################################################
print ('\nCompiling the network.')
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import metrics
bestModel.compile (optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   )#metrics = ['binary_accuracy', metrics.Precision ()])
#bestModel.compile (loss = 'binary_crossentropy',
#                   optimizer = Adam (lr = LEARNING_RATE),
#                   metrics = ['binary_accuracy',
#                              #metrics.Recall (),
#                              metrics.Precision ()])

###############################################################################
## Fit the network
###############################################################################

import time

print ('\nFitting the network.')
startTime = time.time ()
#history = bestModel.fit (X_train, y_train,
#                         batch_size = BATCH_SIZE,
#                         epochs = NUMBER_OF_EPOCHS,
#                         verbose = 2, #1 = progress bar, not useful for logging
#                         workers = 0,
#                         use_multiprocessing = True,
#                         #class_weight = 'auto',
#                         validation_data = (X_val, y_val))
bestModel.fit (X_train, y_train, epochs = NUMBER_OF_EPOCHS,
               use_multiprocessing = True, verbose = 2)
training_time = time.time () - startTime
print (str (training_time), 's to train model.')

###################

# # define  my model
# model = Sequential()
# model.add(LSTM(50, activation= 'relu' , input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(1))
# model.compile(optimizer= 'adam' , loss= 'mse' )




# import time
# # fit model

# # Measure time of this training
# start_time = time.time()

# model.fit(X_train, y_train, epochs=50, verbose=0)

# print("--- %s seconds ---" % (time.time() - start_time))

# y_pred = model.predict(X_test, verbose=0)

# y_pred_rounded = np.round(y_pred, 0)

# y_pred_rounded = abs(pd.Series(y_pred_rounded.reshape(y_pred.shape[0])))


# import tensorflow as tf
# m = tf.keras.metrics.Precision()
# m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
# m.result().numpy()

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# def plot_cm(labels, predictions, p=0.5):
#     cm = confusion_matrix(labels, predictions > p)
#     plt.figure(figsize=(5,5))
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title('Confusion matrix @{:.2f}'.format(p))
#     plt.ylabel('Actual label')
#     plt.xlabel('Predicted label')

#     print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
#     print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
#     print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
#     print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
#     print('Total Fraudulent Transactions: ', np.sum(cm[1]))


# In[28]:

y_pred = bestModel.predict (X_test)

y_pred = np.round(y_pred, 0)

y_pred = abs(pd.Series(y_pred.reshape(y_pred.shape[0])))



print('y_test:', y_test)

###############################################################################
## Obtain metrics from the above model 
###############################################################################
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


## Giving the output
f = open("output_rnn.txt","a")

f.write('\n\n\nrnn Metrics: Random State ==')
f.write(str(STATE))
# Precision == TP / (TP + FP)
precision = precision_score(y_test, y_pred)
print('Precision Score: ', precision)
f.write('\nPrecision:')
f.write(str(precision))

# Recall == TP / (TP + FN)
recall = recall_score(y_test, y_pred)
print('Recall Score: ', recall_score(y_test, y_pred))
f.write('\nRecall:')
f.write(str(precision))

# Accuracy 
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)
f.write('\nAccuracy:')
f.write(str(accuracy))

# f1 
f_one_score = f1_score(y_test, y_pred)
print('F1 Score: ', f_one_score)
f.write('\nf_one_score:')
f.write(str(f_one_score))

cohen = str(cohen_kappa_score(y_test, y_pred))
print('Cohen Kappa Score: ', cohen)
f.write('\nCohen: ')
f.write(str(cohen))


f.write('\nMade in ')
f.write(str(training_time))
f.write(' seconds\n')

# Multilabel Confusion Matrix: 
# [tn fp]
# [fn tp]
confusion_matrix = str(multilabel_confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(confusion_matrix)
f.write('\nCofusion Matrix: ')
f.write(confusion_matrix)

f.close()








# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:


# # Author: Ernesto Rodríguez
# # github.com/ernestorodg

# ###############################################################################
# ## Analyse Bezerra's dataset for intrusion detection using Decision Trees
# ###############################################################################


# # In[2]:


# import pandas as pd
# import numpy as np
# import sys

# ###############################################################################
# ## Define constants 
# ###############################################################################


# # Random state for reproducibility
# STATE = 0
# np.random.seed(10)
# # List of available attacks on the dataset


# # Especific to the repository 
# DATASET_DIRECTORY = r'../datasets/Dataset-bezerra-IoT-20200528T203526Z-001/Dataset-IoT/'
# NETFLOW_DIRECTORY = r'NetFlow/'


# # There are different csv files on the Dataset, with different types of data:

# # Some meanings:
# # MC: Media Center
# # I: One hour of legitimate and malicious NetFlow data from profile.
# # L: One hour of legitimate NetFlow data from profile.

# MC = r'MC/'
# ST = r'ST/'
# SC = r'SC/'


# # MC_I_FIRST: Has infected data by Hajime, Aidra and BashLite botnets 
# MC_I_FIRST = r'MC_I1.csv'

# # MC_I_SECOND: Has infected data from Mirai botnets
# MC_I_SECOND = r'MC_I2.csv'

# # MC_I_THIR: Has infected data from Mirai, Doflo, Tsunami and Wroba botnets
# MC_I_THIRD = r'MC_I3.csv'

# # MC_L: Has legitimate data, no infection
# MC_L = r'MC_L.csv'


# # Constants for ST
# ST_I_FIRST = r'ST_I1.csv'
# ST_I_SECOND = r'ST_I2.csv'
# ST_I_THIRD = r'ST_I3.csv'
# ST_L = r'ST_L.csv'

# # Constants for SC
# SC_I_FIRST = r'SC_I1.csv'
# SC_I_SECOND = r'SC_I2.csv'
# SC_I_THIRD = r'SC_I3.csv'
# SC_L = r'SC_L.csv'


# # In[3]:


# ###############################################################################
# ## Load dataset
# ###############################################################################

# # For MC data:
# df_mc_I_first = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_FIRST)
# df_mc_I_second = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_SECOND)
# df_mc_I_third = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_THIRD)

# # Add legitimate rows from MC_L
# legitimate_frame_mc = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_L)

# ###################

# # For ST data:
# df_st_I_first = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_FIRST)
# df_st_I_second = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_SECOND)
# df_st_I_third = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_THIRD)

# # Add legitimate rows from SC_L
# legitimate_frame_st = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_L)


# ###################

# # For SC data:
# df_sc_I_first = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_FIRST)
# df_sc_I_second = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_SECOND)
# df_sc_I_third = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_THIRD)

# # Add legitimate rows from MC_L
# legitimate_frame_sc = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_L)

# dataframes_list = [df_mc_I_first,
#                 df_mc_I_second,
#                 df_mc_I_third,
#                 legitimate_frame_mc,
#                 df_st_I_first,
#                 df_st_I_second,
#                 df_st_I_third,
#                 legitimate_frame_st,
#                 df_sc_I_first,
#                 df_sc_I_second,
#                 df_sc_I_third,
#                 legitimate_frame_sc]

# # Joining the differents DataFrames
# prev_df = pd.concat(dataframes_list)


# # In[4]:


# prev_df


# # In[5]:


# ###############################################################################
# ## Modify the DataFrame
# ###############################################################################


# # Sample the dataset if necessary
# # df = prev_df.sample (frac = 0.1, replace = True, random_state = 0)
# # Not taking a random sample:
# df = prev_df.iloc[:100588, :]


# # We can see that this dataset has a temporal description.
# # So it is not a good idea to randomly remove rows if using RNN

# # In this case we drop the index column, since pandas library creates an index
# # automatically. 
# df = df.drop(df.columns[0], axis=1)

# # Also drop columns that has no significant data
# df = df.drop(df.columns[14:], axis=1)

# df = df.drop(['ts', 'te'], axis=1)

# # Trying another drops to see relation between features and results
# df = df.drop(['fwd', 'stos', 'sa', 'da'], axis=1)
# # 'sp', 'dp', 'sa',  'da',  

# # Counting number of null data
# nanColumns = [i for i in df.columns if df [i].isnull ().any ()]

# # Remove NaN and inf values
# df.replace ('Infinity', np.nan, inplace = True) ## Or other text values
# df.replace (np.inf, np.nan, inplace = True) ## Remove infinity
# df.replace (np.nan, 0, inplace = True)


# # In[6]:


# ###############################################################################
# ## Slice the dataframe (usually the last column is the target)
# ###############################################################################

# X = pd.DataFrame(df.iloc [:, 1:])

# # Selecting other columns
# # X = pd.concat([X, df.iloc[:, 2]], axis=1)

# y = df.iloc [:, 0]
# print('Number of non-attacks: ', y.value_counts()[0])
# print('Number of attacks: ', y.value_counts()[1])

# # See Output, only available on jupyter-notebooks
# # X


# # In[7]:


# ###############################################################################
# ## Create artificial non-attacks samples using Random Oversampling
# ###############################################################################

# from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE

# ros = RandomOverSampler(random_state=42)

# X, y = ros.fit_resample(X, y)

# print('Number of non-attacks: ', y.value_counts()[0])
# print('Number of attacks: ', y.value_counts()[1])


# # In[8]:


# # ###############################################################################
# # ## Create artificial non-attacks samples using Random undersampling
# # ###############################################################################

# # from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE

# # ros = RandomUnderSampler(random_state=42)

# # X, y = ros.fit_resample(X, y)

# # print('Number of non-attacks: ', y.value_counts()[0])
# # print('Number of attacks: ', y.value_counts()[1])


# # In[9]:


# ####################################################################
# # Treating categorical data before splitting the dataset into the differents sets
# ####################################################################
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OrdinalEncoder

# cat_cols = X.columns[X.dtypes == 'O'] # Returns array with the columns that has Object types elements

# # Check wether cat_cols is empty or not. If it is empty, do not do anything
# if list(cat_cols):
#     categories = [
#         X[column].unique() for column in X[cat_cols]]

#     for cat in categories:
#         cat[cat == None] = 'missing'  # noqa

#     # Replacing missing values
#     categorical_imputer = SimpleImputer(missing_values=None, 
#                                         strategy='constant', 
#                                         fill_value='missing')

#     X[cat_cols] = categorical_imputer.fit_transform(X[cat_cols])

#     # Encoding the categorical data
#     categorical_encoder = OrdinalEncoder(categories = categories)
#     categorical_encoder.fit(X[cat_cols])
#     X[cat_cols] = categorical_encoder.transform(X[cat_cols])


# # In[10]:


# ###############################################################################
# ## Split dataset into train and test sets if not using cross validation
# ###############################################################################
# from sklearn.model_selection import train_test_split
# TEST_SIZE = 1/5
# VALIDATION_SIZE = 1/5


# X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = TEST_SIZE,
#                                                      random_state = STATE)


# print ('\nSplitting dataset (validation/train):', 1/5)
# X_train_val, X_val, y_train_val, y_val = train_test_split (
#                                              X_train,
#                                              y_train,
#                                              test_size = VALIDATION_SIZE,
#                                              random_state = STATE)


# X_train = pd.DataFrame(X_train)
# X_test = pd.DataFrame(X_test)
# X_train_val = pd.DataFrame(X_train_val)
# X_val = pd.DataFrame(X_val)


# # In[11]:


# ####################################################################
# # Treat numerical data 
# ####################################################################
# from sklearn.preprocessing import StandardScaler


# num_cols = X_train.columns[(X_train.dtypes == 'float64') | (X_train.dtypes == 'int64')] # Returns array with the columns that has float types elements

# # Scaling numerical values

# numerical_scaler = StandardScaler()
# numerical_scaler.fit(X_train)
# X_train = numerical_scaler.transform(X_train)

# X_test = numerical_scaler.transform(X_test)

# # X_train


# # In[12]:


# y_test.value_counts()


# # In[13]:


# from numpy import array
# LENGTH = 5

# sets_list = [X_train, X_test]
# for index, data in enumerate(sets_list):
#     n = data.shape[0]
#     samples = []

#     # step over the X_train.shape[0] (samples) in jumps of 200 (time_steps)
#     for i in range(0,n,LENGTH):
#         # grab from i to i + 200
#         sample = data[i:i+LENGTH]
#         samples.append(sample)

#     # convert list of arrays into 2d array
#     new_data = list()
#     new_data = np.array(new_data)
#     for i in range(len(samples)):
#         new_data = np.append(new_data, samples[i])
        
#     sets_list[index] = new_data.reshape(len(samples), LENGTH, data.shape[1])
     
    
# X_train = sets_list[0]
# X_test = sets_list[1]


# # In[ ]:





# # In[14]:


# answer_list = [y_train, y_test]
# for index, answer in enumerate(answer_list):

#     new_answer = list()
#     new_answer = np.array(new_answer)
#     answer = np.array(answer)
#     for i in range (0, len(answer), LENGTH):
#         new_answer = np.append(new_answer, answer[i])
#     answer_list[index] = new_answer


# # In[15]:


# ## Preparing the test to see performance
# y_train = answer_list[0]
# y_test = answer_list[1]


# # In[16]:


# pd.Series(y_train).value_counts()


# # In[ ]:





# # In[17]:


# y_train.shape


# # In[18]:


# # univariate lstm

# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense

# # define model
# model = Sequential()
# model.add(LSTM(50, activation= 'relu' , input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(1))
# model.compile(optimizer= 'adam' , loss= 'mse' )


# # In[19]:


# import time
# # fit model

# # Measure time of this training
# start_time = time.time()

# model.fit(X_train, y_train, epochs=50, verbose=0)

# print("--- %s seconds ---" % (time.time() - start_time))


# # In[20]:


# y_pred = model.predict(X_test, verbose=0)
# print(y_pred)


# # In[21]:


# pd.Series(y_train).value_counts()


# # In[22]:


# y_pred_rounded = np.round(y_pred, 0)


# # In[23]:


# y_pred_rounded.shape


# # In[24]:


# y_pred_rounded = pd.Series(y_pred_rounded.reshape(8000))


# # In[30]:





# # In[26]:


# import tensorflow as tf
# m = tf.keras.metrics.Precision()
# m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
# m.result().numpy()


# import matpyp..lot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# plot_cm(labels, predictions, p=0.5):
# elot_cm(labels, predictions, p=0.5):
#     cm = confusion_matrix(labels, predictions > p)
#     plt.figure(figsize=(5,5))
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title('Confusion matrix @{:.2f}'.format(p))
#     plt.ylabel('Actual label')
#     plt.xlabel('Predicted label')

#     print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
#     print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
#     print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
#     print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
#     print('Total Fraudulent Transactions: ', np.sum(cm[1]))


# # In[29]:


# y_test, y_pred_rounded)


# # In[34]:


# ###############################################################################
# ## Obtain metrics from the above model 
# ###############################################################################
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import multilabel_confusion_matrix
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix


# # Predicting from the test slice
# y_pred = y_pred_rounded

# ## Giving the output
# f = open("output_decision_tree.txt","a")

# f.write('\n\n\ndecision_tree Metrics: Random State ==')
# f.write(str(STATE))
# # Precision == TP / (TP + FP)
# precision = precision_score(y_test, y_pred)
# print('Precision Score: ', precision)
# f.write('\nPrecision:')
# f.write(str(precision))

# # Recall == TP / (TP + FN)
# recall = recall_score(y_test, y_pred)
# print('Recall Score: ', recall_score(y_test, y_pred))
# f.write('\nRecall:')
# f.write(str(precision))

# # Accuracy 
# train_score = model.score(X_test, y_test)
# print('Accuracy: ', train_score)
# f.write('\nAccuracy:')
# f.write(str(train_score))

# # f1 
# f_one_score = f1_score(y_test, y_pred)
# print('F1 Score: ', f_one_score)
# f.write('\nf_one_score:')
# f.write(str(f_one_score))

# cohen = str(cohen_kappa_score(y_test, y_pred))
# print('Cohen Kappa Score: ', cohen)
# f.write('\nCohen: ')
# f.write(str(cohen))


# f.write('\nMade in ')
# f.write(str(training_time))
# f.write(' seconds\n')

# # Multilabel Confusion Matrix: 
# # [tn fp]
# # [fn tp]
# confusion_matrix = str(multilabel_confusion_matrix(y_test, y_pred, labels=[0, 1]))
# print(confusion_matrix)
# f.write('\nCofusion Matrix: ')
# f.write(confusion_matrix)

# f.close()




