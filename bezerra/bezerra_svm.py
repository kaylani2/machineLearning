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


# In[67]:


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


####################################################################
# Treating categorical data before splitting the dataset into the differents sets
####################################################################
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from numpy import empty

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


# In[71]:


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


# In[72]:


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

X_val = numerical_scaler.transform(X_val)

# X_train




###############################################################################
## Create learning model and tune hyperparameters
###############################################################################
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
import time

# Because this section was already made
# # Defining the classifier 
# clf = SVC()


# ### -1 indices -> train
# ### 0  indices -> validation
# test_fold = np.repeat ([-1, 0], [X_train_val.shape [0], X_val.shape [0]])
# myPreSplit = PredefinedSplit (test_fold)
# #myPreSplit.get_n_splits ()
# #myPreSplit.split ()
# #for train_index, test_index in myPreSplit.split ():
# #    print ("TRAIN:", train_index, "TEST:", test_index)

# # Best: {'C': 0.009, 'gamma': 1, 'kernel': 'linear'}


# parameters = {'C' : [0.001,.009,0.01,.09,1,5,10,25,50],
#               'kernel' : ['linear', 'rbf'],
#               'gamma' : [1, 0.1, 0.01]}
# clf = SVC()
# model = GridSearchCV (estimator = clf,
#                       param_grid = parameters,
#                       scoring = 'f1_weighted',
#                       cv = myPreSplit,
#                       verbose = 1)

# model.fit (np.concatenate ((X_train_val, X_val), axis = 0),
#            np.concatenate ((y_train_val, y_val), axis = 0))

# print (model.best_params_)


# # In[79]:


# # ###############################################################################
# # ## Obtain metrics from the validation model 
# # ###############################################################################

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


# y_pred = model.predict(X_test)

# # New Model Evaluation metrics 
# print('Parameters for tuned model: ')
# print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
# print('Precision Score : ' + str(precision_score(y_test,y_pred)))
# print('Recall Score : ' + str(recall_score(y_test,y_pred)))
# print('F1 Score : ' + str(f1_score(y_test,y_pred)))
# print('Cohen Kappa Score: ', str(cohen_kappa_score(y_test, y_pred)))


# #Logistic Regression (Grid Search) Confusion matrix
# confusion_matrix(y_test,y_pred)
# print('\n\n\n\n')


# # In[80]:


# ###############################################################################
# ## Plotting confusion matrix
# ###############################################################################
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

# plot_confusion_matrix(model, X_test, y_test)  # doctest: +SKIP
# plt.savefig("svm_tuned.png", format="png")
# plt.show()  # doctest: +SKIP
# # td  sp  dp  pr  flg  ipkt ibyt




# In[81]:


###############################################################################
## Train the model with other parameters
###############################################################################

# Measure time of this training
start_time = time.time()

# Assign the model to be used with adjusted parameters
clf = SVC()

# Training the model
model = clf.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
training_time = time.time() - start_time

# In[82]:


###############################################################################
## Obtain metrics from the above model 
###############################################################################
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# Predicting from the test slice
y_pred = model.predict(X_test)


## Giving the output
f= open("output_svm.txt","a")

f.write('\n\nsvm Metrics: Random State ==')
f.write(str(STATE))

# Precision == TP / (TP + FP)
precision = precision_score(y_test, y_pred)
print('Precision Score: ', precision)
f.write('\nPrecision: ')
f.write(str(precision))

# Recall == TP / (TP + FN)
recall = recall_score(y_test, y_pred)
print('Recall Score: ', recall_score(y_test, y_pred))
f.write('\nRecall: ')
f.write(str(precision))

# Accuracy 
train_score = model.score(X_test, y_test)
print('Accuracy: ', train_score)
f.write('\nAccuracy: ')
f.write(str(train_score))

# f1 
f_one_score = f1_score(y_test, y_pred)
print('F1 Score: ', f_one_score)
f.write('\nf_one_score: ')
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
# In[83]:


###############################################################################
## Plotting confusion matrix
###############################################################################

plot_confusion_matrix(model, X_test, y_test)  # doctest: +SKIP
plt.savefig("svm.png", format="png")
plt.show()  # doctest: +SKIP
# td  sp  dp  pr  flg  ipkt ibyt


