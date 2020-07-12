# Author: Ernesto Rodríguez
# github.com/ernestorodg

###############################################################################
## Analyse mqtt-iot's dataset for intrusion detection using rnn
###############################################################################

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

# LENGTH will be the length of the made step to reshape the input matrix 
LENGTH = 200


###############################################################################
## Define dataset files 
###############################################################################
 
# Directories (Specific to the repository) 
DATASET_DIRECTORY = r'../../../datasets/mqtt-iot/'
NETFLOW_DIRECTORY = r'uniflow_features/'


# There are different csv files on the directory:

# Some meanings:
# bruteforce: bruteforce attacks flow
# normal: normal traffic
# scan_A: scan attack
# sparta: sparta attach
# scan_sU: scan attack
 
NORMAL_FLOW = r'uniflow_normal.csv'

BRUTEFORCE = r'uniflow_mqtt_bruteforce.csv'


###############################################################################
## Load dataset
###############################################################################

# For MC data:
df_normal = pd.read_csv (DATASET_DIRECTORY + NETFLOW_DIRECTORY + NORMAL_FLOW)
df_malign = pd.read_csv (DATASET_DIRECTORY + NETFLOW_DIRECTORY + BRUTEFORCE)

dataframes_list = [df_normal, df_malign]

# Joining the differents DataFrames
prev_df = pd.concat(dataframes_list, ignore_index=True)

print(prev_df.describe(include='all'))
print(prev_df)

###############################################################################
## Modify the DataFrame
###############################################################################


# Sample the dataset if necessary
df = prev_df.sample (frac = 1, replace = True, random_state = 0)

# We can see that this dataset has a temporal description.
# So it is not a good idea to randomly remove rows if using RNN

# Trying another drops to see relation between features and results
df = df.drop(['ip_src', 'ip_dst'], axis=1)

# Counting number of null data
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]

# Remove NaN and inf values
df.replace ('Infinity', np.nan, inplace = True) ## Or other text values
df.replace (np.inf, np.nan, inplace = True) ## Remove infinity
df.replace (np.nan, 0, inplace = True)

# If it is not possible to use imblearn, try using the code below
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

# The number of samples must be divisible by LENGTH to the implemented loops work
TRAFFIC = 1  # Oversampling will be done
# TRAFFIC = 1 # Undersampling will be done

print('Number of non-attacks before dropping: ', int(df.is_attack.value_counts()[TRAFFIC]))
remove_n = int(df.is_attack.value_counts()[TRAFFIC]) % LENGTH 
print('Removing ', remove_n, ' rows...')
drop_indices = list(df.index[df.is_attack == TRAFFIC])
drop_indices = drop_indices[0:remove_n]
df = df[~np.in1d(np.arange(len(df)), drop_indices)]
print('Number of non-attacks after dropping: ', int(df.is_attack.value_counts()[TRAFFIC]))
print('\n')


###############################################################################
## Slice the dataframe (usually the last column is the target)
###############################################################################

X = pd.DataFrame(df.iloc [:, :-1])

# Selecting other columns
# X = pd.concat([X, df.iloc[:, 2]], axis=1)

y = df.iloc [:, -1]
print('Values before Undersampling or Oversampling')
print('Number of non-attacks: ', y.value_counts()[0])
print('Number of attacks: ', y.value_counts()[1])
print('\n\n\n')


###############################################################################
## Modify number of samples
###############################################################################


from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE

if TRAFFIC == 0: 
    # Oversampling
    ros = RandomOverSampler(random_state=42)

    X, y = ros.fit_resample(X, y)

    print('Number of non-attacks: ', y.value_counts()[0])
    print('Number of attacks: ', y.value_counts()[1])
else: 
    # Undersampling
    ros = RandomUnderSampler(random_state=42)

    X, y = ros.fit_resample(X, y)

    print('Number of non-attacks: ', y.value_counts()[0])
    print('Number of attacks: ', y.value_counts()[1])




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


# answer_list = [y_train, y_test]
# for index, answer in enumerate(answer_list):

#     new_answer = list()
#     new_answer = np.array(new_answer)
#     answer = np.array(answer)
#     for i in range (0, len(answer), LENGTH):
#         new_answer = np.append(new_answer, answer[i])
#     answer_list[index] = new_answer


# def window_stack(a, stepsize=LENGTH, width=3):
#     n = a.shape[0]
#     return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

# ## Preparing the test to see performance
# y_train = answer_list[0]
# y_test = answer_list[1]


# # In[17]:


# pd.Series(y_train).value_counts()

STEPS = 10
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

# define model
model = Sequential()
model.add(LSTM(50, activation= 'relu' , input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer= 'adam' , loss= 'mse' )


# In[20]:


import time
# fit model

# Measure time of this training
start_time = time.time()

model.fit(X_train, y_train, epochs=50, verbose=0)

print("--- %s seconds ---" % (time.time() - start_time))


# In[21]:


y_pred = model.predict(X_test, verbose=0)
print(y_pred)


# In[22]:


pd.Series(y_train).value_counts()


# In[31]:


y_pred_rounded = np.round(y_pred, 0)


# In[32]:


y_pred_rounded.shape


# In[24]:


y_pred_rounded = abs(pd.Series(y_pred_rounded.reshape(y_pred.shape[0])))


# In[30]:


y_pred.shape


# In[25]:


import tensorflow as tf
m = tf.keras.metrics.Precision()
m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
m.result().numpy()


# In[26]:


np.array(y_test)


# In[27]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


# In[28]:


plot_cm(y_test, y_pred_rounded)

print('y_test:', y_test)
print('y_pred_rounded', y_pred_rounded)
###############################################################################
## Analyze results
###############################################################################
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score

print ('\nPerformance on Test set:')
print ('Confusion matrix:')
print ('Accuracy:', accuracy_score (y_test, y_pred_rounded))
print ('Precision:', precision_score (y_test, y_pred_rounded))
print ('Recall:', recall_score (y_test, y_pred_rounded))
print ('F1:', f1_score (y_test, y_pred_rounded))
print ('Cohen Kappa:', cohen_kappa_score (y_test, y_pred_rounded))
# In[ ]:




