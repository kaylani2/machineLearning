# Authors: Kaylani Bochie and Ernesto Rodríguez
# github.com/kaylani2
# github.com/ernestorodg

###############################################################################
## Analyse Bezerra's dataset for intrusion detection using Decision Trees
###############################################################################

# We import everything used on all the codes, so it is easier to scale or reproduce.

import sys
import time
import pandas as pd
import os
import math
import numpy as np
from numpy import mean, std
from unit import remove_columns_with_one_value, remove_nan_columns, load_dataset
from unit import display_general_information, display_feature_distribution
from collections import Counter
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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import keras.utils
from keras import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from keras.optimizers import RMSprop, Adam
from keras.constraints import maxnorm

###############################################################################
## Define constants
###############################################################################


# Random state for reproducibility
try:
  # If defined at argv:
  STATE = int (sys.argv [1])
except:
  # If not defined, it will be 0
  STATE = 0
np.random.seed (10)
# List of available attacks on the dataset
STEPS = 10
print ('STATE:', STATE)


TARGET = 'Label'

####################################################################
# Load the dataset
####################################################################


df = load_dataset ()
print ("Data Loaded")
remove_columns_with_one_value (df, verbose = False)
remove_nan_columns (df, 0.6, verbose = False)
#making the final DataFrame
#dropping the number of the rows column
df = df.drop (df.columns [0], axis = 1)

#dropping unrelated columns
df.drop (axis = 'columns', columns= ['ts', 'te', 'sa', 'da'], inplace = True)


###############################################################################
## Slice the dataframe (usually the last column is the target)
###############################################################################

X = pd.DataFrame (df.iloc [:, 1:])

# For selecting other columns do the following:
# X = pd.concat ( [X, df.iloc [:, 2]], axis=1)

y = df.iloc [:, 0]
print ('Number of non-attacks: ', y.value_counts () [0])
print ('Number of attacks: ', y.value_counts () [1])





###############################################################################
## Modify number of samples if necessary
###############################################################################


# from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
# from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE

# if TRAFFIC == 0:
#     # Oversampling
#     ros = RandomOverSampler (random_state=42)

#     X, y = ros.fit_resample (X, y)

#     print ('Number of non-attacks: ', y.value_counts () [0])
#     print ('Number of attacks: ', y.value_counts () [1])
# else:
#     # Undersampling
#     ros = RandomUnderSampler (random_state=42)

#     X, y = ros.fit_resample (X, y)

#     print ('Number of non-attacks: ', y.value_counts () [0])
#     print ('Number of attacks: ', y.value_counts () [1])


###############################################################################
### Encode categorical features
print ('Encoding categorical features (ordinal encoding).')
my_encoder = OrdinalEncoder ()
df ['flg'] = my_encoder.fit_transform (df ['flg'].values.reshape (-1, 1))
df ['pr'] = my_encoder.fit_transform (df ['pr'].values.reshape (-1, 1))
print ('Objects:', list (df.select_dtypes ( ['object']).columns))


###############################################################################
## Quick sanity check
###############################################################################
display_general_information (df)


###############################################################################
## Split dataset into train and test sets
###############################################################################
### Dataset too big? Drop, uncomment the next lines.
# drop_indices = np.random.choice (df.index, int (df.shape [0] * 0.5),
#                                  replace = False)
# df = df.drop (drop_indices)
TEST_SIZE = 3/10
VALIDATION_SIZE = 1/4
print ('Splitting dataset (test/train):', TEST_SIZE)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                               df.loc [:, df.columns != TARGET],
                                               df [TARGET],
                                               test_size = TEST_SIZE,
                                               random_state = STATE,)
print ('Splitting dataset (validation/train):', VALIDATION_SIZE)
X_train_df, X_val_df, y_train_df, y_val_df = train_test_split (
                                             X_train_df,
                                             y_train_df,
                                             test_size = VALIDATION_SIZE,
                                             random_state = STATE,)
print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_val_df shape:', X_val_df.shape)
print ('y_val_df shape:', y_val_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)

X_train_df.sort_index (inplace = True)
y_train_df.sort_index (inplace = True)
X_val_df.sort_index (inplace = True)
y_val_df.sort_index (inplace = True)
X_test_df.sort_index (inplace = True)
y_test_df.sort_index (inplace = True)



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


###############################################################################
## Apply normalization
###############################################################################
### NOTE: Only use derived information from the train set to avoid leakage.
print ('\nApplying normalization.')
startTime = time.time ()
scaler = StandardScaler ()

scaler.fit (X_train)
X_train = scaler.transform (X_train)
X_val = scaler.transform (X_val)
X_test = scaler.transform (X_test)
print (str (time.time () - startTime), 'to normalize data.')



###############################################################################
## Rearrange samples for RNN
###############################################################################
print ('\nRearranging dataset for the RNN.')
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_val shape:', X_val.shape)
print ('y_val shape:', y_val.shape)
print ('y_test shape:', y_test.shape)

STEPS = 3
FEATURES = X_train.shape [1]
def window_stack (a, stride = 1, numberOfSteps = 3):
    return np.hstack ( [ a [i:1+i-numberOfSteps or None:stride] for i in range (0,numberOfSteps) ])

X_train = window_stack (X_train, stride = 1, numberOfSteps = STEPS)
X_train = X_train.reshape (X_train.shape [0], STEPS, FEATURES)
X_val = window_stack (X_val, stride = 1, numberOfSteps = STEPS)
X_val = X_val.reshape (X_val.shape [0], STEPS, FEATURES)
X_test = window_stack (X_test, stride = 1, numberOfSteps = STEPS)
X_test = X_test.reshape (X_test.shape [0], STEPS, FEATURES)

y_train = y_train [ (STEPS - 1):]
y_val = y_val [ (STEPS - 1):]
y_test = y_test [ (STEPS - 1):]

print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_val shape:', X_val.shape)
print ('y_val shape:', y_val.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)


################################################################################
### Create learning model and tune hyperparameters
################################################################################
'''

### K: One hot encode the output.
#numberOfClasses = len (df [TARGET].unique ())
#print ('y_val:')
#print (y_val [:50])
#print (y_val.shape)
#y_train = keras.utils.to_categorical (y_train, numberOfClasses)
#y_val = keras.utils.to_categorical (y_val, numberOfClasses)
#y_test = keras.utils.to_categorical (y_test, numberOfClasses)

### -1 indices -> train
### 0  indices -> validation
test_fold = np.repeat ( [-1, 0], [X_train.shape [0], X_val.shape [0]])
myPreSplit = PredefinedSplit (test_fold)


print ('y_val:')
print (y_val [:50])
print (y_val.shape)
#y_val = y_val.argmax (axis = 1)
print ('y_val:')
print (y_val [:50])
print (y_val.shape)
#y_train = y_train.argmax (axis = 1)


def create_model (learn_rate = 0.01, dropout_rate = 0.0, weight_constraint = 0, units = 50):
 model = Sequential ()
 model.add (LSTM (units = units, activation = 'relu' , input_shape= (X_train.shape [1], X_train.shape [2])))
 model.add (Dense (1, activation = 'sigmoid'))
 model.compile (optimizer = 'adam', loss = 'binary_crossentropy',)
 return model
model = KerasClassifier (build_fn = create_model, verbose = 2)
batch_size = [5000, 1000]#10, 30, 50]
epochs = [5]#, 5, 10]
learn_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
weight_constraint = [0]#, 2, 3, 4, 5]
units = [10, 50, 100]
param_grid = dict (batch_size = batch_size, epochs = epochs,
                  dropout_rate = dropout_rate, learn_rate = learn_rate,
                  weight_constraint = weight_constraint, units = units)
grid = GridSearchCV (estimator = model, param_grid = param_grid,
                    scoring = 'f1_weighted', cv = myPreSplit, verbose = 2,
                    n_jobs = -1)
grid_result = grid.fit (np.concatenate ((X_train, X_val), axis = 0),
                       np.concatenate ((y_train, y_val), axis = 0))
print (grid_result.best_params_)
print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_ ['mean_test_score']
stds = grid_result.cv_results_ ['std_test_score']
params = grid_result.cv_results_ ['params']
for mean, stdev, param in zip (means, stds, params):
 print ("%f (%f) with: %r" % (mean, stdev, param))
sys.exit ()

'''



###############################################################################
## Finished model
METRICS = [keras.metrics.TruePositives (name = 'TP'),
           keras.metrics.FalsePositives (name = 'FP'),
           keras.metrics.TrueNegatives (name = 'TN'),
           keras.metrics.FalseNegatives (name = 'FN'),
           keras.metrics.BinaryAccuracy (name = 'Acc.'),
           keras.metrics.Precision (name = 'Prec.'),
           keras.metrics.Recall (name = 'Recall'),
           keras.metrics.AUC (name = 'AUC'),]
BATCH_SIZE = 1000
NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.0
clf = Sequential ()
clf.add (LSTM (100, activation = 'relu', #return_sequences = True,
                     input_shape = (X_train.shape [1], X_train.shape [2])))
clf.add (Dropout (DROPOUT_RATE))
#clf.add (LSTM (50, activation='relu'))
clf.add (Dense (1, activation = 'sigmoid'))

print ('Model summary:')
clf.summary ()

###############################################################################
## Compile the network
###############################################################################
print ('\nCompiling the network.')
clf.compile (optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = METRICS)


###############################################################################
## Fit the network
###############################################################################
print ('\nFitting the network.')
startTime = time.time ()
history = clf.fit (X_train, y_train,
                         batch_size = BATCH_SIZE,
                         epochs = NUMBER_OF_EPOCHS,
                         verbose = 2, #1 = progress bar, not useful for logging
                         workers = 0,
                         use_multiprocessing = True,
                         #class_weight = 'auto',
                         validation_data = (X_val, y_val))
#clf.fit (X_train, y_train, epochs = NUMBER_OF_EPOCHS,
               #use_multiprocessing = True, verbose = 2)
print (str (time.time () - startTime), 's to train model.')


###############################################################################
## Analyze results
###############################################################################

### K: Only before publishing... Don't peek.
print ('\nPerformance on TEST set:')
y_pred = clf.predict (X_test)
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
