# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Support Vector Machine

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


###############################################################################
## Define constants
###############################################################################
# Random state for reproducibility
STATE = 0
np.random.seed (STATE)

pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 5)

BOT_IOT_DIRECTORY = '../../../../datasets/bot-iot/'
BOT_IOT_FEATURE_NAMES = 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv'
BOT_IOT_FILE_5_PERCENT_SCHEMA = 'UNSW_2018_IoT_Botnet_Full5pc_{}.csv' # 1 - 4
FIVE_PERCENT_FILES = 4
BOT_IOT_FILE_FULL_SCHEMA = 'UNSW_2018_IoT_Botnet_Dataset_{}.csv' # 1 - 74
FULL_FILES = 74
FILE_NAME = BOT_IOT_DIRECTORY + BOT_IOT_FILE_5_PERCENT_SCHEMA#FULL_SCHEMA
FEATURES = BOT_IOT_DIRECTORY + BOT_IOT_FEATURE_NAMES
NAN_VALUES = ['?', '.']
TARGET = 'category'

###############################################################################
## Load dataset
###############################################################################
featureDf = pd.read_csv (FEATURES)
featureColumns = featureDf.columns.to_list ()
featureColumns = [f.strip () for f in featureColumns]

print ('Reading', FILE_NAME.format (str (1)))
df = pd.read_csv (FILE_NAME.format ('1'), #names = featureColumns,
                  index_col = 'pkSeqID', dtype = {'pkSeqID' : np.int32},
                  na_values = NAN_VALUES, low_memory = False)

for fileNumber in range (2, FIVE_PERCENT_FILES + 1):#FULL_FILES + 1):
  print ('Reading', FILE_NAME.format (str (fileNumber)))
  aux = pd.read_csv (FILE_NAME.format (str (fileNumber)),
                     #names = featureColumns,
                     index_col = 'pkSeqID',
                     dtype = {'pkSeqID' : np.int32}, na_values = NAN_VALUES,
                     low_memory = False)
  df = pd.concat ([df, aux])


###############################################################################
## Display generic (dataset independent) information
###############################################################################
#print ('Dataframe shape (lines, columns):', df.shape, '\n')
#print ('First 5 entries:\n', df [:5], '\n')
#print ('entries:\n', df [4000000//4 - 5:4000000//4 + 5], '\n')
df.info (verbose = False) # Make it true to find individual atribute types

print ('\nDataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('Number of NaN columns:', len (nanColumns))
print ('NaN columns:', nanColumns, '\n')


###############################################################################
## Display specific (dataset dependent) information
###############################################################################
print ('\nAttack types:', df ['attack'].unique ())
print ('Attack distribution:')
print (df ['attack'].value_counts ())
print ('\nCateogry types:', df [TARGET].unique ())
print ('Cateogry distribution:')
print (df [TARGET].value_counts ())
print ('\nSubcategory types:', df ['subcategory'].unique ())
print ('Subcategory distribution:')
print (df ['subcategory'].value_counts ())


###############################################################################
## Data pre-processing
###############################################################################
#df.replace (['NaN', 'NaT'], np.nan, inplace = True)
#df.replace ('?', np.nan, inplace = True)
#df.replace ('Infinity', np.nan, inplace = True)

###############################################################################
### Remove columns with only one value
print ('\nColumn | # of different values')
# nUniques = df.nunique () ### K: Takes too long. WHY?
nUniques = []
for column in df.columns:
  nUnique = df [column].nunique ()
  nUniques.append (nUnique)
  print (column, '|', nUnique)

print ('\nRemoving attributes that have only one (or zero) sampled value.')
for column, nUnique in zip (df.columns, nUniques):
  if (nUnique <= 1): # Only one value: DROP.
    df.drop (axis = 'columns', columns = column, inplace = True)

print ('\nColumn | # of different values')
for column in df.columns:
  nUnique = df [column].nunique ()
  print (column, '|', nUnique)

###############################################################################
### Remove NaN columns (with a lot of NaN values)
print ('\nColumn | NaN values')
print (df.isnull ().sum ())
print ('Removing attributes with more than half NaN values.')
df = df.dropna (axis = 'columns', thresh = df.shape [0] // 2)
print ('Dataframe contains NaN values:', df.isnull ().values.any ())
print ('\nColumn | NaN values (after dropping columns)')
print (df.isnull ().sum ())

###############################################################################
### Input missing values
### K: Look into each attribute to define the best inputing strategy.
### K: NOTE: This must be done after splitting to dataset to avoid data leakge.
columsWithMissingValues = ['sport', 'dport']
### K: Examine values.
for column in df.columns:
  nUnique = df [column].nunique ()
for column, nUnique in zip (df.columns, nUniques):
    if (nUnique < 5):
      print (column, df [column].unique ())
    else:
      print (column, 'unique values:', nUnique)

# sport  unique values: 91168     # most_frequent?
# dport  unique values: 115949    # most_frequent?
imputingStrategies = ['most_frequent', 'most_frequent']


###############################################################################
### Handle categorical values
### K: Look into each attribute to define the best encoding strategy.
df.info (verbose = False)
### K: dtypes: float64(11), int64(8), object(9)
myObjects = list (df.select_dtypes ( ['object']).columns)
print ('\nObjects:', myObjects, '\n')
### K: Objects:
  # 'flgs',
  # 'proto',
  # 'saddr',
  # 'sport',
  # 'daddr',
  # 'dport',
  # 'state',
# LABELS:
  # TARGET,
  # 'subcategory'

print ('\nCheck for high cardinality.')
print ('Column | # of different values | values')
for column in myObjects:
  print (column, '|', df [column].nunique (), '|', df [column].unique ())

### K: NOTE: saddr and daddr (source address and destination address) may incur
### into overfitting for a particular scenario of computer network. Since the
### classifier will use these IPs and MACs to aid in classifying the traffic.
### We may want to drop these attributes to guarantee IDS generalization.
df.drop (axis = 'columns', columns = 'saddr', inplace = True)
df.drop (axis = 'columns', columns = 'daddr', inplace = True)

print ('\nHandling categorical attributes (label encoding).')
from sklearn.preprocessing import LabelEncoder
myLabelEncoder = LabelEncoder ()
df ['flgs'] = myLabelEncoder.fit_transform (df ['flgs'])
df ['proto'] = myLabelEncoder.fit_transform (df ['proto'])
df ['sport'] = myLabelEncoder.fit_transform (df ['sport'].astype (str))
df ['dport'] = myLabelEncoder.fit_transform (df ['dport'].astype (str))
df ['state'] = myLabelEncoder.fit_transform (df ['state'])
print ('Objects:', list (df.select_dtypes (['object']).columns), '\n')

###############################################################################
### Drop unused targets
### K: NOTE: attack and attack_subcategory are labels for different
### applications, not attributes. They must not be used to aid classification.
print ('\nDropping attack and attack_subcategory.')
print ('These are labels for other scenarios.')
df.drop (axis = 'columns', columns = 'attack', inplace = True)
df.drop (axis = 'columns', columns = 'subcategory', inplace = True)


###############################################################################
## Encode Label
###############################################################################
print ('\nEnconding label.')
myLabels = df [TARGET].unique ()
print ('Label types before conversion:', myLabels)
for label, code in zip (myLabels, range (len (myLabels))):
  df [TARGET].replace (label, code, inplace = True)
print ('Label types after conversion:', df [TARGET].unique ())


###############################################################################
## Split dataset into train, validation and test sets
###############################################################################
from sklearn.model_selection import train_test_split
TEST_SIZE = 4/10
VALIDATION_SIZE = 1/10
print ('\nSplitting dataset (test/train):', TEST_SIZE)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                               df.iloc [:, :-1],
                                               df.iloc [:, -1],
                                               test_size = TEST_SIZE,
                                               random_state = STATE)
print ('\nSplitting dataset (validation/train):', VALIDATION_SIZE)
X_train_df, X_val_df, y_train_df, y_val_df = train_test_split (
                                             X_train_df,
                                             y_train_df,
                                             test_size = VALIDATION_SIZE,
                                             random_state = STATE)

print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_val_df shape:', X_val_df.shape)
print ('y_val_df shape:', y_val_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)


###############################################################################
## Imput missing data
###############################################################################
### K: NOTE: Only use derived information from the train set to avoid leakage.

from sklearn.impute import SimpleImputer
for myColumn, myStrategy in zip (columsWithMissingValues, imputingStrategies):
  myImputer = SimpleImputer (missing_values = np.nan, strategy = myStrategy)
  myImputer.fit (X_train_df [myColumn].values.reshape (-1, 1))
  X_train_df [myColumn] = myImputer.transform (X_train_df [myColumn].values.reshape (-1, 1))
  X_val_df [myColumn] = myImputer.transform (X_val_df [myColumn].values.reshape (-1, 1))
  X_test_df [myColumn] = myImputer.transform (X_test_df [myColumn].values.reshape (-1, 1))


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
### K: NOTE: Only use derived information from the train set to avoid leakage.
#print ('\nApplying normalization (standard)')
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler ()
#scaler.fit (X_train)
#X_train = scaler.transform (X_train)
#X_val = scaler.transform (X_val)
#X_test = scaler.transform (X_test)


###############################################################################
## Handle imbalanced data
###############################################################################
"""
### K: 10,000 samples per attack
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
print ('\nHandling imbalanced label distribution.')

### Only oversample
myOversampler = RandomOverSampler (sampling_strategy = 'not majority',
                                   random_state = STATE)
X_over, y_over = myOversampler.fit_resample (X_train, y_train)

### Only undersample
myUndersampler = RandomUnderSampler (sampling_strategy = 'not minority',
                                     random_state = STATE)
X_under, y_under = myUndersampler.fit_resample (X_train, y_train)

### Balanced
MAX_SAMPLES = int (1e4)
labels = dict (Counter (y_train))

sampleDictOver = {k : max (labels [k], MAX_SAMPLES) for k in labels}
balancedOverSampler = RandomOverSampler (sampling_strategy = sampleDictOver,
                                         random_state = STATE)

X_bal, y_bal = balancedOverSampler.fit_resample (X_train, y_train)
labels = dict (Counter (y_bal))

sampleDictUnder = {k : min (labels [k], MAX_SAMPLES) for k in labels}
balancedUnderSampler = RandomUnderSampler (sampling_strategy = sampleDictUnder,
                                           random_state = STATE)

X_bal, y_bal = balancedUnderSampler.fit_resample (X_bal, y_bal)

print ('Real:', Counter (y_train))
print ('Over:', Counter (y_over))
print ('Under:', Counter (y_under))
print ('Balanced:', Counter (y_bal))
"""


###############################################################################
## Create learning model (Support Vector Machine) and tune hyperparameters
###############################################################################
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
### -1 indices -> train
### 0  indices -> validation
test_fold = np.repeat ([-1, 0], [X_train.shape [0], X_val.shape [0]])
myPreSplit = PredefinedSplit (test_fold)

# 30 minutos na pepe nao fechou...
parameters = {'C' : [0.001, 0.01, 0.1, 1],
              'kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              'degree' : [1, 2, 3],
              'class_weight' : [None, 'balanced']}

clf = SVC ()
bestModel = GridSearchCV (estimator = clf,
                          param_grid = parameters,
                          scoring = 'f1_weighted',
                          cv = myPreSplit,
                          verbose = 2,
                          n_jobs = -1)

bestModel.fit (np.concatenate ((X_train, X_val), axis = 0),
               np.concatenate ((y_train, y_val), axis = 0))
print (bestModel.best_params_)


###############################################################################
## Analyze results
###############################################################################
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score
y_pred = bestModel.predict (X_val)
print ('\nPerformance on VALIDATION set:')
print ('Confusion matrix:')
print (confusion_matrix (y_val, y_pred,
                         labels = df [TARGET].unique ()))
print ('Accuracy:', accuracy_score (y_val, y_pred))
print ('Precision:', precision_score (y_val, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_val, y_pred, average = 'macro'))
print ('F1:', f1_score (y_val, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_val, y_pred,
                       labels = df [TARGET].unique ()))

sys.exit ()
### K: NOTE: Only look at test results when publishing...
print ('\nPerformance on TEST set:')
y_pred = bestModel.predict (X_test)
print ('Confusion matrix:')
print (confusion_matrix (y_test, y_pred,
                         labels = df [TARGET].unique ()))
print ('Accuracy:', accuracy_score (y_test, y_pred))
print ('Precision:', precision_score (y_test, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_test, y_pred, average = 'macro'))
print ('F1:', f1_score (y_test, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_test, y_pred,
                       labels = df [TARGET].unique ()))
