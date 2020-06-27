# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Naive Bayes

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

BOT_IOT_DIRECTORY = '../../../datasets/bot-iot/'
BOT_IOT_FEATURE_NAMES = 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv'
BOT_IOT_FILE_5_PERCENT_SCHEMA = 'UNSW_2018_IoT_Botnet_Full5pc_{}.csv' # 1 - 4
FIVE_PERCENT_FILES = 4
BOT_IOT_FILE_FULL_SCHEMA = 'UNSW_2018_IoT_Botnet_Dataset_{}.csv' # 1 - 74
FULL_FILES = 74
FILE_NAME = BOT_IOT_DIRECTORY + BOT_IOT_FILE_FULL_SCHEMA
FEATURES = BOT_IOT_DIRECTORY + BOT_IOT_FEATURE_NAMES
NAN_VALUES = ['?', '.']

###############################################################################
## Load dataset
###############################################################################
featureDf = pd.read_csv (FEATURES)
featureColumns = featureDf.columns.to_list ()
featureColumns = [f.strip () for f in featureColumns]

df = pd.read_csv (FILE_NAME.format ('1'), names = featureColumns,
                  index_col = 'pkSeqID', dtype = {'pkSeqID' : np.int32},
                  na_values = NAN_VALUES)

for fileNumber in range (2, 3):#FULL_FILES + 1):
  print ('Reading', FILE_NAME.format (str (fileNumber)))
  #pd.concat ([df, pd.read_csv (FILE_NAME.format (str (fileNumber)),
  #                            names = featureColumns, index_col = 'pkSeqID',
  #                            dtype = {'pkSeqID' : np.int32})])

  aux = pd.read_csv (FILE_NAME.format (str (fileNumber)),
                     names = featureColumns, index_col = 'pkSeqID',
                     dtype = {'pkSeqID' : np.int32}, na_values = NAN_VALUES)

  df = pd.concat ([df, aux])

#pd.DataFrame (df).reset_index (drop = True)


###############################################################################
## Display generic (dataset independent) information
###############################################################################
#print ('Dataframe shape (lines, columns):', df.shape, '\n')
#print ('First 5 entries:\n', df [:5], '\n')
#print ('entries:\n', df [4000000//4 - 5:4000000//4 + 5], '\n')
df.info (verbose = False) # Make it true to find individual atribute types

print ('Dataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('Number of NaN columns:', len (nanColumns))
print ('NaN columns:', nanColumns, '\n')


###############################################################################
## Display specific (dataset dependent) information
###############################################################################
print ('\nAttack types:', df ['attack'].unique ())
print ('Attack distribution:')
print (df ['attack'].value_counts ())
print ('\nCateogry types:', df ['category'].unique ())
print ('Cateogry distribution:')
print (df ['category'].value_counts ())
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
print ('\n\nColumn | # of different values')
# nUniques = df.nunique () ### K: Takes too long. WHY?
nUniques = []
for column in df.columns:
  nUnique = df [column].nunique ()
  nUniques.append (nUnique)
  print (column, '|', nUnique)

print ('Removing attributes that have only one (or zero) sampled value.')
for column, nUnique in zip (df.columns, nUniques):
  if (nUnique <= 1): # Only one value: DROP.
    df.drop (axis = 'columns', columns = column, inplace = True)

print ('\n\nColumn | # of different values')
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
print ('\n')


###############################################################################
### Handle categorical values
### K: Look into each attribute to define the best encoding strategy.
df.info (verbose = False)
### K: dtypes: float64(11), int64(8), object(9)
#print (df.columns.to_series ().groupby (df.dtypes).groups, '\n\n')
print ('Objects:', list (df.select_dtypes ( ['object']).columns), '\n')
### K: Objects:
  # 'flgs',
  # 'proto',
  # 'saddr',
  # 'sport',
  # 'daddr',
  # 'dport',
  # 'state',
# LABELS:
  # 'category',
  # 'subcategory'
sys.exit ()


### K: NOTE: ip.flags.df and ip.flags.mf only have numerical values, but have
### been loaded as objects because (probably) of missing values, so we can
### just convert them instead of treating them as categorical.
print ('\nHandling categorical attributes (label encoding).')
print ('ip.flags.df and ip.flags.mf have been incorrectly read as objects.')
print ('Converting them to numeric.')
df ['ip.flags.df'] = pd.to_numeric (df ['ip.flags.df'])
df ['ip.flags.mf'] = pd.to_numeric (df ['ip.flags.mf'])

### K: 'packet_type': {in, out} -> {0, 1}
from sklearn.preprocessing import LabelEncoder
myLabelEncoder = LabelEncoder ()
df ['packet_type'] = myLabelEncoder.fit_transform (df ['packet_type'])

print ('Objects:', list (df.select_dtypes ( ['object']).columns), '\n')

###############################################################################
### Drop unused targets
### K: NOTE: class_is_malicious and class_device_type are labels for different
### applications, not attributes. They must not be used to aid classification.
print ('Dropping class_device_type and class_is_malicious.')
print ('These are labels for other scenarios.')
df.drop (axis = 'columns', columns = 'class_device_type', inplace = True)
df.drop (axis = 'columns', columns = 'class_is_malicious', inplace = True)


###############################################################################
## Encode Label
###############################################################################
print ('Enconding label.')
print ('Label types before conversion:', df ['class_attack_type'].unique ())
#df ['class_attack_type'] = df ['class_attack_type'].replace ('N/A', 0)
#df ['class_attack_type'] = df ['class_attack_type'].replace ('DoS', 1)
#df ['class_attack_type'] = df ['class_attack_type'].replace ('iot-toolkit', 2)
#df ['class_attack_type'] = df ['class_attack_type'].replace ('MITM', 3)
#df ['class_attack_type'] = df ['class_attack_type'].replace ('Scanning', 4)
print ('Label types after conversion:', df ['class_attack_type'].unique ())


###############################################################################
## Split dataset into train and test sets
###############################################################################
from sklearn.model_selection import train_test_split
TEST_SIZE = 4/10
print ('\nSplitting dataset (test/train):', TEST_SIZE)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                               df.iloc [:, :-1],
                                               df.iloc [:, -1],
                                               test_size = TEST_SIZE,
                                               random_state = STATE)
print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)


###############################################################################
## Imput missing data
###############################################################################
### K: NOTE: Only use derived information from the train set to avoid leakage.
# ip.hdr_len [20. 24. nan] ## mean or most frequent?
# ip.dsfield.dscp [ 0.  4. 48. 32.  6. 46.  5. nan] most frequent?
# ip.len  unique values: 512 ## median?
# ip.flags [40.  0. 21. 20.  1. nan] most frequent?
# ip.frag_offset  unique values: 190 ## median?
# ip.ttl  unique values: 61 ## integer mean?
# ip.proto [ 1.  6. 17.  2. nan] ## most frequent?

#print ('\n\nColumn | NaN values (before imputing)')
#print ('\nTrain:')
#print (X_train_df.isnull ().sum ())
#print ('\nTest:')
#print (X_test_df.isnull ().sum ())

from sklearn.impute import SimpleImputer
for myColumn, myStrategy in zip (columsWithMissingValues, imputingStrategies):
  myImputer = SimpleImputer (missing_values = np.nan, strategy = myStrategy)
  myImputer.fit (X_train_df [myColumn].values.reshape (-1, 1))
  X_train_df [myColumn] = myImputer.transform (X_train_df [myColumn].values.reshape (-1, 1))
  X_test_df [myColumn] = myImputer.transform (X_test_df [myColumn].values.reshape (-1, 1))

# Round ip.ttl
X_train_df ['ip.ttl'] = X_train_df ['ip.ttl'].round (decimals = 0)
X_test_df ['ip.ttl'] = X_test_df ['ip.ttl'].round (decimals = 0)

#print ('\n\nColumn | NaN values (before imputing)')
#print ('\nTrain:')
#print (X_train_df.isnull ().sum ())
#print ('\nTest:')
#print (X_test_df.isnull ().sum ())

###############################################################################
## Convert dataframe to a numpy array
###############################################################################
print ('\nConverting dataframe to numpy array.')
X_train = X_train_df.values
X_test = X_test_df.values
y_train = y_train_df.values
y_test = y_test_df.values
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)


###############################################################################
## Apply normalization
###############################################################################
print ('Applying normalization (standard)')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
scaler.fit (X_train)
X_train = scaler.transform (X_train)
X_test = scaler.transform (X_test)


###############################################################################
## Handle imbalanced data
###############################################################################
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
sampleDictOver = {'N/A': max (labels ['N/A'], MAX_SAMPLES),
                  'Scanning': max (labels ['Scanning'], MAX_SAMPLES),
                  'DoS': max (labels ['DoS'], MAX_SAMPLES),
                  'MITM': max (labels ['MITM'], MAX_SAMPLES),
                  'iot-toolkit': max (labels ['iot-toolkit'], MAX_SAMPLES)
                 }
balancedOverSampler = RandomOverSampler (sampling_strategy = sampleDictOver,
                                         random_state = STATE)
X_bal, y_bal = balancedOverSampler.fit_resample (X_train, y_train)
labels = dict (Counter (y_bal))
sampleDictUnder = {'N/A': min (labels ['N/A'], MAX_SAMPLES),
                   'Scanning': min (labels ['Scanning'], MAX_SAMPLES),
                   'DoS': min (labels ['DoS'], MAX_SAMPLES),
                   'MITM': min (labels ['MITM'], MAX_SAMPLES),
                   'iot-toolkit': min (labels ['iot-toolkit'], MAX_SAMPLES)
                  }
balancedUnderSampler = RandomUnderSampler (sampling_strategy = sampleDictUnder,
                                           random_state = STATE)
X_bal, y_bal = balancedUnderSampler.fit_resample (X_bal, y_bal)

print ('Real:', Counter (y_train))
print ('Over:', Counter (y_over))
print ('Under:', Counter (y_under))
print ('Balanced:', Counter (y_bal))


###############################################################################
## Create learning model (Naive Bayes)
###############################################################################
for myX, myY, sampling in zip ([X_train, X_over, X_under, X_bal],
                               [y_train, y_over, y_under, y_bal],
                               ['Real', 'Over', 'Under', 'Balanced']):
  print ('Creating learning model.')
  print ('Sampling:', sampling)
  print ('X shape:', myX.shape)
  from sklearn.naive_bayes import GaussianNB
  model = GaussianNB ()
  model.fit (myX, myY)


  ###############################################################################
  ## Analyze results
  ###############################################################################
  ### K: NOTE: Only look at test results when publishing...
  from sklearn.metrics import confusion_matrix, precision_score, recall_score
  from sklearn.metrics import f1_score, classification_report, accuracy_score
  from sklearn.metrics import cohen_kappa_score
  y_pred = model.predict (X_test)

  print ('Confusion matrix:')
  print (confusion_matrix (y_test, y_pred,
                           labels = df ['class_attack_type'].unique ()))

  print ('Classification report:')
  print (classification_report (y_test, y_pred,
                                labels = df ['class_attack_type'].unique (),
                                digits = 3))

  print ('\n\n')
  print ('Accuracy:', accuracy_score (y_test, y_pred))
  print ('Precision:', precision_score (y_test, y_pred, average = 'macro'))
  print ('Recall:', recall_score (y_test, y_pred, average = 'macro'))
  print ('F1:', f1_score (y_test, y_pred, average = 'macro'))
  print ('Cohen Kappa:', cohen_kappa_score (y_test, y_pred,
                         labels = df ['class_attack_type'].unique ()))

sys.exit ()
