# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Multilayer Perceptron
### K: From the article:
## "Finally, for classifying the type of attack, the final sample size was set
## to acquire a sample of 50,000 packets (10,000 packets per attack) from a
## total of 220,785 packets."

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import arff


###############################################################################
## Define constants
###############################################################################
# Random state for reproducibility
STATE = 0
np.random.seed (STATE)
## Hard to not go over 80 columns
IOT_DIRECTORY = '../../../../datasets/cardiff/IoT-Arff-Datasets/'
IOT_ATTACK_TYPE_FILENAME = 'AttackTypeClassification.arff'
FILE_NAME = IOT_DIRECTORY + IOT_ATTACK_TYPE_FILENAME

###############################################################################
## Load dataset
###############################################################################
pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 5)
data = arff.loadarff (FILE_NAME)
df = pd.DataFrame (data [0])
print ('Dataframe shape (lines, collumns):', df.shape, '\n')
print ('First 5 entries:\n', df [:5], '\n')

### Decode byte strings into ordinary strings:
print ('Decoding byte strings into ordinary strings.')
strings = df.select_dtypes ( [np.object])
strings = strings.stack ().str.decode ('utf-8').unstack ()
for column in strings:
  df [column] = strings [column]
print ('Done.\n')

###############################################################################
## Display generic (dataset independent) information
###############################################################################
print ('Dataframe shape (lines, collumns):', df.shape, '\n')
print ('First 5 entries:\n', df [:5], '\n')
df.info (verbose = False) # Make it true to find individual atribute types
#print (df.describe ()) # Brief statistical description on NUMERICAL atributes

print ('Dataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('Number of NaN columns:', len (nanColumns))
#print ('NaN columns:', nanColumns, '\n')


###############################################################################
## Display specific (dataset dependent) information
###############################################################################
print ('Label types:', df ['class_attack_type'].unique ())
print ('Label distribution:\n', df ['class_attack_type'].value_counts ())


###############################################################################
## Data pre-processing
###############################################################################
df.replace (['NaN', 'NaT'], np.nan, inplace = True)
df.replace ('?', np.nan, inplace = True)
df.replace ('Infinity', np.nan, inplace = True)

###############################################################################
### Remove columns with only one value
print ('\n\nColumn | # of different values')
print (df.nunique ())
print ('Removing attributes that have only one sampled value.')
nUniques = df.nunique ()
for column, nUnique in zip (df.columns, nUniques):
  if (nUnique == 1): # Only one value: DROP.
    df.drop (axis = 'columns', columns = column, inplace = True)

print ('\n\nColumn | # of different values')
print (df.nunique ())

###############################################################################
### Remove NaN columns (with a lot of NaN values)
print ('\n\nColumn | NaN values')
print (df.isnull ().sum ())
### K: 150k samples seems to be a fine cutting point for this dataset
print ('Removing attributes with more than half NaN and inf values.')
df = df.dropna (axis = 'columns', thresh = 150000)
print ('Dataframe contains NaN values:', df.isnull ().values.any ())
print ('\n\nColumn | NaN values (after dropping columns)')
print (df.isnull ().sum ())

### K: NOTE: Not a good idea to drop these samples since it reduces
### K: the number of available MITM samples by a lot.
### K: So this is not a good strategy...
#print ('Label distribution before dropping rows:')
#print (df ['class_attack_type'].value_counts ())
### K: This leaves us with the following attributes to encode:
### Attribute            NaN values
#   ip.hdr_len           7597
#   ip.dsfield.dscp      7597
#   ip.dsfield.ecn       7597
#   ip.len               7597
#   ip.flags             7597
#   ip.flags.df          7597
#   ip.flags.mf          7597
#   ip.frag_offset       7597
#   ip.ttl               7597
#   ip.proto             7597
#   ip.checksum.status   7597
### K: Options: Remove these samples or handle them later.
### K: Removing them for now.
#print ('Removing samples with NaN values (not a lot of these).')
#df = df.dropna (axis = 'rows', thresh = df.shape [1])
#print ('Label distribution after dropping rows:')
#print (df ['class_attack_type'].value_counts ())
#print ('Column | NaN values (after dropping rows)')
#print (df.isnull ().sum ())
#print ('Dataframe contains NaN values:', df.isnull ().values.any ())

###############################################################################
### Input missing values
### K: Look into each attribute to define the best inputing strategy.
### K: NOTE: This must be done after splitting to dataset to avoid data leakge.
columsWithMissingValues = ['ip.hdr_len', 'ip.dsfield.dscp', 'ip.len',
                           'ip.flags', 'ip.flags.df', 'ip.flags.mf',
                           'ip.frag_offset', 'ip.ttl', 'ip.proto']
### K: Examine values.
nUniques = df.nunique ()
for column, nUnique in zip (df.columns, nUniques):
  if (column in columsWithMissingValues):
    print (column, df [column].unique ())
    print (column, ' unique values:', nUnique)

# ip.hdr_len [20. 24. nan]                          most frequent?
# ip.dsfield.dscp [ 0.  4. 48. 32.  6. 46.  5. nan] most frequent?
# ip.len  unique values: 512                        median?
# ip.flags [40.  0. 21. 20.  1. nan]                most frequent?
# ip.flags.df ['1' '0' nan]                         most frequent?
# ip.flags.mf ['0' '1' nan]                         most frequent?
# ip.frag_offset  unique values: 190                median?
# ip.ttl  unique values: 61                         mean? (round afterwards)
# ip.proto [ 1.  6. 17.  2. nan]                    most frequent?
### K: Now we use these strategies after splitting the dataset.
imputingStrategies = ['most_frequent', 'most_frequent', 'median',
                      'most_frequent', 'most_frequent', 'most_frequent',
                      'median', 'mean', 'most_frequent']


###############################################################################
### Handle categorical values
### K: Look into each attribute to define the best encoding strategy.
df.info (verbose = False)
### K: dtypes: float64 (27), int64 (1), object (5)
#print (df.columns.to_series ().groupby (df.dtypes).groups, '\n\n')
print ('Objects:', list (df.select_dtypes ( ['object']).columns), '\n')
### K: Objects: [
# 'ip.flags.df', {0, 1}
# 'ip.flags.mf', {0, 1}
# 'packet_type', {in, out}
# LABELS:
# 'class_device_type',
# 'class_is_malicious' {0, 1}
#]

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
print ('Encoding label.')
print ('Label types before conversion:', df ['class_attack_type'].unique ())
df ['class_attack_type'] = df ['class_attack_type'].replace ('N/A', 0)
df ['class_attack_type'] = df ['class_attack_type'].replace ('DoS', 1)
df ['class_attack_type'] = df ['class_attack_type'].replace ('iot-toolkit', 2)
df ['class_attack_type'] = df ['class_attack_type'].replace ('MITM', 3)
df ['class_attack_type'] = df ['class_attack_type'].replace ('Scanning', 4)
print ('Label types after conversion:', df ['class_attack_type'].unique ())


###############################################################################
## Split dataset into train and test sets
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
  X_val_df [myColumn] = myImputer.transform (X_val_df [myColumn].values.reshape (-1, 1))
  X_test_df [myColumn] = myImputer.transform (X_test_df [myColumn].values.reshape (-1, 1))

# Round ip.ttl
X_train_df ['ip.ttl'] = X_train_df ['ip.ttl'].round (decimals = 0)
X_val_df ['ip.ttl'] = X_val_df ['ip.ttl'].round (decimals = 0)
X_test_df ['ip.ttl'] = X_test_df ['ip.ttl'].round (decimals = 0)

#print ('\n\nColumn | NaN values (before imputing)')
#print ('\nTrain:')
#print (X_train_df.isnull ().sum ())
#print ('\nVal:')
#print (X_val_df.isnull ().sum ())
#print ('\nTest:')
#print (X_test_df.isnull ().sum ())

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
print ('Applying normalization (standard)')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
scaler.fit (X_train)
X_train = scaler.transform (X_train)
X_test = scaler.transform (X_test)
X_val = scaler.transform (X_val)

#### K: One hot encode the output.
#import keras.utils
#from keras.utils import to_categorical
#numberOfClasses = len (df ['class_attack_type'].unique ())
#y_train = keras.utils.to_categorical (y_train, numberOfClasses)
#y_test = keras.utils.to_categorical (y_test, numberOfClasses)


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
## Create learning model (MLP)
###############################################################################
### K: One hot encode the output.
import keras.utils
from keras.utils import to_categorical
numberOfClasses = len (df ['class_attack_type'].unique ())
y_train = keras.utils.to_categorical (y_train, numberOfClasses)
y_val = keras.utils.to_categorical (y_val, numberOfClasses)
y_test = keras.utils.to_categorical (y_test, numberOfClasses)


print ('Creating learning model.')
from keras.models import Sequential
from keras.layers import Dense, Dropout
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 4
LEARNING_RATE = 0.001
model = Sequential ()
model.add (Dense (units = 15, activation = 'relu',
                  input_shape = (X_train.shape [1], )))
model.add (Dense (20, activation = 'relu'))
model.add (Dense (numberOfClasses, activation = 'softmax'))
print ('Model summary:')
model.summary ()


###############################################################################
## Compile the network
###############################################################################
print ('Compiling the network.')
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import metrics
model.compile (loss = 'categorical_crossentropy',
               optimizer = Adam (lr = LEARNING_RATE),
               metrics = ['accuracy',
                          metrics.CategoricalAccuracy (),
               ])


###############################################################################
## Fit the network
###############################################################################
print ('Fitting the network.')
history = model.fit (X_train, y_train,
                     batch_size = BATCH_SIZE,
                     epochs = NUMBER_OF_EPOCHS,
                     verbose = 2, # 1 = progress bar, not useful for logging
                     validation_data = (X_val, y_val)
                    )


###############################################################################
## Analyze results
###############################################################################
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score
### K: NOTE: Only look at test results when publishing...
# model.predict outputs one hot encoding
y_pred = model.predict (X_test)
#print ('y_pred shape:', y_pred.shape)
#print ('y_test shape:', y_test.shape)
#print (y_pred [:50])
y_pred = y_pred.round ()
#print (y_pred [:50])
#print (y_test [:50])
#print (confusion_matrix (y_test, y_pred))
#print (classification_report (y_test, y_pred, digits = 3))
#scoreArray = model.evaluate (X_test, y_test, verbose = True)
#print ('Test loss:', scoreArray [0])
#print ('Test accuracy:', scoreArray [1])
print ('Confusion matrix:')
print (confusion_matrix (y_test.argmax (axis = 1), y_pred.argmax (axis = 1),
                         labels = df ['class_attack_type'].unique ()))
print ('\n\n')
print ('Accuracy:', accuracy_score (y_test, y_pred))
print ('Precision:', precision_score (y_test, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_test, y_pred, average = 'macro'))
print ('F1:', f1_score (y_test, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_test.argmax (axis = 1),
                                          y_pred.argmax (axis = 1),
                       labels = df ['class_attack_type'].unique ()))



import matplotlib.pyplot as plt

plt.plot (history.history ['categorical_accuracy'])
plt.plot (history.history ['val_categorical_accuracy'])
plt.title ('Model accuracy')
plt.ylabel ('Categorical Accuracy')
plt.xlabel ('Epoch')
plt.legend (['Train', 'Validation'], loc = 'upper left')
plt.show ()

plt.plot (history.history ['accuracy'])
plt.plot (history.history ['val_accuracy'])
plt.title ('Model accuracy')
plt.ylabel ('Accuracy')
plt.xlabel ('Epoch')
plt.legend (['Train', 'Validation'], loc = 'upper left')
plt.show ()

plt.plot (history.history ['loss'])
plt.plot (history.history ['val_loss'])
plt.title ('Model loss')
plt.ylabel ('Loss')
plt.xlabel ('Epoch')
plt.legend (['Train', 'Validation'], loc = 'upper left')
plt.show ()

sys.exit ()
