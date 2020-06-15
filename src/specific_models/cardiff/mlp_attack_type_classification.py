# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

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
IOT_DIRECTORY = '../../../datasets/cardiff/IoT-Arff-Datasets/'
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

## Fraction dataframe for quicker testing (copying code is hard)
#df = df.sample (frac = 0.1, replace = True, random_state = STATE)
#print ('Using fractured dataframe.')

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
#print ('Dataframe attributes:\n', df.keys (), '\n')
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
## Perform some form of basic preprocessing
###############################################################################
df.replace ( ['NaN', 'NaT'], np.nan, inplace = True)
df.replace ('?', np.nan, inplace = True)
df.replace ('Infinity', np.nan, inplace = True) ## Maybe other text values
## Remove NaN values
print ('Column | NaN values')
print (df.isnull ().sum ())
### K: 150k samples seems to be a fine cutting point for this dataset
print ('Removing attributes with more than half NaN and inf values.')
df = df.dropna (axis = 'columns', thresh = 150000)
print ('Dataframe contains NaN values:', df.isnull ().values.any ())
print ('Column | NaN values (after dropping columns)')
print (df.isnull ().sum ())
### K: This leaves us with the following attributes to encode:
### Attribute            NaN values
#   ip.hdr_len           7597
#   ip.dsfield.dscp      7597
#   ip.dsfield.ecn       7597
#   ip.len               7597
#   ip.flags             7597 # {0, 1}
#   ip.frag_offset       7597
#   ip.ttl               7597
#   ip.proto             7597
#   ip.checksum.status   7597
### K: Options: Remove these samples or handle them later.
### K: Removing them for now.
print ('Removing samples with NaN values (not a lot of these).')
df = df.dropna (axis = 'rows', thresh = df.shape [1])
print ('Column | NaN values (after dropping rows)')
print (df.isnull ().sum ())
print ('Dataframe contains NaN values:', df.isnull ().values.any ())

### K: We probably want to remove attributes that have only one sampled value.
print ('Removing attributes that have only one sampled value.')
print ('Column | # of different values')
print (type (df.nunique ()))
nUniques = df.nunique ()
for column, nUnique in zip (df.columns, nUniques):
  if (nUnique <= 7):
    print (column, df [column].unique ())
  else:
    print (column, nUnique)

  if (nUnique == 1): # Only one value: DROP.
    df.drop (axis = 'columns', columns = column, inplace = True)

print ('\n\n', df.nunique ())


###############################################################################
## Encode Label
###############################################################################
print ('Enconding label.')
print ('Label types before conversion:', df ['class_attack_type'].unique ())
df ['class_attack_type'] = df ['class_attack_type'].replace ('N/A', 0)
df ['class_attack_type'] = df ['class_attack_type'].replace ('DoS', 1)
df ['class_attack_type'] = df ['class_attack_type'].replace ('iot-toolkit', 2)
df ['class_attack_type'] = df ['class_attack_type'].replace ('MITM', 3)
df ['class_attack_type'] = df ['class_attack_type'].replace ('Scanning', 4)
print ('Label types after conversion:', df ['class_attack_type'].unique ())


###############################################################################
## Last look before working with numpy arrays
###############################################################################
### K: We're looking for:
### - How many categorical attributes are there and their names
### - Brief statistical description before applying normalization
df.info (verbose = False)
### K: dtypes: float64 (27), int64 (1), object (23)
#print (df.columns.to_series ().groupby (df.dtypes).groups, '\n\n')
print (df.describe (), '\n\n') # Statistical description
print ('Objects:', list (df.select_dtypes ( ['object']).columns), '\n')
### K: Objects: [
# 'ip.flags.df', {0, 1}
# 'ip.flags.mf', {0, 1}
# 'packet_type', {in, out}
# LABELS:
# 'class_device_type', {AmazonEcho, BelkinCam, Hive, SmartThings,
#                       Lifx, TPLinkCam, TPLinkPlug, AP, Firewall, unknown}
# 'class_is_malicious' {0, 1}
#]
### K: Look into each attribute to define the best encoding strategy.
### K: NOTE: packet_type and class_device_type are labels for different
### applications, not attributes. They must not be used to aid classification.
print ('Dropping class_device_type and class_is_malicious.')
print ('These are labels for other scenarios.')
df.drop (axis = 'columns', columns = 'class_device_type', inplace = True)
df.drop (axis = 'columns', columns = 'class_is_malicious', inplace = True)
### K: NOTE: ip.flags.df and ip.flags.mf only have numerical values, but have
### been loaded as objects because (probably) of missing values, so we can
### just convert them instead of treating them as categorical.
print ('ip.flags.df and ip.flags.mf have been incorrectly read as objects.')
print ('Converting them to numeric.')
df ['ip.flags.df'] = pd.to_numeric (df ['ip.flags.df'])
df ['ip.flags.mf'] = pd.to_numeric (df ['ip.flags.mf'])
print (df.nunique ())
print ('Objects:', list (df.select_dtypes ( ['object']).columns), '\n')

###############################################################################
## Handle categorical attributes
###############################################################################
### K: Using a single strategy for now...
print ('\nHandling categorical attributes (label).')
from sklearn.preprocessing import LabelEncoder
myLabelEncoder = LabelEncoder ()
df ['packet_type'] = myLabelEncoder.fit_transform (df ['packet_type'])

### TODO: onehotencoder ta dando nan na saida, ajeitar isso ai
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder (handle_unknown = 'error')
#enc_df = pd.DataFrame (enc.fit_transform (df [ ['packet_type']]).toarray ())
#df = df.join (enc_df)
#df.drop (axis = 'columns', columns = 'packet_type', inplace = True)
#
#### K: NOTE: This transformed the dataframe in a way that the last column is
#### no longer the target. We have to fix that:
#cols_at_end = ['class_attack_type']
#df = df [ [c for c in df if c not in cols_at_end]
#        + [c for c in cols_at_end if c in df]]

### K: One last look.
nUniques = df.nunique ()
for column, nUnique in zip (df.columns, nUniques):
  if (nUnique <= 7):
    print (column, df [column].unique ())
  else:
    print (column, nUnique)



###############################################################################
## Convert dataframe to a numpy array
###############################################################################
print ('\nConverting dataframe to numpy array.')
X = df.iloc [:, :-1].values
y = df.iloc [:, -1].values


###############################################################################
## Split dataset into train and test sets
###############################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 4/10,
                                                     random_state = STATE)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)

###############################################################################
## Apply normalization
###############################################################################
print ('TODO: normalization')
print ('Applying normalization (standard)')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
scaler.fit (X_train)
print ('Mean before scalling:', scaler.mean_)
X_train = scaler.transform (X_train)
scaler.fit (X_train)
print ('Mean after scalling:', scaler.mean_)

scaler.fit (X_test)
X_test = scaler.transform (X_test)


###############################################################################
## Create learning model (MLP)
###############################################################################
print ('Creating learning model.')
from keras.models import Sequential
from keras.layers import Dense, Dropout
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 12
LEARNING_RATE = 0.001
numberOfClasses = len (df ['class_attack_type'].unique ())
model = Sequential ()
#model.add (Dense (units = 512, activation = 'relu',
model.add (Dense (units = 15, activation = 'relu',
                  input_shape = (X_train.shape [1], )))
#model.add (Dense (256, activation = 'relu'))
#model.add (Dense (128, activation = 'relu'))
model.add (Dense (20, activation = 'relu'))
model.add (Dense (numberOfClasses, activation = 'softmax'))
print ('Model summary:')
model.summary ()

###############################################################################
## Compile the network
###############################################################################
print ('Compiling the network.')
#for LEARNING_RATE in ( [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]):
#for LEARNING_RATE in ( [ 0.00001, 0.001 ]):
print ('lr:', LEARNING_RATE)
from keras.optimizers import RMSprop
from keras.optimizers import Adam
model.compile (loss = 'sparse_categorical_crossentropy',
               optimizer = Adam (lr = LEARNING_RATE),
               metrics = ['accuracy'])

###############################################################################
## Fit the network
###############################################################################
print ('Fitting the network.')
history = model.fit (X_train, y_train,
                     batch_size = BATCH_SIZE,
                     epochs = NUMBER_OF_EPOCHS,
                     verbose = 1,
                     validation_split = 1/10)

###############################################################################
## Analyze results
###############################################################################
### K: NOTE: Only look at test results when publishing...
#scoreArray = model.evaluate (X_test, y_test, verbose = 0)
#print ('Test loss:', scoreArray [0])
#print ('Test accuracy:', scoreArray [1])

import matplotlib.pyplot as plt
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

###############################################################################
## Feature selection (don't apply them all...)
## Keep in mind that our target here is a categorical feature...
## This is usually done after transforming the data to a numpy array
## NOTE: You should do this after splitting the data between train and test
## This is just an example to illustrate code usage
###############################################################################
### Remove low variance features
from sklearn.feature_selection import VarianceThreshold
temporaryDf = df [df.columns [:-1]] ## No label
pd.set_option ('display.max_rows', None)
print (temporaryDf.var ()) # Compute each variance
pd.set_option ('display.max_rows', 15)
selector = VarianceThreshold (threshold = (3))
## Note: As of May 27th, 2020, VarianceThreshold throws an undocumented
## ValueError exception when none of the features meet the threshold value...
selector.fit (temporaryDf)
temporaryDf  = temporaryDf.loc [:, selector.get_support ()]
print (temporaryDf.describe ()) # After removing

###############################################################################
## Feature selection after transforming to numpy arrays
## Keep in mind that our target here is a categorical feature...
###############################################################################
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
print ('Performing feature selection.')
print ('X_train shape before:', X_train.shape)
print ('X_test shape before:', X_test.shape)
classifier = ExtraTreesClassifier (n_estimators = 50, random_state = STATE)
classifier = classifier.fit (X_train, y_train)
print ('Feature, importance')
for feature in zip (df [:-1], classifier.feature_importances_):
  print (feature)
model = SelectFromModel (classifier, prefit = True)
X_train = model.transform (X_train)
X_test = model.transform (X_test)
print ('X_train shape after:', X_train.shape)
print ('X_test shape after:', X_test.shape)


sys.exit ()
