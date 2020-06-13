# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import arff


# Random state for reproducibility
STATE = 0
## Hard to not go over 80 columns
IOT_DIRECTORY = '../../../datasets/cardiff/IoT-Arff-Datasets/'
IOT_ATTACK_TYPE_FILENAME = 'AttackTypeClassification.arff'
FILE_NAME = IOT_DIRECTORY + IOT_ATTACK_TYPE_FILENAME

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
df.replace (['NaN', 'NaT'], np.nan, inplace = True)
df.replace ('?', np.nan, inplace = True)
df.replace ('Infinity', np.nan, inplace = True) ## Maybe other text values
## Remove NaN values
print ('Remove NaN and inf values:')
print ('Column | NaN values')
print (df.isnull ().sum ())
### K: 150k samples seems to be a fine cutting point for this dataset
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
df = df.dropna (axis = 'rows', thresh = df.shape [1])
print ('Column | NaN values (after dropping rows)')
print (df.isnull ().sum ())
print ('Dataframe contains NaN values:', df.isnull ().values.any ())

### K: We probably want to remove attributes that have only one sampled value.
print ('Column | # of different values')
print (type (df.nunique ()))
nUniques = df.nunique ()
#print (df.unique ())
for column, nUnique in zip (df.columns, nUniques):
  #print (column, nUnique)
  if (nUnique <= 3):
    print (column, df [column].unique ())
  else:
    print ('x')

  if (nUnique == 1): # Only one value: DROP.
    df.drop (axis = 'columns', columns = column, inplace = True)

print ('\n\n', df.nunique ())

###############################################################################
## Encode Label
###############################################################################
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
# 'ip.version', {4, 6}
# 'ip.flags.rb', {0, 1}
# 'ip.flags.df', {0, 1}
# 'ip.flags.mf', {0, 1}
# 'tcp.flags.res', {0, 1}
# 'tcp.flags.ns', {0, 1}
# 'tcp.flags.cwr', {0, 1}
# 'tcp.flags.ecn', {0, 1}
# 'tcp.flags.urg', {0, 1}
# 'tcp.flags.ack', {0, 1}
# 'tcp.flags.push', {0, 1}
# 'tcp.flags.reset', {0, 1}
# 'tcp.flags.syn', {0, 1}
# 'tcp.flags.fin', {0, 1}
# 'dns.flags.response', {0, 1}
# 'dns.flags.opcode', {0, 1}
# 'dns.flags.truncated', {0, 1}
# 'dns.flags.recdesired', {0, 1}
# 'dns.flags.z', {0, 1}
# 'dns.flags.checkdisable', {0, 1}
# 'packet_type', {in, out}
# LABELS:
# 'class_device_type', {AmazonEcho,BelkinCam,Hive,SmartThings,Lifx,TPLinkCam,TPLinkPlug,AP,Firewall,unknown}
# 'class_is_malicious' {0, 1}
#]
### K: Look into each attribute to define the best encoding strategy.
### K: NOTE: packet_type and class_device_type are labels for different
### applications, not attributes. They must not be used to aid classification.
print (df.nunique ())


###############################################################################
## Convert dataframe to a numpy array
###############################################################################
print ('\nConverting dataframe to numpy array.')
X = df.iloc [:, :-3].values
y = df.iloc [:, -1].values

###############################################################################
## Handle categorical attributes
###############################################################################
### K: Using a single strategy for now...

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



###############################################################################
## Create learning model (MLP)
###############################################################################
print ('Creating learning model.')
from keras.models import Sequential
from keras.layers import Dense, Dropout
BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 0.001
numberOfClasses = len (df ['class_attack_type'].unique ())
model = Sequential ()
model.add (Dense (units = 512, activation = 'relu',
                  input_shape = (X_train.shape [1], )))
model.add (Dense (256, activation = 'relu'))
model.add (Dense (128, activation = 'relu'))
model.add (Dense (numberOfClasses, activation = 'softmax'))
print ('Model summary:')
model.summary ()



sys.exit ()







###############################################################################
## Apply scaling (this could also be done after converting to numpy arrays)
###############################################################################
print ('Description BEFORE scaling:')
print (df.describe ()) # Before scaling
from sklearn.preprocessing import MinMaxScaler
mmScaler = MinMaxScaler ()
## Standard feature range: (0, 1)
df [df.columns [:-1]] = mmScaler.fit_transform (df [df.columns [:-1]])
## You may also use set of columns instead of the entire dataframe:
#df [ [' Flow Duration']] = mmScaler.fit_transform (df [ [' Flow Duration']])
print ('Description AFTER scaling:')
print (df.describe ()) # After scaling

## Alternatively, this could be done using a standard scaler (zero mean)
#from sklearn.preprocessing import StandardScaler
#sScaler = StandardScaler ()
#df [df.columns] = sScaler.fit_transform (df [df.columns])
#print ('Description AFTER scaling:')
#print (df.describe ()) # After scaling

## There are other, more robust scalers, specially resistant to outliers.
## Docs: https://scikit-learn.org/

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
