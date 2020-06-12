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
IOT = IOT_DIRECTORY + IOT_ATTACK_TYPE_FILENAME

pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 6)
data = arff.loadarff (IOT)
df = pd.DataFrame (data [0])
print ('Dataframe shape (lines, collumns):', df.shape, '\n')
print ('First 5 entries:\n', df [:5], '\n')
print ('Dataframe attributes:\n', df.keys (), '\n')

### Decode byte strings into ordinary strings:
print ('Decode byte strings into ordinary strings:')
strings = df.select_dtypes ([np.object])
strings = strings.stack ().str.decode ('utf-8').unstack ()
for column in strings:
  df [column] = strings [column]

## Fraction dataframe for quicker testing (copying code is hard)
#df = df.sample (frac = 0.1, replace = True, random_state = STATE)
#print ('Using fractured dataframe.')

###############################################################################
## Display generic (dataset independent) information
###############################################################################
print ('Dataframe shape (lines, collumns):', df.shape, '\n')
print ('First 5 entries:\n', df [:5], '\n')
print ('Dataframe attributes:\n', df.keys (), '\n')
df.info (verbose = False) # Make it true to find individual atribute types
print (df.describe ()) # Brief statistical description on NUMERICAL atributes

print ('Dataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('Number of NaN columns:', len (nanColumns))
print ('NaN columns:', nanColumns)

###############################################################################
## Display specific (dataset dependent) information
###############################################################################
print ('Label types:', df ['class_attack_type'].unique ())
print ('Label distribution:\n', df ['class_attack_type'].value_counts ())

###############################################################################
## Perform some form of basic preprocessing
###############################################################################
## Remove NaN and inf values
print ('Remove NaN and inf values:')
print ('Column | NaN values')
print (df.isnull ().sum ())
### K: 70k samples seems to be a fine cutting point for this dataset
### K: nao ta removendo umas colunas com > 70k samples com NaN
df = df.dropna (axis = 'columns', thresh = 70000)


#df.replace ('Infinity', np.nan, inplace = True) ## Or other text values
#df.replace (np.inf, np.nan, inplace = True) ## Remove infinity
#df.replace (np.nan, 0, inplace = True)

print ('Dataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('NaN columns:', nanColumns)
print ('Description after removing NaN and inf')
print (df.describe ())
print ('Column | NaN values')
print (df.isnull ().sum ())

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
df.info (verbose = False)

###############################################################################
## Convert dataframe to a numpy array
###############################################################################
X = df.iloc [:, :-3].values
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
## Create learning model (MLP)
###############################################################################
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



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
