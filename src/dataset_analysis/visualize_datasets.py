# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

## Load dataset, describe, hadle categorical attributes
## CICIDS used as an example

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# Random state for eproducibility
STATE = 0
## Hard to not go over 80 columns
CICIDS_DIRECTORY = '../../datasets/cicids/MachineLearningCVE/'
CICIDS_MONDAY_FILENAME = 'Monday-WorkingHours.pcap_ISCX.csv'
CICIDS_WEDNESDAY_FILENAME = 'Wednesday-workingHours.pcap_ISCX.csv'
CICIDS_MONDAY = CICIDS_DIRECTORY + CICIDS_MONDAY_FILENAME
CICIDS_WEDNESDAY = CICIDS_DIRECTORY + CICIDS_WEDNESDAY_FILENAME

###############################################################################
## Load dataset
###############################################################################
df = pd.read_csv (CICIDS_WEDNESDAY)

## Fraction dataframe for quicker testing (copying code is hard)
df = df.sample (frac = 0.1, replace = True, random_state = STATE)
print ('Using fractured dataframe.')

###############################################################################
## Display generic (dataset independent) information
###############################################################################
print ('Dataframe shape (lines, collumns):', df.shape, '\n')
print ('First 5 entries:\n', df [:5], '\n')
print ('Dataframe attributes:\n', df.keys (), '\n')
## Note the pesky spaces before ALMOST all attributes
## This is annoying and could be removed, but we'll try to operate on the
## dataset "as is"
df.info (verbose = False) # Make it true to find individual atribute types
print (df.describe ()) # Brief statistical description on NUMERICAL atributes
print ('Dataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('NaN columns:', nanColumns)

## NOTE: Pearson doesn't really make sense for a classification problem...
## This is just an example to illustrate code usage
## Reminder: pearson only considers numerical atributes (ignores categorical)
## You'll probably want to scale the data before applying PCA, since the
## algorithm would be skewed by the features with higher variance originated
## from the units used.
#correlationMatrix =  df.corr (method = 'pearson')
#print ('Pearson:', correlationMatrix)
## You may want to plot the correlation matrix, but it gets hard to read
## when you have too many attributes. It's probably better to get the values
## you want with a set threshold directly from the matrix.
#import matplotlib.pyplot as plt
#import seaborn as sns
#plt.figure (figsize = (12, 10))
#cor = df.corr ()
#sns.heatmap (cor, annot = True, cmap = plt.cm.Reds)
#plt.show ()

## Optional: plot dispersion graphs
## This may be hard to read for some distributions
#columnNames = df.columns
#for column in columnNames:
#  df.plot.scatter (x = column, y = ' Label')
#  plt.show ()

###############################################################################
## Display specific (dataset dependent) information, we're using CICIDS
###############################################################################
## Remember the pesky spaces?
print ('Label types:', df [' Label'].unique ())
print ('Label distribution:\n', df [' Label'].value_counts ())
## Note that we may want to group the attacks together when handling the
## target as a categorical attribute, since there are so few samples of some
## of them.

###############################################################################
## Perform some form of basic preprocessing
###############################################################################
## For basic feature selection the correlation matrix can help identify
## highly correlated features (which are bad/cursed). In order to find
## which features have the highest predictive power, it's necessary
## to convert the target to a numeric value. After that it's possible to use
## a simple filter approach or a wrapper method (backward elimination, forward
## selection...) to select features.
## You may also choose to convert the dataframe to a numpy array and continue.

## Remove NaN and inf values
df.replace ('Infinity', np.nan, inplace = True) ## Or other text values
df.replace (np.inf, np.nan, inplace = True) ## Remove infinity
df.replace (np.nan, 0, inplace = True)
## We can also use scikit-learn to use other strategies for substitution
print ('Dataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('NaN columns:', nanColumns)

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
#from sklearn.feature_selection import VarianceThreshold
#temporaryDf = df [df.columns [:-1]] ## No label
#pd.set_option ('display.max_rows', None)
#print (temporaryDf.var ()) # Compute each variance
#pd.set_option ('display.max_rows', 15)
#selector = VarianceThreshold (threshold = (.100))
### Note: As of May 27th, 2020, VarianceThreshold throws an undocumented
### ValueError exception when none of the features meet the threshold value...
#selector.fit (temporaryDf)
#temporaryDf  = temporaryDf.loc [:, selector.get_support ()]
#print (temporaryDf.describe ()) # After removing

###############################################################################
## Encode categorical attributes (this may be done before finding pearson)
###############################################################################
print ('Label types before conversion:', df [' Label'].unique ())
df [' Label'] = df [' Label'].replace ('BENIGN', 0)
df [' Label'] = df [' Label'].replace ('DoS slowloris', 1)
df [' Label'] = df [' Label'].replace ('DoS Slowhttptest', 2)
df [' Label'] = df [' Label'].replace ('DoS Hulk', 3)
df [' Label'] = df [' Label'].replace ('DoS GoldenEye', 4)
df [' Label'] = df [' Label'].replace ('Heartbleed', 5)
print ('Label types after conversion:', df [' Label'].unique ())
df.info (verbose = False)

###############################################################################
## Convert dataframe to a numpy array (usually the last column is the target)
###############################################################################
X = df.iloc [:, :-1].values
y = df.iloc [:, -1].values

###############################################################################
## Split dataset into train and test sets
###############################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 1/3,
                                                     random_state = STATE)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)

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
