# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

## CICIDS used as an example

import pandas as pd
import numpy as np
import sys

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
df = df.sample (frac = 0.1, replace = True, random_state = 0)
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

## Reminder: pearson only considers numerical atributes (ignores catgorical)
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
## Encode categorical attributes (this may be done before finding pearson)
###############################################################################
print ('Label types before conversion:', df [' Label'].unique ())
df [' Label'] = df [' Label'].replace ('BENIGN', 0)
df [' Label'] = df [' Label'].replace ('DoS slowloris', 1)
df [' Label'] = df [' Label'].replace ('DoS Slowhttptest', 1)
df [' Label'] = df [' Label'].replace ('DoS Hulk', 1)
df [' Label'] = df [' Label'].replace ('DoS GoldenEye', 1)
df [' Label'] = df [' Label'].replace ('Heartbleed', 1)
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
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 1/5,
                                                     random_state = STATE)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)

###############################################################################
## Create learning model (Random Forest), fit model, analyze results
###############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import  precision_score, recall_score, mean_squared_error
import math
## Important: This mockup is just for showing the effect of the regularization
## strenght in performance and should not be used for hyperparameter tuning.
## To do that (tuning) just create another subset for validation and use the
## test set ONLY for publication.

print ( ' Estimators  Max Depth  MSE-OUT   RMSE-OUT   Acc. IN    Acc. OUT')
print ( ' ----        -------    -------   --------   --------   --------')
for ne in range (10, 70, 10):
  for md in range (2, 10):
    classifier = RandomForestClassifier (n_estimators = ne, random_state = 0,
                                        max_depth = md)
    classifier.fit (X_train, y_train)
    y_train_pred = classifier.predict (X_train)
    y_test_pred  = classifier.predict (X_test)
    acc_in  = r2_score (y_train, y_train_pred)
    acc_out = r2_score (y_test, y_test_pred)
    y_pred_train = y_train_pred
    y_pred_test = y_test_pred
    mse_out = mean_squared_error (y_test, y_pred_test)
    rmse_out = math.sqrt (mse_out)

    print (str ('   %2d' % ne      ) + '  ' +
           str ('   %2f' % md      ) + '  ' +
           str ('%10.4f' % mse_out ) + '  ' +
           str ('%10.4f' % rmse_out) + '  ' +
           str ('%10.4f' % acc_in  ) + '  ' +
           str ('%10.4f' % acc_out )
          ))

sys.exit ()
