# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

## Load dataset, describe, hadle categorical attributes
## Use MLP for classification
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
#plt.figure (figsize = (12,10))
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
## Create learning model (MLP)
###############################################################################
from keras.models import Sequential
from keras.layers import Dense, Dropout
BATCH_SIZE = 128
NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 0.001
numberOfClasses = len (df [' Label'].unique ())
model = Sequential ()
## @TODO: fix input_shape to get number of attributes
model.add (Dense (units = 512, activation = 'relu', input_shape= (78, )))
model.add (Dense (256, activation = 'relu'))
model.add (Dense (128, activation = 'relu'))
model.add (Dense (numberOfClasses, activation = 'softmax'))
print ('Model summary:')
model.summary ()

###############################################################################
## Compile the network
###############################################################################
from keras.optimizers import RMSprop
from keras.optimizers import Adam
model.compile (loss = 'sparse_categorical_crossentropy',
               optimizer = Adam (lr = LEARNING_RATE),
               metrics = ['accuracy'])

###############################################################################
## Fit the network
###############################################################################
history = model.fit (X_train, y_train,
                     batch_size = BATCH_SIZE,
                     epochs = NUMBER_OF_EPOCHS,
                     verbose = 1,
                     validation_split = 1/10)

###############################################################################
## Analyze results
###############################################################################
scoreArray = model.evaluate (X_test, y_test, verbose = 0)
print ('Test loss:', scoreArray [0])
print ('Test accuracy:', scoreArray [1])

import matplotlib.pyplot as plt
plt.plot (history.history ['accuracy'])
plt.plot (history.history ['val_accuracy'])
plt.title ('Model accuracy')
plt.ylabel ('Accuracy')
plt.xlabel ('Epoch')
plt.legend (['Train', 'Test'], loc = 'upper left')
plt.show ()

plt.plot (history.history ['loss'])
plt.plot (history.history ['val_loss'])
plt.title ('Model loss')
plt.ylabel ('Loss')
plt.xlabel ('Epoch')
plt.legend (['Train', 'Test'], loc = 'upper left')
plt.show ()

sys.exit ()
