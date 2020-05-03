# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

## Carregar dataset, descrever, tratar atributos categoricos e plotar graficos
## CICIDS usado como exemplo por enquanto

import pandas as pd
import sys

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
#trainFrame = pd.read_csv (TRAIN_FILE)#, header = None)
#testFrame = pd.read_csv (TEST_FILE)#, header = None)
#df = pd.concat ([trainFrame, testFrame], ignore_index = True)

###############################################################################
## Display generic (dataset independent) information
###############################################################################
print ('Dataframe shape (lines, collumns):', df.shape, '\n')
print ('First 5 entries:\n', df [:5], '\n')
print ('Dataframe attributes:\n', df.keys (), '\n')
## Note the pesky spaces before ALMOST all attributes
## This is annoying and could be removed, but will try to operate on the
## dataset "as is"
df.info (verbose = False) # Make it true to find individual atribute types
print (df.describe ()) # Brief statistical description on NUMERIC atributes
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

###############################################################################
## Converter dataframe para arrays numpy (usualmente a ultima coluna eh o alvo)
###############################################################################
X = df.iloc [:, :-1].values
y = df.iloc [:, -1].values
sys.exit ()

###############################################################################
## Tratar atributos categóricos
###############################################################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder ()
X[:, 1] = labelencoder_X.fit_transform (X[:, 1])
X[:, 2] = labelencoder_X.fit_transform (X[:, 2])
X[:, 3] = labelencoder_X.fit_transform (X[:, 3])
labelencoder_y = LabelEncoder ()
y = labelencoder_y.fit_transform (y)
print ('Informacao:')
df.info (verbose = True)

sys.exit ()
