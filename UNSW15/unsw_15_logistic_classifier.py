# Author: Ernesto Rodríguez
# github.com/ernestorodg


import pandas as pd
import numpy as np
import sys

# Random state for reproducibility
STATE = 0

UNSW_NB15_DIRECTORY = r'../datasets/unsw-nb15/UNSW-NB15 - CSV Files/'
UNSW_NB15_FIRST = 'UNSW-NB15_1.csv'
UNSW_NB15_SECOND = 'UNSW-NB15_2.csv'
UNSW_NB15_THIRD = 'UNSW-NB15_3.csv'
UNSW_NB15_FOURTH = 'UNSW-NB15_4.csv'
UNSW_NB15_FEATURES = 'NUSW-NB15_features.csv'

###############################################################################
## Load dataset
###############################################################################
df = pd.read_csv (UNSW_NB15_DIRECTORY + UNSW_NB15_FIRST)

## Fraction dataframe for quicker testing (copying code is hard)
df = df.sample (frac = 0.1, replace = True, random_state = 0)
print ('Using fractured dataframe.')


###############################################################################
## On UNSW-NB15 Dataset, we have to put manually the Dataframe attributes
###############################################################################

## All the labels where collected manually from "NUSW-NB15_features". Trea
columns_label = np.array([ 'srcip','sport','dstip',
                          'dsport','proto','state',
                          'dur','sbytes','dbytes',
                          'sttl','dttl','sloss',
                          'dloss','service','Sload',
                          'Dload','Spkts','Dpkts',
                          'swin','dwin','stcpb',
                          'dtcpb','smeansz','dmeansz',
                          'trans_depth','res_bdy_len','Sjit',
                          'Djit','Stime','Ltime',
                          'Sintpkt','Dintpkt','tcprtt',
                          'synack','ackdat','is_sm_ips_ports',
                          'ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
                          'ct_ftp_cmd','ct_srv_src','ct_srv_dst',
                          'ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
                          'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat',
                          'Label'])

## Add the columns label to our dataset

df.columns = columns_label


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

# # Reminder: pearson only considers numerical atributes (ignores catgorical)
# correlationMatrix =  df.corr (method = 'pearson')
# print ('Pearson:', correlationMatrix)
# # You may want to plot the correlation matrix, but it gets hard to read
# # when you have too many attributes. It's probably better to get the values
# # you want with a set threshold directly from the matrix.
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure (figsize = (12, 10))
# cor = df.corr ()
# sns.heatmap (cor, annot = True, cmap = plt.cm.Reds)
# plt.show ()

###############################################################################
## Display specific (dataset dependent) information, we're using UNSW_15
###############################################################################

print ('Attack label types:', df ['attack_cat'].unique ())
print ('Attack label distribution:\n', df ['attack_cat'].value_counts ())
## Note that we may want to group the attacks together when handling the
## target as a categorical attribute, since there are so few samples of some
## of them.


# ###############################################################################
# ## Perform some form of basic preprocessing
# ###############################################################################
# ## For basic feature selection the correlation matrix can help identify
# ## highly correlated features (which are bad/cursed). In order to find
# ## which features have the highest predictive power, it's necessary
# ## to convert the target to a numeric value. After that it's possible to use
# ## a simple filter approach or a wrapper method (backward elimination, forward
# ## selection...) to select features.
# ## You may also choose to convert the dataframe to a numpy array and continue.

## Remove NaN and inf values
df.replace ('Infinity', np.nan, inplace = True) ## Or other text values
df.replace (np.inf, np.nan, inplace = True) ## Remove infinity
df.replace (np.nan, 0, inplace = True)


## Remove error values from the dataset
df.replace ('0xc0a8', 0, inplace = True)
df.replace ('0x20205321', 0, inplace = True)




# df.replace ('-', 0, inplace = True)
# df.replace ('dns', 0, inplace = True)
## We can also use scikit-learn to use other strategies for substitution
print ('Dataframe contains NaN values:', df.isnull ().values.any ())
nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
print ('NaN columns:', nanColumns)

###############################################################################
## Encode categorical attributes (this may be done before finding pearson)
###############################################################################
print ('Label types before conversion:', df ['attack_cat'].unique ())
df ['attack_cat'] = df ['attack_cat'].replace ('Exploits', 1)
df ['attack_cat'] = df ['attack_cat'].replace ('Generic', 1)
df ['attack_cat'] = df ['attack_cat'].replace (' Fuzzers', 1)
df ['attack_cat'] = df ['attack_cat'].replace ('DoS', 1)
df ['attack_cat'] = df ['attack_cat'].replace ('Analysis', 1)
df ['attack_cat'] = df ['attack_cat'].replace ('Worms', 1)
df ['attack_cat'] = df ['attack_cat'].replace ('Reconnaissance', 1)
df ['attack_cat'] = df ['attack_cat'].replace ('Backdoors', 1)
df ['attack_cat'] = df ['attack_cat'].replace ('Shellcode', 1)
print ('Label types after conversion:', df ['attack_cat'].unique ())
df.info (verbose = False)

# 0xc0a8 <= o que caralhos é isso?

# ###############################################################################
# ## Convert dataframe to a numpy array (usually the last column is the target)
# ###############################################################################
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
## Create learning model (LogisticR), fit model, analyze results
###############################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

###############################################################################
## LabelEncoder will substitute  a categorical data for a countable one
###############################################################################

label_encoderX = LabelEncoder()


for value in range(0, (X_train[0].shape)[0]):
  if type(X_train[0, value]) == type('This is a string'):
    X_train[:, value] = label_encoderX.fit_transform(X_train[:, value])


for value in range(0, (X_test[0].shape)[0]):
  if type(X_test[0, value]) == type('This is a string'):
    X_test[:, value] = label_encoderX.fit_transform(X_test[:, value])



# Solução do bug aqui embaixo:
# Pro bug aparecer é só comentar o primeiro for

for linha in range(0, 56000):
  for coluna in range(0, 47):
    if (X_train[linha, coluna] == '-'):
      X_train[linha, coluna] = 0

for linha in range(0, 56000):
  for coluna in range(0, 47):
    if (X_train[linha, coluna] == '-'):
      print(linha, coluna)

###############################################################################
## Fit the model
###############################################################################
## Important: This mockup is just for showing the effect of hyperparameter
## selection in performance and should not be used for hyperparameter tuning.
## To do that (tuning) just create another subset for validation and use the
## test set ONLY for publication.
print ( '    C     Acc. IN    Acc. OUT')
print ( ' ----     -------    --------')
for k in range (-6, 10):
  ## C: Inverse of regularization strength; must be a positive float. Like in
  ## support vector machines, smaller values specify stronger regularization.
  c = 10**k
  lr = LogisticRegression (C = c, penalty = 'l2', solver = 'liblinear',
                          multi_class = 'auto', max_iter = 50,
                          random_state = STATE)

  lr = lr.fit (X_train, y_train)
  y_train_pred = lr.predict (X_train)
  y_test_pred = lr.predict (X_test)
  acc_in  = accuracy_score (y_train, y_train_pred)
  acc_out = accuracy_score (y_test, y_test_pred)

  print (str ( '   %2f' % c) + '  ' + str ( '%10.4f' % acc_in) + '  ' +
         str ( '%10.4f' % acc_out))

sys.exit ()