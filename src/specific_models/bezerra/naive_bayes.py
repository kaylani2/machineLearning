import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

STATE = 0
STATES = [0, 10, 100, 1000, 10000]

pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 5)

DATASET_DIR = '../../../datasets/Dataset-IoT/'
NETFLOW_DIRS = ['MC/NetFlow/', 'SC/NetFlow/', 'ST/NetFlow/']


# MC_I_FIRST: Has infected data by Hajime, Aidra and BashLite botnets'
# MC_I_SECOND: Has infected data from Mirai botnets
# MC_I_THIR: Has infected data from Mirai, Doflo, Tsunami and Wroba botnets
# MC_L: Has legitimate data, no infection


path_types = ['MC', 'SC', 'ST']
data_set_files = [ [r'MC_I{}.csv'.format(index) for index in range(1, 4)],
                   [r'SC_I{}.csv'.format(index) for index in range(1, 4)],
                   [r'ST_I{}.csv'.format(index) for index in range(1, 4)] ]

for path, files in zip(path_types, data_set_files):
    files.append(path + '_L.csv')

################
##reading data##
################

for n, (path, files) in enumerate(zip(NETFLOW_DIRS, data_set_files), start=1):
    for csvFile in files:
        if n == 1:
            df = pd.read_csv(DATASET_DIR + path + csvFile)
        else:
            aux_df = pd.read_csv(DATASET_DIR + path + csvFile)
            df = pd.concat([df, aux_df], ignore_index=True)

print ("Data Loaded")

#making the final DataFrame
#dropping the number of the rows column
df = df.drop(df.columns[0], axis=1)

#dropping bad columns
nUniques = df.nunique()
for column, nUnique in zip (df.columns, nUniques):
    if(nUnique == 1):
        df.drop(axis='columns', columns=column, inplace=True)

#dropping unrelated columns
df.drop(axis='columns', columns=['ts', 'te', 'sa', 'da'], inplace=True)


#sampling the df
df = df.sample (frac=1, replace=True, random_state=0)
#################################
## Encoding the data           ##
#################################

from sklearn import preprocessing

cat_cols, num_cols = df.columns[df.dtypes == 'O'], df.columns[df.dtypes != 'O']
num_cols = num_cols[1:]

categories = [df[column].unique() for column in df[cat_cols]]

categorical_encoder = preprocessing.OrdinalEncoder(categories=categories)
categorical_encoder.fit(df[cat_cols])
df[cat_cols] = categorical_encoder.transform(df[cat_cols])

############################################
## Split dataset into train and test sets ##
############################################

for state in STATES:
  np.random.seed (state)

  from sklearn.model_selection import train_test_split
  TEST_SIZE = 1/5
  print ('\nSplitting dataset (test/train):', TEST_SIZE)
  X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                                 df.iloc [:, 1:],
                                                 df.iloc [:, 0],
                                                 test_size = TEST_SIZE,
                                                 random_state = state)
  print ('X_train_df shape:', X_train_df.shape)
  print ('y_train_df shape:', y_train_df.shape)
  print ('X_test_df shape:', X_test_df.shape)
  print ('y_test_df shape:', y_test_df.shape)

  ########################################
  ## Convert dataframe to a numpy array ##
  ########################################
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
  ## Create learning model (Naive Bayes)
  ###############################################################################


  from sklearn.naive_bayes import GaussianNB
  model = GaussianNB ()
  startTime = time.time()
  model.fit (X_train, y_train)
  print ("{} s to train model".format(time.time() - startTime))
  ###############################################################################
  ## Analyze results
  ###############################################################################

  from sklearn.metrics import confusion_matrix, precision_score, recall_score
  from sklearn.metrics import f1_score, classification_report, accuracy_score
  from sklearn.metrics import cohen_kappa_score
  y_pred = model.predict (X_test)

  TARGET = 'Label'

  print ('Confusion matrix:')
  print (confusion_matrix (y_test, y_pred,
                          labels = df [TARGET].unique ()))

  print ('Classification report:')
  print (classification_report (y_test, y_pred,
                              labels = df [TARGET].unique (),
                              digits = 3))

  print ('Accuracy:', accuracy_score (y_test, y_pred))
  print ('Precision:', precision_score (y_test, y_pred, average = 'macro'))
  print ('Recall:', recall_score (y_test, y_pred, average = 'macro'))
  print ('F1:', f1_score (y_test, y_pred, average = 'macro'))
  print ('Cohen Kappa:', cohen_kappa_score (y_test, y_pred,
                          labels = df [TARGET].unique ()))
