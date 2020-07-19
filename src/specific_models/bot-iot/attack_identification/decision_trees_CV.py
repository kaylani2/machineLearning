# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Decision trees

import sys
import time
import pandas as pd
import numpy as np
from unit import remove_columns_with_one_value, remove_nan_columns, load_dataset
from unit import display_general_information
from collections import Counter
#from imblearn.over_sampling import RandomOverSampler, RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
import keras.utils
from keras import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from keras.optimizers import RMSprop, Adam
from keras.constraints import maxnorm

###############################################################################
## Define constants
###############################################################################
# Random state for reproducibility
STATE = 0
np.random.seed (STATE)

pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 5)

BOT_IOT_DIRECTORY = '../../../../datasets/bot-iot/'
BOT_IOT_FEATURE_NAMES = 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv'
BOT_IOT_FILE_5_PERCENT_SCHEMA = 'UNSW_2018_IoT_Botnet_Full5pc_{}.csv' # 1 - 4
FIVE_PERCENT_FILES = 4
BOT_IOT_FILE_FULL_SCHEMA = 'UNSW_2018_IoT_Botnet_Dataset_{}.csv' # 1 - 74
FULL_FILES = 74
FILE_NAME = BOT_IOT_DIRECTORY + BOT_IOT_FILE_5_PERCENT_SCHEMA#FULL_SCHEMA
FEATURES = BOT_IOT_DIRECTORY + BOT_IOT_FEATURE_NAMES
NAN_VALUES = ['?', '.']
TARGET = 'attack'
INDEX_COLUMN = 'pkSeqID'

###############################################################################
## Load dataset
###############################################################################
#featureDf = pd.read_csv (FEATURES)
#featureColumns = featureDf.columns.to_list ()
#featureColumns = [f.strip () for f in featureColumns]

#df = pd.DataFrame ()
#for fileNumber in range (1, FIVE_PERCENT_FILES + 1):#FULL_FILES + 1):
#  print ('Reading', FILE_NAME.format (str (fileNumber)))
#  aux = pd.read_csv (FILE_NAME.format (str (fileNumber)),
#                     #names = featureColumns,
#                     index_col = 'pkSeqID',
#                     dtype = {'pkSeqID' : np.int32}, na_values = NAN_VALUES,
#                     low_memory = False)
#  df = pd.concat ([df, aux])
df = load_dataset (FILE_NAME, FIVE_PERCENT_FILES, INDEX_COLUMN, NAN_VALUES)


###############################################################################
## Display generic (dataset independent) information
###############################################################################
#print ('Dataframe shape (lines, columns):', df.shape, '\n')
#print ('First 5 entries:\n', df [:5], '\n')
#df.info (verbose = False) # Make it true to find individual atribute types
#print ('\nDataframe contains NaN values:', df.isnull ().values.any ())
#nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
#print ('Number of NaN columns:', len (nanColumns))
#print ('NaN columns:', nanColumns, '\n')
display_general_information (df)


###############################################################################
## Display specific (dataset dependent) information
###############################################################################
print ('\nAttack types:', df ['attack'].unique ())
print ('Attack distribution:')
print (df ['attack'].value_counts ())
print ('\nCateogry types:', df ['category'].unique ())
print ('Cateogry distribution:')
print (df ['subcategory'].value_counts ())
print ('\nSubcategory types:', df ['subcategory'].unique ())
print ('Subcategory distribution:')
print (df ['subcategory'].value_counts ())


###############################################################################
## Data pre-processing
###############################################################################
#df.replace (['NaN', 'NaT'], np.nan, inplace = True)
#df.replace ('?', np.nan, inplace = True)
#df.replace ('Infinity', np.nan, inplace = True)

###############################################################################
### Remove columns with only one value
#print ('\nColumn | # of different values')
## nUniques = df.nunique () ### K: Takes too long. WHY?
#nUniques = []
#for column in df.columns:
#  nUnique = df [column].nunique ()
#  nUniques.append (nUnique)
#  print (column, '|', nUnique)
#
#print ('\nRemoving attributes that have only one (or zero) sampled value.')
#for column, nUnique in zip (df.columns, nUniques):
#  if (nUnique <= 1): # Only one value: DROP.
#    df.drop (axis = 'columns', columns = column, inplace = True)
#
#print ('\nColumn | # of different values')
#for column in df.columns:
#  nUnique = df [column].nunique ()
#  print (column, '|', nUnique)
df, log = remove_columns_with_one_value (df, verbose = False)
print (log)

###############################################################################
### Remove redundant columns
### K: These columns are numerical representations of other existing columns.
redundantColumns = ['state_number', 'proto_number', 'flgs_number']
print ('\nRemoving redundant columns:', redundantColumns)
df.drop (axis = 'columns', columns = redundantColumns, inplace = True)

###############################################################################
### Remove NaN columns (with a lot of NaN values)
#print ('\nColumn | NaN values')
#print (df.isnull ().sum ())
#print ('Removing attributes with more than half NaN values.')
#df = df.dropna (axis = 'columns', thresh = df.shape [0] // 2)
#print ('Dataframe contains NaN values:', df.isnull ().values.any ())
#print ('\nColumn | NaN values (after dropping columns)')
#print (df.isnull ().sum ())
df, log = remove_nan_columns (df, 1/2, verbose = False)
print (log)

###############################################################################
### Input missing values
### K: Look into each attribute to define the best inputing strategy.
### K: NOTE: This must be done after splitting to dataset to avoid data leakge.
df ['sport'].replace ('-1', np.nan, inplace = True)
df ['dport'].replace ('-1', np.nan, inplace = True)
### K: Negative port values are invalid.
columnsWithMissingValues = ['sport', 'dport']
# nUniques = df.nunique () ### K: Takes too long. WHY?
#nUniques = []
#for column in df.columns:
#  nUnique = df [column].nunique ()
#  nUniques.append (nUnique)
#
#### K: Examine values.
#for column in df.columns:
#  nUnique = df [column].nunique ()
#for column, nUnique in zip (df.columns, nUniques):
#    if (nUnique < 5):
#      print (column, df [column].unique ())
#    else:
#      print (column, 'unique values:', nUnique)

# sport  unique values: 91168     # most_frequent?
# dport  unique values: 115949    # most_frequent?
imputingStrategies = ['most_frequent', 'most_frequent']


###############################################################################
### Handle categorical values
### K: Look into each attribute to define the best encoding strategy.
#df.info (verbose = False)
### K: dtypes: float64 (11), int64 (8), object (9)
#myObjects = list (df.select_dtypes ( ['object']).columns)
#print ('\nObjects:', myObjects, '\n')
### K: Objects:
  # 'flgs',
  # 'proto',
  # 'saddr',
  # 'sport',
  # 'daddr',
  # 'dport',
  # 'state',
# LABELS:
  # TARGET,
  # 'subcategory'

#print ('\nCheck for high cardinality.')
#print ('Column | # of different values | values')
#for column in myObjects:
#  print (column, '|', df [column].nunique (), '|', df [column].unique ())

### K: NOTE: saddr and daddr (source address and destination address) may incur
### into overfitting for a particular scenario of computer network. Since the
### classifier will use these IPs and MACs to aid in classifying the traffic.
### We may want to drop these attributes to guarantee IDS generalization.
df.drop (axis = 'columns', columns = 'saddr', inplace = True)
df.drop (axis = 'columns', columns = 'daddr', inplace = True)

#print ('\nHandling categorical attributes (label encoding).')
#myLabelEncoder = LabelEncoder ()
#df ['flgs'] = myLabelEncoder.fit_transform (df ['flgs'])
#df ['proto'] = myLabelEncoder.fit_transform (df ['proto'])
#df ['sport'] = myLabelEncoder.fit_transform (df ['sport'].astype (str))
#df ['dport'] = myLabelEncoder.fit_transform (df ['dport'].astype (str))
#df ['state'] = myLabelEncoder.fit_transform (df ['state'])
#print ('Objects:', list (df.select_dtypes (['object']).columns), '\n')

###############################################################################
### Drop unused targets
### K: NOTE: category and subcategory are labels for different
### applications, not attributes. They must not be used to aid detection.
print ('\nDropping category and subcategory.')
print ('These are labels for other scenarios.')
df.drop (axis = 'columns', columns = 'category', inplace = True)
df.drop (axis = 'columns', columns = 'subcategory', inplace = True)


###############################################################################
## Encode Label
###############################################################################
### K: Binary classification. Already encoded.



###############################################################################
## Split dataset into train, and test sets
###############################################################################
### K: Dataset is too big...
drop_indices = np.random.choice (df.index, int (df.shape [0] * 0.5),
                                 replace = False)
df = df.drop (drop_indices)
TEST_SIZE = 2/10
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                               df.iloc [:, :-1],
                                               df.iloc [:, -1],
                                               test_size = TEST_SIZE,
                                               random_state = STATE,)
                                               #shuffle = False)

print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)


###############################################################################
## Create processing pipeline for cross-validation
###############################################################################
print ('Creating pipeline.')
steps = list ()
my_imputer = SimpleImputer (missing_values = np.nan, strategy = 'most_frequent')
my_scaler = StandardScaler ()
my_model = DecisionTreeClassifier (criterion = 'gini', max_depth = 10,
                                   min_samples_split = 2, splitter = 'best')
steps.append (('imputer', my_imputer))
steps.append (('scaler', my_scaler))
steps.append (('model', my_model))
pipeline = Pipeline (steps = steps)

print ('Starting cross-validation.')
cv = RepeatedStratifiedKFold (n_splits = 10, n_repeats = 3,
                              random_state = STATE)
scores = cross_val_score (pipeline, X_train_df, y_train_df,
                          scoring = 'accuracy', cv = cv, n_jobs = -1)
print ('Accuracy: %.3f (%.3f)' % (mean (scores)*100, std (scores)*100))


sys.exit ()
###############################################################################
### K: Isolate columns by pre-processor and strategy
###############################################################################
### most_frequent
most_frequent_strategy = ['sport', 'dport']
my_imputer = SimpleImputer (missing_values = np.nan, strategy = 'most_frequent')
steps = list ()
steps.append (('imputer', my_imputer))
most_frequent_transformer = Pipeline (steps)
###############################################################################
### standard_scaler
standard_scaler_strategy = ['everything_else']
my_scaler = StandardScaler ()
steps = list ()
steps.append (('scaler', my_scaler))
standard_scaler_transformer = Pipeline (steps)
###############################################################################
### label_encoder
label_encoder_strategy = ['everything']
my_label_encoder = LabelEncoder ()
steps = list ()
steps.append (('scaler', my_scaler))
label_encoder_transformer = Pipeline (steps)

preprocessor = ColumnTransformer (transformers = [
              ('mf', most_frequent_strategy, most_frequent_transformer),
              ('ss', standard_scaler_strategy, standard_scaler_transformer),
              ('le', label_encoder_strategy, label_encoder_transformer)])
###############################################################################
### Create model and assemble pipeline
my_model = DecisionTreeClassifier (criterion = 'gini', max_depth = 10,
                                   min_samples_split = 2, splitter = 'best')
steps = list ()
steps.append (('preprocessor', preprocessor))
steps.append (('classifier', my_model))
clf = Pipeline (steps = steps)
clf.fit (X_train, y_train)
print ('Model score: %.3f' % clf.score (X_test, y_test))







###############################################################################
## Create learning model (Decision Tree) and tune hyperparameters
###############################################################################
### -1 indices -> train
### 0  indices -> validation
test_fold = np.repeat ([-1, 0], [X_train.shape [0], X_val.shape [0]])
myPreSplit = PredefinedSplit (test_fold)

#parameters = {'criterion' : ['gini', 'entropy'],
#              'splitter' : ['best', 'random'],
#              'max_depth' : [1, 10, 100, None],
#              'min_samples_split' : [2, 3, 4]}
#clf = DecisionTreeClassifier ()
#bestModel = GridSearchCV (estimator = clf,
#                          param_grid = parameters,
#                          scoring = 'f1_weighted',
#                          cv = myPreSplit,
#                          verbose = 1,
#                          n_jobs = -1)
#
#startTime = time.time ()
#bestModel.fit (np.concatenate ((X_train, X_val), axis = 0),
#               np.concatenate ((y_train, y_val), axis = 0))
#print (bestModel.best_params_)
#print (str (time.time () - startTime), 's to search grid.')


#{'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2, 'splitter': 'best'}
startTime = time.time ()
bestModel = DecisionTreeClassifier (criterion = 'gini', max_depth = 10,
                                    min_samples_split = 2, splitter = 'best')
bestModel.fit (X_train, y_train)
print (str (time.time () - startTime), 's to train model.')


###############################################################################
## Analyze results
###############################################################################
y_pred = bestModel.predict (X_val)
print ('\nPerformance on VALIDATION set:')
print ('Confusion matrix:')
print (confusion_matrix (y_val, y_pred,
                         labels = df [TARGET].unique ()))
print ('Accuracy:', accuracy_score (y_val, y_pred))
print ('Precision:', precision_score (y_val, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_val, y_pred, average = 'macro'))
print ('F1:', f1_score (y_val, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_val, y_pred,
                       labels = df [TARGET].unique ()))

#sys.exit ()
### K: NOTE: Only look at test results when publishing...
print ('\nPerformance on TEST set:')
y_pred = bestModel.predict (X_test)
print ('Confusion matrix:')
print (confusion_matrix (y_test, y_pred,
                         labels = df [TARGET].unique ()))
print ('Accuracy:', accuracy_score (y_test, y_pred))
print ('Precision:', precision_score (y_test, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_test, y_pred, average = 'macro'))
print ('F1:', f1_score (y_test, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_test, y_pred,
                       labels = df [TARGET].unique ()))
