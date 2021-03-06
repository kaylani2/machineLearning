# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Naive Bayes
import sys
import time
import pandas as pd
import os
import math
import numpy as np
from numpy import mean, std
from unit import remove_columns_with_one_value, remove_nan_columns, load_dataset
from unit import display_general_information, display_feature_distribution
from collections import Counter
#from imblearn.over_sampling import RandomOverSampler, RandomUnderSampler
import sklearn
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
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
pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 5)
BOT_IOT_DIRECTORY = '../../../../datasets/bot-iot/'
BOT_IOT_FEATURE_NAMES = 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv'
BOT_IOT_FILE_5_PERCENT_SCHEMA = 'UNSW_2018_IoT_Botnet_Full5pc_{}.csv' # 1 - 4
FIVE_PERCENT_FILES = 4
BOT_IOT_FILE_FULL_SCHEMA = 'UNSW_2018_IoT_Botnet_Dataset_{}.csv' # 1 - 74
FULL_FILES = 74
FILE_NAME = BOT_IOT_DIRECTORY + BOT_IOT_FILE_5_PERCENT_SCHEMA
FEATURES = BOT_IOT_DIRECTORY + BOT_IOT_FEATURE_NAMES
NAN_VALUES = ['?', '.']
TARGET = 'attack'
INDEX_COLUMN = 'pkSeqID'
LABELS = ['attack', 'category', 'subcategory']
STATE = 0
try:
  STATE = int (sys.argv [1])
except:
  pass
#for STATE in [1, 2, 3, 4, 5]:
np.random.seed (STATE)
print ('STATE:', STATE)


###############################################################################
## Load dataset
###############################################################################
df = load_dataset (FILE_NAME, FIVE_PERCENT_FILES, INDEX_COLUMN, NAN_VALUES)


###############################################################################
## Clean dataset
###############################################################################
###############################################################################
### Remove columns with only one value
df, log = remove_columns_with_one_value (df, verbose = False)
print (log)


###############################################################################
### Remove redundant columns, useless columns and unused targets
### K: _number columns are numerical representations of other existing columns.
### K: category and subcategory are other labels.
### K: saddr and daddr may specialize the model to a single network
redundant_columns = ['state_number', 'proto_number', 'flgs_number']
other_targets = ['category', 'subcategory']
misc_columns = ['saddr', 'daddr']
print ('Removing redundant columns:', redundant_columns)
print ('Removing useless targets:', other_targets)
print ('Removing misc columns:', misc_columns)
columns_to_remove = redundant_columns + other_targets + misc_columns
df.drop (axis = 'columns', columns = columns_to_remove, inplace = True)

###############################################################################
### Remove NaN columns (with a lot of NaN values)
df, log = remove_nan_columns (df, 1/2, verbose = False)
print (log)

###############################################################################
### Encode categorical features
print ('Encoding categorical features (ordinal encoding).')
my_encoder = OrdinalEncoder ()
df ['flgs'] = my_encoder.fit_transform (df ['flgs'].values.reshape (-1, 1))
df ['proto'] = my_encoder.fit_transform (df ['proto'].values.reshape (-1, 1))
df ['sport'] = my_encoder.fit_transform (df ['sport'].astype (str).values.reshape (-1, 1))
df ['dport'] = my_encoder.fit_transform (df ['dport'].astype (str).values.reshape (-1, 1))
df ['state'] = my_encoder.fit_transform (df ['state'].values.reshape (-1, 1))
print ('Objects:', list (df.select_dtypes ( ['object']).columns))


###############################################################################
## Quick sanity check
###############################################################################
display_general_information (df)


###############################################################################
## Split dataset into train and test sets
###############################################################################
### K: Dataset is too big? Drop.
# drop_indices = np.random.choice (df.index, int (df.shape [0] * 0.5),
#                                  replace = False)
# df = df.drop (drop_indices)
TEST_SIZE = 3/10
print ('Splitting dataset (test/train):', TEST_SIZE)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                               df.loc [:, df.columns != TARGET],
                                               df [TARGET],
                                               test_size = TEST_SIZE,
                                               random_state = STATE,)
print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)


###############################################################################
## Define processing pipeline for training (hyperparameter are optimized)
###############################################################################
###############################################################################
### standard_scaler ### K: Non object features
object_features = (list (df.select_dtypes ( ['object']).columns))
remaining_features = list (df.columns)
for feature in object_features:
  remaining_features.remove (feature)
remaining_features.remove (TARGET)

standard_scaler_features = remaining_features
my_scaler = StandardScaler ()
steps = list ()
steps.append (('scaler', my_scaler))
standard_scaler_transformer = Pipeline (steps)

###############################################################################
### Assemble column transformer
preprocessor = ColumnTransformer (transformers = [
              ('sca', standard_scaler_transformer, standard_scaler_features)])

###############################################################################
### feature selector
NUMBER_OF_FEATURES = 9
SCORE_FUNCTION = f_classif
my_feature_selector = SelectKBest (score_func = SCORE_FUNCTION, k = NUMBER_OF_FEATURES)
steps = list ()
steps.append (('feature_selector', my_feature_selector))
feature_selector_transformer = Pipeline (steps)

###############################################################################
### Assemble pipeline for training
clf = GaussianNB ()
clf = Pipeline (steps = [ ('preprocessor', preprocessor),
                        ('feature_selector', feature_selector_transformer),
                        ('classifier', clf)],
                verbose = True)

###############################################################################
### Train
startTime = time.time ()
clf = clf.fit (X_train_df, y_train_df)
print (str (time.time () - startTime), 's to train model.')


###############################################################################
## Evaluate performance
###############################################################################
print ('\nPerformance on TRAIN set:')
y_pred = clf.predict (X_train_df)
my_confusion_matrix = confusion_matrix (y_train_df, y_pred, labels = df [TARGET].unique ())
tn, fp, fn, tp = my_confusion_matrix.ravel ()
print ('Confusion matrix:')
print (my_confusion_matrix)
print ('Accuracy:', accuracy_score (y_train_df, y_pred))
print ('Precision:', precision_score (y_train_df, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_train_df, y_pred, average = 'macro'))
print ('F1:', f1_score (y_train_df, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_train_df, y_pred,
                       labels = df [TARGET].unique ()))
print ('TP:', tp)
print ('TN:', tn)
print ('FP:', fp)
print ('FN:', fn)

### K: Only before publishing... Don't peek.
sys.exit ()
print ('\nPerformance on TEST set:')
y_pred = clf.predict (X_test_df)
my_confusion_matrix = confusion_matrix (y_test_df, y_pred, labels = df [TARGET].unique ())
tn, fp, fn, tp = my_confusion_matrix.ravel ()
print ('Confusion matrix:')
print (my_confusion_matrix)
print ('Accuracy:', accuracy_score (y_test_df, y_pred))
print ('Precision:', precision_score (y_test_df, y_pred, average = 'macro'))
print ('Recall:', recall_score (y_test_df, y_pred, average = 'macro'))
print ('F1:', f1_score (y_test_df, y_pred, average = 'macro'))
print ('Cohen Kappa:', cohen_kappa_score (y_test_df, y_pred,
                       labels = df [TARGET].unique ()))
print ('TP:', tp)
print ('TN:', tn)
print ('FP:', fp)
print ('FN:', fn)
