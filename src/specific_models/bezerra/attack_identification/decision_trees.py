# Author: Ernesto Rodríguez
# github.com/ernestorodg
# ernesto AT gta DOT ufrj DOT br

###############################################################################
## Analyse Bezerra's dataset for intrusion detection using Decision trees
###############################################################################

# We import everything used on all the codes, so it is easier to scale or reproduce.

import sys
import time
import pandas as pd
import os
import numpy as np
from numpy import mean, std
from unit import remove_columns_with_one_value, remove_nan_columns, load_dataset
from unit import display_general_information, display_feature_distribution
from collections import Counter
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
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import keras.utils
from keras import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from keras.optimizers import RMSprop, Adam
from keras.constraints import maxnorm
from numpy import empty
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor


###############################################################################
## Define constants 
###############################################################################


# Random state for reproducibility
try: 
  # If defined at argv:
  STATE = int(sys.argv[1])
except:
  # If not defined, it will be 0
  STATE = 0
np.random.seed(10)
# List of available attacks on the dataset
print ('STATE:', STATE)

TARGET = 'Label'


####################################################################
# Load the dataset
####################################################################


df = load_dataset ()
print ("Data Loaded")
remove_columns_with_one_value (df, verbose = False)
remove_nan_columns (df, 0.6, verbose = False)
#making the final DataFrame
#dropping the number of the rows column
df = df.drop (df.columns [0], axis = 1)

#dropping unrelated columns
df.drop (axis = 'columns', columns= ['ts', 'te', 'sa', 'da'], inplace = True)

print ('\nHandling categorical attributes (label encoding).')
my_label_encoder = OrdinalEncoder()
df ['flg'] = my_label_encoder.fit_transform (df ['flg'].values.reshape (-1, 1))
df ['pr'] = my_label_encoder.fit_transform (df ['pr'].values.reshape (-1, 1))



print('Columns with object types remaining:') 
print ('Objects:', list (df.select_dtypes ( ['object']).columns))




###############################################################################
## Split dataset into train, and test sets
###############################################################################
### Drop some indices if dataset is too big:
# drop_indices = np.random.choice (df.index, int (df.shape [0] * 0.5),
#                                  replace = False)
# df = df.drop (drop_indices)
TEST_SIZE = 3/10
print ('Splitting dataset (test/train):', TEST_SIZE)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                               df.iloc [:, 1:],
                                               df.iloc [:, 0],
                                               test_size = TEST_SIZE,
                                               random_state = STATE,)
                                               #shuffle = False)

print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)


###############################################################################
### Assemble pipeline for grid search
###############################################################################

'''
### Define pipeline to scale all the attributes
###############################################################################

object_features = (list (df.select_dtypes (['object']).columns))
remaining_features = list (df.columns)
for feature in object_features:
    remaining_features.remove (feature)

# Remove the target
remaining_features.remove ('Label')

standard_scaler_features = remaining_features 
my_scaler = StandardScaler ()
steps = list ()
steps.append (('scaler', my_scaler))
standard_scaler_transformer = Pipeline (steps)

preprocessor = ColumnTransformer (transformers = [
               ('sca', standard_scaler_transformer, standard_scaler_features)])

### Define pipeline to fit the model
###############################################################################


clf = DecisionTreeClassifier ()
clf = Pipeline (steps = [ ('preprocessor', preprocessor),
                          ('classifier', clf)],
               verbose = True)



# classifier__criterion': 'entropy', 'classifier__max_depth': 10, 'classifier__min_samples_split': 2, 'classifier__splitter': 'best

sorted(clf.get_params().keys())
param_grid = {'classifier__criterion' : ['gini', 'entropy'],
              'classifier__splitter' : ['best', 'random'],
              'classifier__max_depth' : [2, 5, 7, 10],#, 100, None],
              'classifier__min_samples_split' : [2, 3, 4, 5]}#, 3, 4]}
cv = RepeatedStratifiedKFold (n_splits = 5, n_repeats = 1, random_state = STATE)
grid = GridSearchCV (estimator = clf, param_grid = param_grid, scoring = 'f1', verbose = 1, n_jobs = -1, cv = cv)
grid_result = grid.fit (X_train_df, y_train_df)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip (means, stds, params):
    print ("%f (%f) with: %r" % (mean, stdev, param))

print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

sys.exit()

'''

###############################################################################
### Define pipeline to scale all the attributes
###############################################################################

object_features = (list (df.select_dtypes (['object']).columns))
remaining_features = list (df.columns)
for feature in object_features:
    remaining_features.remove (feature)

# Remove the target
remaining_features.remove ('Label')

standard_scaler_features = remaining_features 
my_scaler = StandardScaler ()
steps = list ()
steps.append (('scaler', my_scaler))
standard_scaler_transformer = Pipeline (steps)

preprocessor = ColumnTransformer (transformers = [
               ('sca', standard_scaler_transformer, standard_scaler_features)])

###############################################################################
### Define pipeline for tuned model
###############################################################################

clf = DecisionTreeClassifier (criterion = 'entropy', max_depth = 10,
                              min_samples_split = 2, splitter = 'best')
clf = Pipeline (steps = [ ('preprocessor', preprocessor),
                          ('classifier', clf)],
               verbose = True)




###############################################################################
### Fit the model
###############################################################################

startTime = time.time ()
clf = clf.fit (X_train_df, y_train_df)
training_time = time.time () - startTime
print (str (training_time), 's to train model.')


# Predicting from the test slice
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




