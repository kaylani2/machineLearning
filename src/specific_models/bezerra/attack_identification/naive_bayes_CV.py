import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score
from  unit import load_dataset, remove_columns_with_one_value, remove_nan_columns

state = 0
try:
  state = int (sys.argv [1])
except:
  pass
print ("STATE = ", state)
STATES = [0, 10, 100, 1000, 10000]

pd.set_option ('display.max_rows', None)
pd.set_option ('display.max_columns', 5)

df = load_dataset ()
print ("Data Loaded")
remove_columns_with_one_value (df, verbose=False)
remove_nan_columns (df, 0.6, verbose=False)
#making the final DataFrame
#dropping the number of the rows column
df = df.drop(df.columns[0], axis=1)

#dropping unrelated columns
df.drop(axis='columns', columns=['ts', 'te', 'sa', 'da'], inplace=True)


#sampling the df
df = df.sample (frac=1, replace=True, random_state=0)
#################################
## Encoding the data           ##
#################################

cat_cols, num_cols = df.columns[df.dtypes == 'O'], df.columns[df.dtypes != 'O']
num_cols = num_cols[1:]

categories = [df[column].unique() for column in df[cat_cols]]

categorical_encoder = preprocessing.OrdinalEncoder(categories=categories)
categorical_encoder.fit(df[cat_cols])
df[cat_cols] = categorical_encoder.transform(df[cat_cols])

############################################
## Split dataset into train and test sets ##
############################################
# for state in STATES:
np.random.seed (state)

TEST_SIZE = 0.3
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                            df.iloc [:, 1:],
                                            df.iloc [:, 0],
                                            test_size = TEST_SIZE,
                                            random_state = state)
print ('X_train_df shape:', X_train_df.shape)
print ('y_train_df shape:', y_train_df.shape)
print ('X_test_df shape:', X_test_df.shape)
print ('y_test_df shape:', y_test_df.shape)
print (X_train_df.head ())

cols = (len(num_cols) + len(cat_cols)) * [None]
cols[0:len(num_cols)] = num_cols
cols[len(num_cols):] = cat_cols

standard_scaler_features = cols
my_scaler = StandardScaler ()
steps = []
steps.append (('scaler', my_scaler))
standard_scaler_transformer = Pipeline (steps)

preprocessor = ColumnTransformer (transformers = [
            ('sca', standard_scaler_transformer, standard_scaler_features)])

clf = GaussianNB()
clf = Pipeline (steps=[('preprocessor', preprocessor), ('classifier', clf)], verbose=True)

startTime = time.time()
clf = clf.fit (X_train_df, y_train_df)
print (str (time.time() - startTime), 's to train model')

TARGET = 'Label'
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
