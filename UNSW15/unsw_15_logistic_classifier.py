# Author: Ernesto Rodríguez
# github.com/ernestorodg


import pandas as pd
import numpy as np
import sys

# Random state for reproducibility
STATE = 0
np.random.seed(10)
# List of available attacks
ATTACKS = ['Exploits',
          'Generic',
          ' Fuzzers',
          'DoS',
          'Analysis',
          'Worms',
          'Reconnaissance',
          'Backdoors',
          'Shellcode']


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

nanColumns = [i for i in df.columns if df [i].isnull ().any ()]





## Remove NaN and inf values
df.replace ('Infinity', np.nan, inplace = True) ## Or other text values
df.replace (np.inf, np.nan, inplace = True) ## Remove infinity
df.replace (np.nan, 0, inplace = True)


## Remove error values, especific from the dataset
df.replace ('0xc0a8', 0, inplace = True)
df.replace ('0x20205321', 0, inplace = True)


# ###############################################################################
# ## Encode categorical attributes (this may be done before finding pearson)
# ###############################################################################

for attack in ATTACKS:
  df['attack_cat'] = df ['attack_cat'].replace(attack, 1) 

# In this case we drop the last column. 'attack_cat' will be our target
df.drop(['Label'], axis=1)


# Proposition: Having the same amount of attacks and not-attacks rows
if (df.attack_cat.value_counts()[1] < df.attack_cat.value_counts()[0]):
  remove_n =  df.attack_cat.value_counts()[0] - df.attack_cat.value_counts()[1]  # Number of rows to be removed   
  print(remove_n)
  df_to_be_dropped = df[df.attack_cat == 0]
  drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)
  df = df.drop(drop_indices)
else: 
  remove_n =  df.attack_cat.value_counts()[1] - df.attack_cat.value_counts()[0]  # Number of rows to be removed   
  print(remove_n)
  df_to_be_dropped = df[df.attack_cat == 1]
  drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)
  df = df.drop(drop_indices)


###############################################################################
## Slicing the data (usually the last column is the target)
###############################################################################
X = df.iloc [:, :-1]
y = df.iloc [:, -1]


# Reminder: pearson only considers numerical atributes (ignores catgorical)
correlationMatrix =  df.corr (method = 'pearson')
# You may want to plot the correlation matrix, but it gets hard to read
# when you have too many attributes. It's probably better to get the values
# you want with a set threshold directly from the matrix.
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure (figsize = (12, 10))
cor = df.corr ()
sns.heatmap (cor, annot = True, cmap = plt.cm.Reds)
plt.show ()



from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler



cat_cols = X.columns[X.dtypes == 'O'] # Returns array with the columns that has Object types elements
num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')] # Returns array with the columns that has float types elements

# Following, the Categories on each column are saved on an array. On each index is saved
# the Categories on each column.

categories = [
    X[column].unique() for column in X[cat_cols]]


for cat in categories:
    cat[cat == None] = 'missing'  # noqa

# Transforming categorical data into an array of integers
cat_proc_nlin = make_pipeline(
    SimpleImputer(missing_values=None, strategy='constant',
                  fill_value='missing'),
    OrdinalEncoder(categories=categories)
    )

num_proc_nlin = make_pipeline(SimpleImputer(strategy='mean'))

cat_proc_lin = make_pipeline(
    SimpleImputer(missing_values=None,
                  strategy='constant',
                  fill_value='missing'),
    OneHotEncoder(categories=categories)
)

num_proc_lin = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

# transformation to use for non-linear estimators
processor_nlin = make_column_transformer(
    (cat_proc_nlin, cat_cols),
    (num_proc_nlin, num_cols),
    remainder='passthrough')

# transformation to use for linear estimators
processor_lin = make_column_transformer(
    (cat_proc_lin, cat_cols),
    (num_proc_lin, num_cols),
    remainder='passthrough')


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# 10 processes:
svm_pipeline = make_pipeline(processor_nlin,
                               SVC())
knn_pipeline = make_pipeline(processor_nlin,
                               KNeighborsClassifier())

###############################################################################
## Split dataset into train and test sets
###############################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 1/5,
                                                     random_state = STATE)


###############################################################################
## Training the model
###############################################################################

svm_pipeline.fit(X_train, y_train)
print(svm_pipeline.score(X_test, y_test))

knn_pipeline.fit(X_train, y_train)
print(knn_pipeline.score(X_test, y_test))


names = ["Nearest Neighbors", 
         "Linear SVM", 
          "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf_pipeline = make_pipeline(processor_nlin,
                        clf)
    clf_pipeline.fit(X_train, y_train)
    score = clf_pipeline.score(X_test, y_test)
    print('Score of ', name, ':' , score, )



# ###############################################################################
# ## Fit the model
# ###############################################################################
# ## Important: This mockup is just for showing the effect of hyperparameter
# ## selection in performance and should not be used for hyperparameter tuning.
# ## To do that (tuning) just create another subset for validation and use the
# ## test set ONLY for publication.
# print ( '    C     Acc. IN    Acc. OUT')
# print ( ' ----     -------    --------')
# for k in range (-6, 10):
#   ## C: Inverse of regularization strength; must be a positive float. Like in
#   ## support vector machines, smaller values specify stronger regularization.
#   c = 10**k
#   lr = LogisticRegression (C = c, penalty = 'l2', solver = 'liblinear',
#                           multi_class = 'auto', max_iter = 50,
#                           random_state = STATE)

#   lr = lr.fit (X_train, y_train)
#   y_train_pred = lr.predict (X_train)
#   y_test_pred = lr.predict (X_test)
#   acc_in  = accuracy_score (y_train, y_train_pred)
#   acc_out = accuracy_score (y_test, y_test_pred)

#   print (str ( '   %2f' % c) + '  ' + str ( '%10.4f' % acc_in) + '  ' +
#          str ( '%10.4f' % acc_out))

# sys.exit ()