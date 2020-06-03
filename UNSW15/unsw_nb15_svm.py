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
# if (df.attack_cat.value_counts()[1] < df.attack_cat.value_counts()[0]):
#   remove_n =  df.attack_cat.value_counts()[0] - df.attack_cat.value_counts()[1]  # Number of rows to be removed   
#   print(remove_n)
#   df_to_be_dropped = df[df.attack_cat == 0]
#   drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)
#   df = df.drop(drop_indices)
# else: 
#   remove_n =  df.attack_cat.value_counts()[1] - df.attack_cat.value_counts()[0]  # Number of rows to be removed   
#   print(remove_n)
#   df_to_be_dropped = df[df.attack_cat == 1]
#   drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)
#   df = df.drop(drop_indices)



###############################################################################
## Slicing the data (usually the last column is the target)
###############################################################################
X = df.iloc [:, :15]
y = df.iloc [:, -1]


from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


####################################################################
# Categorical data treatment
####################################################################

cat_cols = X.columns[X.dtypes == 'O'] # Returns array with the columns that has Object types elements

categories = [
    X[column].unique() for column in X[cat_cols]]


for cat in categories:
    cat[cat == None] = 'missing'  # noqa


# Replacing missing values
categorical_imputer = SimpleImputer(missing_values=None, 
                                    strategy='constant', 
                                    fill_value='missing')

X[cat_cols] = categorical_imputer.fit_transform(X[cat_cols])

# Encoding the categorical data
categorical_encoder = OrdinalEncoder(categories = categories)

X[cat_cols] = categorical_encoder.fit_transform(X[cat_cols])



# ####################################################################
# # Numerical data treatment
# ####################################################################

# num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')] # Returns array with the columns that has float types elements

# # Scaling numerical values

# numerical_imputer = SimpleImputer(strategy = "mean")
# X[num_cols] = numerical_imputer.fit_transform(X[num_cols])

# numerical_scaler = StandardScaler()
# X[num_cols] = numerical_scaler.fit_transform(X[num_cols])


# ######################################################################

# Assigning the model to be used
svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)

# Transforming the data to numpy arrays
X = X.values
y = y.values





###############################################################################
## Split dataset into train and test sets
###############################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 1/5,
                                                     random_state = STATE)

# Foer time measure
import time
start_time = time.time()


# Training the model
model = svc.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

from sklearn.metrics import precision_score


# Getting the metrics for the model
y_pred = model.predict(X_test)
print('Predicted values:', y_pred[0:100])
print('Real values: ', y_test[0:100])

print('Precision Score: ', precision_score(y_test, y_pred))
print('Accuracy: ', model.score(X_test, y_test))

# # Ploting 
# from matplotlib import pyplot as plt

# # Plot data points and color using their class
# color = ["green" if c == 0 else "red" for c in y]
# # plt.scatter(X[:,0], X[:,1], c = color)
# # plt.plot(X[:,0], X[:,1], linestyle='None')
# # plt.axis("off"), plt.show();


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,2], X[:,4], c = color)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


# plt.show() 





