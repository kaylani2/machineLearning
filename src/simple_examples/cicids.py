import pandas as pd

dataset = '../../../datasets/cicids/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

dataframe = pd.read_csv (dataset, sep = '\s*,\s*', engine = 'python')
print ('Shape of dataframe:', dataframe.shape)
print (dataframe)
print (dataframe.head ())
print (dataframe ['Label'].value_counts ())
#print ('Attributes:')
#print (dataframe.columns.tolist ())
#print (dataframe.info ())

################################# Coeficientes de Pearson ####################################
pearsonCorrelation = dataframe.corr (method = 'pearson')
pd.set_option ('display.max_rows', 7)
pd.set_option ('display.max_columns', 7)
print (pearsonCorrelation)
#print (pearsonCorrelation.to_string ())
pd.reset_option ('display.max_rows')
pd.reset_option ('display.max_columns')

## Make sure not to mix int and float
#from scipy.stats import pearsonr
#columnNames = dataframe.columns
#for column in columnNames:
#  print ('Pearon (%7s)' % column, ') = %6.3f , %6.3e' % pearsonr (dataframe [column], dataframe ['Flow Duration']))
#
#print ('Pearson coefficient =', pearsonr (dataframe ['Destination Port'], dataframe ['Flow Duration']))

#import seaborn as sb



#columnNames = dataframe.columns
#from scipy.stats import pearsonr
#for column in columnNames:
#  print ('pearson (%7s)' % column , ') = %6.3f , %6.3e' % pearsonr (dataframe [column], dataframe ['Label']))



################################# Dividir os dados  ####################################
from sklearn.preprocessing import LabelEncoder

X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1:].values
## TODO: Encode y into integers.
#y = dataframe ['Label']
print ('X:', type (X))
print (X)
print ('y:', type (y))
print (y)
from sklearn import preprocessing
le = preprocessing.LabelEncoder ()
y= le.fit_transform (dataframe ['Label'].values)
print ('y:', type (y))
print (y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 0)


################################# Definir a arquiture da rede  ####################################
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint


model = keras.Sequential ()
model.add (keras.layers.Dense (64, activation = 'relu', input_shape= (x_train.shape[1], )))
model.add (keras.layers.Dense (128, activation = 'relu'))
model.add (keras.layers.Dense (512, activation = 'tanh'))
model.add (keras.layers.Dense (64, activation = 'tanh'))
model.add (keras.layers.Dense (2, activation = 'softmax'))

##model.compile (loss = 'sparse_categorical_crossentropy',
## Old /\: sparse_categorical_crossentropy is for classification with multiple
## categories
model.compile (loss = 'sparse_categorical_crossentropy',
              optimizer = 'Adam',
              metrics = ['mae','accuracy'])

model.build ()
model.summary ()

################################# Treinar a rede  ####################################
treino = model.fit (x_train, y_train, epochs = 50, verbose = 1)
