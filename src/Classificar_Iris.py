import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import IPython




################################# Carregando os dados #####################################
from sklearn.datasets import load_iris
iris_dataset = load_iris ()

print ('\n\n')
print ('Dataset description:')
#print (iris_dataset['DESCR'][:193] + '\n...')
print (iris_dataset['DESCR'][:] + '\n')
print ('Target names:', iris_dataset['target_names'], '\n')
print ('Feature names:\n', iris_dataset['feature_names'], '\n')
print ('Type of data:', type (iris_dataset['data']), '\n')
print ('Shape of data:', iris_dataset['data'].shape, '\n')
print ('First five rows of data:\n', iris_dataset['data'][:5], '\n')
#print ('All rows of data:\n', iris_dataset['data'][:], '\n')
print ('Type of target:', type (iris_dataset['target']), '\n')
print ('Shape of target:', iris_dataset['target'].shape, '\n')
print ('Target:\n', iris_dataset['target'], '\n')
print ('Keys of iris_dataset:\n', iris_dataset.keys ())

################################# Divindo os dados #####################################
print ('\n\n')
print ('Divindo os dados:')
input ('Enter para continuar.')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (iris_dataset['data'],
                                                    iris_dataset['target'],
                                                    random_state=0)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)


################################# Treinando o classificador #####################################
print ('\n\n')
print ('Treinando o classificador.')
input ('Enter para continuar.')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier (n_neighbors=1)
knn.fit (X_train, y_train)

################################# Fazendo predicoes #####################################
print ('\n\n')
print ('Fazendo predicoes.')
input ('Enter para continuar.')

################################# Inserindo uma nova amostra #####################################
X_new = np.array ([[5, 2.9, 1, 0.2]])
print ('X_new.shape:', X_new.shape)
prediction = knn.predict (X_new)
print ('Prediction:', prediction)
print ('Predicted target name:', iris_dataset['target_names'][prediction])


################################# Medindo a acuracia no conjunto de testes #####################################
y_pred = knn.predict (X_test)
print ('Test set predictions:\n', y_pred)
print ('Test set score: {:.2f}'.format (np.mean (y_pred == y_test)))
print ('Test set score: {:.2f}'.format (knn.score (X_test, y_test)))
