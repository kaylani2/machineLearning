# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

## Carregar dataset, descrever, tratar atributos categoricos e plotar graficos
## CICIDS usado como exemplo por enquanto

import pandas as pd
import sys

## Hard to not go over 80 columns
CICIDS_DIRECTORY = '../../datasets/cicids/MachineLearningCVE/'
CICIDS_MONDAY_FILENAME = 'Monday-WorkingHours.pcap_ISCX.csv'
CICIDS_MONDAY = CICIDS_DIRECTORY + CICIDS_MONDAY_FILENAME


###############################################################################
## Carregar o dataset
###############################################################################
df = pd.read_csv (CICIDS_MONDAY)
#trainFrame = pd.read_csv (TRAIN_FILE)#, header = None)
#testFrame = pd.read_csv (TEST_FILE)#, header = None)
#df = pd.concat ([trainFrame, testFrame], ignore_index = True)

###############################################################################
## Exibir informacoes genericas (independe do dataset) sobre o dataset
###############################################################################
print ('Tipo preliminar do dataframe:', type (df), '\n')
print ('Formato do dataframe (linhas, colunas):', df.shape, '\n')
print ('Primeiras 5 linhas do dataframe:\n', df [:5], '\n')
print ('Atributos do dataframe:\n', df.keys (), '\n')
df.info (verbose = False) # Make it true to find individual atribute types
print (df.describe ()) # Brief statistical description on numeric atributes

## Lembrando que pearson so considera atributos numericos e ignora categoricos
correlationMatrix =  df.corr (method = 'pearson')
print ('Pearson:', correlationMatrix)
sys.exit ()


#print ('Tipos de ataque:', df ['class'].unique ())
#print ('Quantidade de ataques:', len (df ['class'].unique ()))
#print ('Graus de severidade de ataques:', df ['severity'].unique ())
#input ('Dataset analisado.')
## Ate aqui trabalhavamos com um dataframe pandas

###############################################################################
## Converter dataframe para arrays numpy (usualmente a ultima coluna eh o alvo)
###############################################################################
X = df.iloc [:, :-1].values
y = df.iloc [:, -1].values

###############################################################################
## Tratar atributos categóricos
###############################################################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder ()
X[:, 1] = labelencoder_X.fit_transform (X[:, 1])
X[:, 2] = labelencoder_X.fit_transform (X[:, 2])
X[:, 3] = labelencoder_X.fit_transform (X[:, 3])
labelencoder_y = LabelEncoder ()
y = labelencoder_y.fit_transform (y)
print ('Informacao:')
df.info (verbose = True)

sys.exit ()
