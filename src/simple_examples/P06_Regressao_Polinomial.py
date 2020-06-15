## Cortesia do professor Heraldo Almeida - https://github.com/HeraldoAlmeida

#==============================================================================
#  Regressao Linear Polinomial
#==============================================================================

#------------------------------------------------------------------------------
#  Importar o conjunto de dados em um dataframe do pandas
#------------------------------------------------------------------------------

import pandas as pd

dataframe = pd.read_csv('../../../datasets/D06_Salario_vs_Nivel.csv')

print('')
print('dataframe =')
print(dataframe)

#------------------------------------------------------------------------------
#  Criar os arrays numericos correspondentes aos atributos e ao alvo
#------------------------------------------------------------------------------

X = dataframe.iloc[:, 1].values
y = dataframe.iloc[:, 2].values

print('')
print('X.shape =',X.shape)
print('y.shape =',y.shape)

#------------------------------------------------------------------------------
#  Como neste caso so existe 1 atributo, X foi criado como vetor e
#  precisa ser redimensionado como matriz de 1 coluna para que o
#  metodo "fit" do regressor possa saber que se tratam de 10 amostras
#  com 1 unico atributo (e nao 1 amostra com 10 atributos)
#------------------------------------------------------------------------------

X = X.reshape(-1,1)  # redimensionar de vetor-linha para vetor-coluna

print('')
print('X.shape =',X.shape)
print('y.shape =',y.shape)

#------------------------------------------------------------------------------
#  Visualizar as amostras em um diagrama de dispersao "Salario vs Nivel"
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.scatter(X, y, color = 'red')
plt.title('Amostras disponiveis para treinamento')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Treinar um regressor linear com o conjunto de dados inteiro
#------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

regressor1 = LinearRegression()
regressor1.fit(X, y)

#------------------------------------------------------------------------------
#  Obter as respostas do regressor linear para o conjunto de treinamento
#------------------------------------------------------------------------------

y_pred_1 = regressor1.predict(X)

#------------------------------------------------------------------------------
#  Visualizar graficamente as respostas do regressor linear
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.scatter(X, y, color = 'red', alpha = 0.5)
plt.scatter(X, y_pred_1, color = 'blue', marker = 'x')
plt.title('Modelo de Regressao Linear')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Criar um grid de 0.00 a 10.00 com passo de 0.01
#------------------------------------------------------------------------------

import numpy as np

X_grid = np.arange(0.00,10.01,0.01).reshape(-1,1)

print('')
print('X_grid =')
print(X_grid)

#------------------------------------------------------------------------------
#  Obter as respostas do regressor linear para cada ponto do grid
#------------------------------------------------------------------------------

y_grid = regressor1.predict(X_grid)

print('')
print('y_grid =')
print(y_grid)

#------------------------------------------------------------------------------
#  Visualizar graficamente o modelo de regressao linear
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.plot(X_grid, y_grid, color = 'blue')
plt.scatter(X, y, color = 'red')
plt.title('Modelo de Regressao Linear')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Verificar desempenho do regressor LINEAR
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho do regressor linear:')
#print('MSE  = %.3f' %           mean_squared_error(y, y_grid) )
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_1)))
print('R2   = %.3f' %                     r2_score(y, y_pred_1) )

#------------------------------------------------------------------------------
#  Treinar um regressor polinomial com o conjunto de dados inteiro
#------------------------------------------------------------------------------

# gerar atributos (coeficientes) do polinomio de grau desejado

from sklearn.preprocessing import PolynomialFeatures

poly_feat = PolynomialFeatures(degree = 9)
X_poly = poly_feat.fit_transform(X)

# treinar o regressor polinomial, ou seja,
# um regressor linear treinado com os atributos polinomiais
# derivados dos atributos originais das amostras

regressor2 = LinearRegression(fit_intercept=False)
regressor2.fit(X_poly, y)

#------------------------------------------------------------------------------
#  Obter as respostas do regressor polinomial para o conjunto de treinamento
#------------------------------------------------------------------------------

y_pred_2 = regressor2.predict(X_poly)

#------------------------------------------------------------------------------
#  Visualizar graficamente as respostas do regressor polinomial
#------------------------------------------------------------------------------

plt.scatter(X, y, color = 'red', alpha = 0.5)
plt.scatter(X, y_pred_2, color = 'blue', marker = 'x')
plt.title('Modelo de Regressao Polinomial')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Obter as respostas do regressor polinomil para cada ponto do grid
#------------------------------------------------------------------------------

X_poly_grid = poly_feat.transform(X_grid)
y_grid = regressor2.predict(X_poly_grid)

#------------------------------------------------------------------------------
#  Visualizar graficamente o modelo de regressao polinomial
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.plot(X_grid, y_grid, color = 'blue')
plt.scatter(X, y, color = 'red')
plt.title('Modelo de Regressao Polinomial')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

#------------------------------------------------------------------------------
#  Comparar o desempenho dos regressores LINEAR e POLINOMIAL
#------------------------------------------------------------------------------

import math
from sklearn.metrics import mean_squared_error, r2_score

print('\nDesempenho do regressor linear:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_1)))
print('R2   = %.3f' %                     r2_score(y, y_pred_1) )

print('\nDesempenho do regressor polinomial:')
print('RMSE = %.3f' % math.sqrt(mean_squared_error(y, y_pred_2)))
print('R2   = %.3f' %                     r2_score(y, y_pred_2) )
