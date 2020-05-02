import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn


### Exibir o dataset
## FORGE
# generate dataset
X, y = mglearn.datasets.make_forge ()
# plot dataset
mglearn.discrete_scatter (X[:, 0], X[:, 1], y)
plt.legend (["Class 0", "Class 1"], loc=4)
plt.xlabel ("First feature")
plt.ylabel ("Second feature")
print ("X.shape:", X.shape)
plt.show ()

## CANCER
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer ()
print ('Dataset Cancer:')
print ("cancer.keys ():\n", cancer.keys ())
print ("Shape of cancer data:", cancer.data.shape)
print ("Sample counts per class:\n",
      {n: v for n, v in zip (cancer.target_names, np.bincount (cancer.target))})
print ("Feature names:\n", cancer.feature_names)

## BOSTON
print ('Dataset Boston:')
from sklearn.datasets import load_boston
boston = load_boston ()
print ("Data shape:", boston.data.shape)
X, y = mglearn.datasets.load_extended_boston ()
print ("X.shape:", X.shape)



## Mostrar o k-nn com 1 e 3 vizinhos
## FORGE
mglearn.plots.plot_knn_classification (n_neighbors=1)
plt.show ()
mglearn.plots.plot_knn_classification (n_neighbors=3)
plt.show ()


X, y = mglearn.datasets.make_forge ()

## Dividir os dados
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, random_state=0)

## Instanciar o classificador
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier (n_neighbors=3)

## Treinar o modelo
clf.fit (X_train, y_train)

## Exibir os resultados usando acuracia media como pontuacao
print ("Test set predictions:", clf.predict (X_test))
print ("Test set accuracy: {:.2f}".format (clf.score (X_test, y_test)))

fig, axes = plt.subplots (1, 3, figsize= (10, 3))
for n_neighbors, ax in zip ([1, 3, 10], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier (n_neighbors=n_neighbors).fit (X, y)
    mglearn.plots.plot_2d_separator (clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter (X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title ("{} neighbor (s)".format (n_neighbors))
    ax.set_xlabel ("feature 0")
    ax.set_ylabel ("feature 1")
    plt.plot ()
axes[0].legend (loc=3)

#plt.savefig ('vizinhos_n' + '.png')
plt.show ()
