import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

## Plot a decision tree
mglearn.plots.plot_animal_tree ()


################################# Carregando os dados #####################################
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer ()

################################# Divindo os dados #####################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (
cancer.data, cancer.target, stratify=cancer.target, random_state=42)


################################# Treinando o classificador #####################################
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier (random_state=0)
tree.fit (X_train, y_train)
print ("Accuracy on training set: {:.3f}".format (tree.score (X_train, y_train)))
print ("Accuracy on test set: {:.3f}".format (tree.score (X_test, y_test)))
## /\ Overfitting

## Parametro max_depth introduzido (pre-pruning)
tree = DecisionTreeClassifier (max_depth=4, random_state=0)
tree.fit (X_train, y_train)
print ("Accuracy on training set: {:.3f}".format (tree.score (X_train, y_train)))
print ("Accuracy on test set: {:.3f}".format (tree.score (X_test, y_test)))

################################# Exibir o resultado do modelo #####################################
from sklearn.tree import export_graphviz
export_graphviz (tree, out_file="tree.dot", class_names=["malignant", "benign"],
feature_names=cancer.feature_names, impurity=False, filled=True)

## Visualizar a arvore
import graphviz
with open ("tree.dot") as f:
  dot_graph = f.read ()
s = graphviz.Source (dot_graph)
s.view ()

#print ("Feature importances:\n{}".format (tree.feature_importances_))


def plot_feature_importances_cancer (model):
  n_features = cancer.data.shape[1]
  plt.barh (range (n_features), model.feature_importances_, align='center')
  plt.yticks (np.arange (n_features), cancer.feature_names)
  plt.xlabel ("Feature importance")
  plt.ylabel ("Feature")
  plt.show ()
plot_feature_importances_cancer (tree)
