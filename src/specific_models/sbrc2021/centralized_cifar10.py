import sys
import os
import time
import flwr as fl
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_model (learn_rate = 0.01, dropout_rate = 0.0, weight_constraint = 0,
                  filter_size = 2):
    model = tf.keras.Sequential(
    [
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.MeanSquaredError()])
    return model


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
VALIDATION_SIZE = 1/4
print ('Splitting dataset (validation/train):', VALIDATION_SIZE)
X_train, X_val, y_train, y_val = train_test_split (X_train,
                                                   y_train,
                                                   test_size = VALIDATION_SIZE)
#y_train = to_categorical(y_train)
#y_val = to_categorical(y_val)
#y_test = to_categorical(y_test)
print ('X_train shape:', X_train.shape)
print ('y_train shape:', y_train.shape)
print ('X_val shape:', X_val.shape)
print ('y_val shape:', y_val.shape)
print ('X_test shape:', X_test.shape)
print ('y_test shape:', y_test.shape)


###############################################################################
## Hyperparameter tuning
test_fold = np.repeat ([-1, 0], [X_train.shape [0], X_val.shape [0]])
myPreSplit = PredefinedSplit (test_fold)


model = KerasClassifier (build_fn = create_model, verbose = 2)
batch_size = [64, 256]#, 2000]#10, 30, 50]
epochs = [5, 10]
learn_rate = [0.01]#, 0.01]#, 0.1, 0.2]
dropout_rate = [0.0]
weight_constraint = [0]#, 2, 3, 4, 5]
filter_size = [2]#, 3]
param_grid = dict (batch_size = batch_size, epochs = epochs,
                   dropout_rate = dropout_rate, learn_rate = learn_rate,
                   weight_constraint = weight_constraint,
                   filter_size = filter_size)
grid = GridSearchCV (estimator = model, param_grid = param_grid,
                     scoring = 'accuracy', cv = myPreSplit, verbose = 2,
                     n_jobs = 2)

grid_result = grid.fit (np.concatenate ((X_train, X_val), axis = 0),
                        np.concatenate ((y_train, y_val), axis = 0))
print (grid_result.best_params_)

print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip (means, stds, params):
    print ("%f (%f) with: %r" % (mean, stdev, param))
