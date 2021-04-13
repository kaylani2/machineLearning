#import sys
#import os
#import time
#import flwr as fl
#import numpy as np
#import tensorflow as tf
#from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
#from keras.utils import to_categorical
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#
#def create_model (learn_rate = 0.01, dropout_rate = 0.0, weight_constraint = 0,
#                  filter_size = 2):
#    model = tf.keras.Sequential(
#    [
#    tf.keras.Input(shape=(32, 32, 3)),
#    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dropout(0.5),
#    tf.keras.layers.Dense(10, activation="softmax"),
#    ])
#    model.compile("adam", "sparse_categorical_crossentropy",
#                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
#                           tf.keras.metrics.MeanSquaredError()])
#    return model
#
#
#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
#VALIDATION_SIZE = 1/4
#print ('Splitting dataset (validation/train):', VALIDATION_SIZE)
#X_train, X_val, y_train, y_val = train_test_split (X_train,
#                                                   y_train,
#                                                   test_size = VALIDATION_SIZE)
#y_train = to_categorical(y_train)
#y_val = to_categorical(y_val)
#y_test = to_categorical(y_test)
#print ('X_train shape:', X_train.shape)
#print ('y_train shape:', y_train.shape)
#print ('X_val shape:', X_val.shape)
#print ('y_val shape:', y_val.shape)
#print ('X_test shape:', X_test.shape)
#print ('y_test shape:', y_test.shape)
#
#
#'''
################################################################################
### Hyperparameter tuning
#test_fold = np.repeat ([-1, 0], [X_train.shape [0], X_val.shape [0]])
#myPreSplit = PredefinedSplit (test_fold)
#
#
#model = KerasClassifier (build_fn = create_model, verbose = 2)
#batch_size = [64, 256]#, 2000]#10, 30, 50]
#epochs = [5, 10]
#learn_rate = [0.01]#, 0.01]#, 0.1, 0.2]
#dropout_rate = [0.0]
#weight_constraint = [0]#, 2, 3, 4, 5]
#filter_size = [2]#, 3]
#param_grid = dict (batch_size = batch_size, epochs = epochs,
#                   dropout_rate = dropout_rate, learn_rate = learn_rate,
#                   weight_constraint = weight_constraint,
#                   filter_size = filter_size)
#grid = GridSearchCV (estimator = model, param_grid = param_grid,
#                     scoring = 'accuracy', cv = myPreSplit, verbose = 2,
#                     n_jobs = 2)
#
#grid_result = grid.fit (np.concatenate ((X_train, X_val), axis = 0),
#                        np.concatenate ((y_train, y_val), axis = 0))
#print (grid_result.best_params_)
#
#print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip (means, stds, params):
#    print ("%f (%f) with: %r" % (mean, stdev, param))
#'''
#
################################################################################
### Finished model
#METRICS = [tf.keras.metrics.TruePositives (name = 'TP'),
#           tf.keras.metrics.FalsePositives (name = 'FP'),
#           tf.keras.metrics.TrueNegatives (name = 'TN'),
#           tf.keras.metrics.FalseNegatives (name = 'FN'),
#           tf.keras.metrics.BinaryAccuracy (name = 'Acc.'),
#           tf.keras.metrics.Precision (name = 'Prec.'),
#           tf.keras.metrics.Recall (name = 'Recall'),
#           tf.keras.metrics.AUC (name = 'AUC'),]
#BATCH_SIZE = 10000
#LEARNING_RATE = 0.001
#NUMBER_OF_EPOCHS = 5
#clf = create_model ()
#
################################################################################
### Compile the network
################################################################################
#clf.compile (optimizer = 'adam',
#             loss = 'binary_crossentropy',
#             metrics = METRICS)
#clf.summary ()
#
#
################################################################################
### Fit the network
################################################################################
#print ('\nFitting the network.')
#startTime = time.time ()
#history = clf.fit (X_train, y_train,
#                   batch_size = BATCH_SIZE,
#                   epochs = NUMBER_OF_EPOCHS,
#                   verbose = 2, #1 = progress bar, not useful for logging
#                   workers = 0,
#                   use_multiprocessing = True,
#                   #class_weight = 'auto',
#                   validation_data = (X_val, y_val))
#print (str (time.time () - startTime), 's to train model.')
#model.save('final_model.h5')


# save the final model to file
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# load train and test dataset
def load_dataset():
  # load dataset
  (trainX, trainY), (testX, testY) = cifar10.load_data()
  # one hot encode target values
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)
  return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
  # convert from integers to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')
  # normalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  # return normalized images
  return train_norm, test_norm

# define cnn model
def define_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))
  # compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# run the test harness for evaluating a model
def run_test_harness():
  # load dataset
  trainX, trainY, testX, testY = load_dataset()
  # prepare pixel data
  trainX, testX = prep_pixels(trainX, testX)
  # define model
  model = define_model()
  # fit model
  model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=0)
  # save model
  model.save('final_model.h5')

# entry point, run the test harness
run_test_harness()
