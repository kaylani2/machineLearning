import sys
import os
import time
import flwr as fl
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam

def get_smaller_model():
  #model = tf.keras.Sequential(
  #[
  #tf.keras.Input(shape=(32, 32, 3)),
  ##tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
  #tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  ##tf.keras.layers.Dropout (0.2),
  ##tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
  #tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
  #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.Flatten(),
  #tf.keras.layers.Dropout(0.1),
  #tf.keras.layers.Dense(10, activation="softmax"),
  #])
  #model.compile(SGD (lr=0.0001), "sparse_categorical_crossentropy",
                #metrics=[tf.keras.metrics.CategoricalAccuracy()])
  # compile model
  #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  opt = SGD(lr=0.0001, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary ()
  return model

def get_larger_model():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Dropout(0.1))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Dropout(0.1))
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(Dense(10, activation='softmax'))

  # compile model
  opt = SGD(lr=0.0001, momentum=0.9)
  #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary ()
  return model
