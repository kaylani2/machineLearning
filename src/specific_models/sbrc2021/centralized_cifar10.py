import sys
from matplotlib import pyplot
import os
import time
import flwr as fl
import numpy as np
import tensorflow as tf
from model import get_larger_model
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
# K: Prevent TF from using GPU (not enough memory)
os.environ ["CUDA_VISIBLE_DEVICES"] = "-1"

def summarize_diagnostics (history):
  # plot loss
  pyplot.subplot (211)
  pyplot.title ('Cross Entropy Loss')
  pyplot.plot (history.history ['loss'], color='blue', label='train')
  pyplot.plot (history.history ['val_loss'], color='orange', label='test')
  # plot accuracy
  pyplot.subplot (212)
  pyplot.title ('Classification Accuracy')
  pyplot.plot (history.history ['accuracy'], color='blue', label='train')
  pyplot.plot (history.history ['val_accuracy'], color='orange', label='test')
  # save plot to file
  filename = sys.argv [0].split ('/') [-1]
  pyplot.savefig (filename + '_plot.png')
  pyplot.close ()

# load train and test dataset
def load_dataset ():
  # load dataset
  (X_train, y_train), (X_test, y_test) = cifar10.load_data ()
  # one hot encode target values
  y_train = to_categorical (y_train)
  y_test = to_categorical (y_test)
  return X_train, y_train, X_test, y_test

def main ():
  EPOCH_BATCHES = 4
  X_train, y_train, X_test, y_test = load_dataset ()
  try:
    model = load_model ('./models/cnn.h5')
    print ('Model loaded from disk.')
  except:
    model = get_larger_model ()
    print ('New model.')
  for _ in range (EPOCH_BATCHES):
    history = model.fit (X_train, y_train, epochs=25, batch_size=128, validation_split=0.2, verbose=1)
    model.save ('./models/cnn.h5')

  summarize_diagnostics (history)

def test ():
  _, _, X_test, y_test = load_dataset ()
  model = load_model ('./models/cnn.h5')
  print ('Model loaded from disk.')
  _, acc = model.evaluate (X_test, y_test, verbose=0)
  print ('Accuracy: %.3f' % (acc * 100.0))

if __name__ == "__main__":
  main ()
  #test ()
