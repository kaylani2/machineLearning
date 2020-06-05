## @TODO: streamline virtual environment, lock compatible versions
## Versoes atuais (07/11/2019)
# Python version: 3.6.7 (default, Oct 22 2018, 11:32:17)
# [GCC 8.2.0]
# pandas version: 0.25.2
# matplotlib version: 3.1.1
# NumPy version: 1.17.3
# SciPy version: 1.3.1
# IPython version: 7.8.0
# scikit-learn version: 0.21.3
# keras version: 2.3.1
# pickle version: 4.0

## O import do Keras vai produzir uma penca de warnings para a nova versao do scikit

import sys
print ('Python version:', sys.version)

import pandas as pd
print ('pandas version:', pd.__version__)

import matplotlib
print ('matplotlib version:', matplotlib.__version__)

import numpy as np
print ('NumPy version:', np.__version__)

import mglearn
print ('Mglearn version:', mglearn.__version__)

import tflearn
print ('tflearn version:', tflearn.__version__)

import scipy as sp
print ('SciPy version:', sp.__version__)

import IPython
print ('IPython version:', IPython.__version__)

import sklearn
print ('scikit-learn version:', sklearn.__version__)

## Keras produces a FutureWarning with scikit >= 1.17.0
## You can downgrade to scikit 1.16.1 to fix it
import keras
print ('keras version:', keras.__version__)

## Not working \/
#import tersoflow
#print ('tersoflow version:', tersoflow.__version__)

#import pylab
#print ('pylab', pylab.__version__)
## pylab is a module in matplotlib

import pickle
print ('pickle version:', pickle.format_version)

