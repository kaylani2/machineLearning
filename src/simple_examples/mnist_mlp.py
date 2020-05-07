## Creates a MLP and uses it to classify handwritten digits (mnist)
from __future__ import print_function

import keras
import matplotlib.pyplot as plt
import mglearn



batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
## Load the data
### 1: Data Collection
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data ()

print ('x_train type:', type (x_train))
print ('x_train shape before reshape', x_train.shape)

## Plot the first 4 images
image_datas = x_train [:4]
f, axarr = plt.subplots (2, 2)
axarr [0, 0].imshow (image_datas [0], cmap='gray')
axarr [0, 1].imshow (image_datas [1], cmap='gray')
axarr [1, 0].imshow (image_datas [2], cmap='gray')
axarr [1, 1].imshow (image_datas [3], cmap='gray')
plt.show ()

## Turn the 28x28 pixels images into a length 784 array
x_train = x_train.reshape (60000, 784)
x_test = x_test.reshape (10000, 784)
print ('x_train shape after reshape', x_train.shape)


## Apply normalization
### 2: Data pre-processing
x_train = x_train.astype ('float32')
x_test = x_test.astype ('float32')
x_train /= 255
x_test /= 255
print ('x_train shape after normalization', x_train.shape)
print (x_train.shape [0], 'train samples')
print (x_test.shape [0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical (y_train, num_classes)
y_test = keras.utils.to_categorical (y_test, num_classes)

from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential ()
model.add (Dense (512, activation='relu', input_shape= (784, )))
model.add (Dropout (0.2))
model.add (Dense (512, activation='relu'))
model.add (Dropout (0.2))
model.add (Dense (num_classes, activation='softmax'))

print ('Model summary:')
model.summary ()

## Compile the network
from keras.optimizers import RMSprop
model.compile (loss='categorical_crossentropy',
              optimizer=RMSprop (),
              metrics= ['accuracy'])

### Train the network
history = model.fit (x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data= (x_test, y_test))

## Evaluate and print the results
score = model.evaluate (x_test, y_test, verbose=0)
print ('Test loss:', score [0])
print ('Test accuracy:', score [1])
