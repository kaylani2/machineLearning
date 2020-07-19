import numpy as np 
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

DEVICE = 'GPU/:0'
tf.keras.backend.set_floatx('float64')

#getting the csv
CICIDS_DIRECTORY = '../../datasets/cicids/MachineLearningCVE/'
CICIDS_MONDAY_FILENAME = 'Monday-WorkingHours.pcap_ISCX.csv'
CICIDS_WEDNESDAY_FILENAME = 'Wednesday-workingHours.pcap_ISCX.csv'
CICIDS_MONDAY = CICIDS_DIRECTORY + CICIDS_MONDAY_FILENAME
CICIDS_WEDNESDAY = CICIDS_DIRECTORY + CICIDS_WEDNESDAY_FILENAME

dataFrame = pd.read_csv(CICIDS_WEDNESDAY)
## Remove NaN and inf values
dataFrame.replace ('Infinity', np.nan, inplace = True) ## Or other text values
dataFrame.replace (np.inf, np.nan, inplace = True) ## Remove infinity
dataFrame.replace (np.nan, 0, inplace = True)

#converting labels
dataFrame [' Label'] = dataFrame [' Label'].replace ('BENIGN', 0)
dataFrame [' Label'] = dataFrame [' Label'].replace ('DoS slowloris', 1)
dataFrame [' Label'] = dataFrame [' Label'].replace ('DoS Slowhttptest', 2)
dataFrame [' Label'] = dataFrame [' Label'].replace ('DoS Hulk', 3)
dataFrame [' Label'] = dataFrame [' Label'].replace ('DoS GoldenEye', 4)
dataFrame [' Label'] = dataFrame [' Label'].replace ('Heartbleed', 5)

#splitting dataset
train, test = train_test_split(dataFrame, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#make dataFrame into a data set
def df_to_dataset(dataFrame, shuffle=True, batch_size=32):
    dataFrame = dataFrame.copy()
    labels = dataFrame.pop(' Label')
    data_set = tf.data.Dataset.from_tensor_slices((dict(dataFrame), labels))
    if shuffle:
        data_set = data_set.shuffle(buffer_size=len(dataFrame))
    data_set = data_set.batch(batch_size)
    return data_set

#transform each part of the dataFrame into the data_set format
BATCH_SIZE = 32
train_ds = df_to_dataset(train, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test, shuffle=False, batch_size=BATCH_SIZE)

#feature columns preset for the model
## Note: All columns of this dataset are numeric
###1st approach: using all columns.
feature_columns = []
numeric_headers = []
categorical_headers = []
count = 0
for feature, label in train_ds.take(1):
    for key in list(feature.keys()):
        feature_columns.append(feature_column.numeric_column(key))

###2nd approach: Get the most relevant columns (60% of the total) 


#feature_layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

initializer = tf.initializers.VarianceScaling(scale=2.0)
hidden_layer_size, num_classes = 128, 6

layers = [
    feature_layer,
    tf.keras.layers.Dense(hidden_layer_size, activation='relu', initializer=initializer),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu', initializer=initializer),
    tf.keras.layers.Dense(num_classes, activation='softmax', initializer=initializer),

]
model = tf.keras.Sequential(layers)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.sparse_categorical_accuracy])

with tf.device(DEVICE):
    model.fit(train_ds, validation_data=val_ds, epochs=5)