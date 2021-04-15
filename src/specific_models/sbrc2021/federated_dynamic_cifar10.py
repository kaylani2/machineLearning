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
import sys
import os
import time
from multiprocessing import Process
from typing import Tuple
from keras.optimizers import SGD

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.server.strategy import FedAvg
from model import get_smaller_model

import dataset

# generate random integer values
from random import seed
from random import randint

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# K: Prevent TF from using GPU (not enough memory)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    # Exposes the server by default on port 8080
    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})


def start_client(dataset: DATASET) -> None:
    """Start a single client with the provided dataset."""


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

    opt = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Unpack the CIFAR-10 dataset partition
    (x_train, y_train), (x_test, y_test) = dataset

    # Define a Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            """Return current weights."""
            return model.get_weights()

        def fit(self, parameters, config):
            """Fit model and return new weights as well as number of training
            examples."""
            model.set_weights(parameters)
            # Remove steps_per_epoch if you want to train over the full dataset
            # https://keras.io/api/models/model_training_apis/#fit-method
            #nap_time = randint (0, 5)
            #time.sleep (nap_time)
            #print ("Slept for", nap_time,  "seconds.")
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_split=0.2)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            """Evaluate using provided parameters."""
            #model.save ('federated_constant_cnn.h5')
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            print ('"Loss:', loss, ". Accuracy:", accuracy, ".")
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=CifarClient())


def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Load the dataset partitions
    partitions = dataset.load(num_partitions=num_clients)

    # Start all the clients
    for partition in partitions:
        client_process = Process(target=start_client, args=(partition,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    try:
        num_rounds = int (sys.argv [1])
        num_clients = int (sys.argv [2])
        fraction_fit = int (sys.argv [3])
        epochs = int (sys.argv [4])
        batch_size = int (sys.argv [5])
        steps_per_epoch = int (sys.argv [6])
    except:
        num_rounds = 5
        num_clients = 2
        fraction_fit = 1
        epochs = 5
        batch_size = 64
        steps_per_epoch = 5

    start_time = time.time ()
    run_simulation(num_rounds=num_rounds, num_clients=num_clients,
                   fraction_fit=fraction_fit)
    print (str (time.time () - start_time), 'seconds to run', num_rounds,
                                            'rounds with', num_clients,
                                            'clients, fractioned:',
                                            fraction_fit, '.')
