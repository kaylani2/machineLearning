import os

import flwr as fl
import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == "__main__":
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32)#, steps_per_epoch=3)
            print ('fitting')
            print (model.get_weights ())
            print (len (x_train))
            return model.get_weights(), len(x_train), len(x_train)
            #return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            print ('evaluating')
            print (loss)
            print (len (x_test))
            print ({"accuracy": accuracy})
            return len(x_test), loss, accuracy
            #return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())
