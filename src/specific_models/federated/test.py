import nest_asyncio
nest_asyncio.apply()

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

print (tff.federated_computation(lambda: 'Hello, World!')())
