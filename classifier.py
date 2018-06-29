
"""
module classifier.py
"""

import tensorflow as tf
import numpy as np
import keras

global model,graph

model_json = open('data/model.json','r').read()
model = keras.models.model_from_json(model_json)
model.load_weights('data/model.h5')
graph = tf.get_default_graph()

print([layer.output for layer in model.layers])

def classify(x):
    with graph.as_default():
        y = model.predict(x.reshape((1,28,28,1))).reshape((10,))
        n = int(np.argmax(y,axis=0))
        y = [float(i) for i in y]
        return (y,n)