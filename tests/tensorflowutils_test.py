import json
import unittest

import numpy as np
import tensorflow as tf

from mlutils.tensorflowutils import tensorflowutils


class TestStringMethods(unittest.TestCase):

    # def test_fit(self):
    #     c = json.loads(tensorflowutils.example_conf)
    #     model = tensorflowutils.build_model_core(c)
    #     x = np.random.rand(200, 60, 1)
    #     y = np.random.rand(200, 1)
    #     tensorflowutils.fit_model(model, x=x, y=y, epochs=5, batch_size=1)

    def test_train(self):
        c = json.loads(tensorflowutils.example_conf)
        x = np.random.rand(100, 60, 1)
        y = np.random.rand(100, 1)
        x_test = np.random.rand(2, 60, 1)
        y_test = np.random.rand(2, 1)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        tensorflowutils.train_model(c, x, y, x_test, y_test, epochs=5, batch_size=64, callbacks=[callback])


if __name__ == '__main__':
    unittest.main()
