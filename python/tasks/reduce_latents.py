# Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import sys
import os

import numpy as np
import tensorflow as tf

from tasks import utils

class PixelNormLayer(tf.keras.layers.Layer):
    """
    Pixelwise feature vector normalization.
    """
    def call(self, x):
        return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


model_path = sys.argv[1]

generator = tf.keras.models.load_model(model_path)
generator.summary()
utils.print_sizes(generator)

w = generator._layers[2].weights[0].numpy()

u, s, vh = np.linalg.svd(w)

KEEP_DIMS = 512
w_new = np.dot(np.diag(s[:KEEP_DIMS]), vh[:KEEP_DIMS, :])

dense = tf.keras.layers.Dense(8192, input_shape=[KEEP_DIMS],
                               use_bias=False,
                               name='Dense',
                               kernel_initializer=tf.keras.initializers.Constant(w_new)
                               )
input_var = tf.keras.layers.Input((KEEP_DIMS,), name='latents')

layers = [input_var, PixelNormLayer(), dense] + generator._layers[3:]
generator2 = tf.keras.Sequential(layers)

generator2.summary()
utils.print_sizes(generator2)

# Cut from the full vectors to ensure the same beginnings for various dimensions
latents = np.random.RandomState(1).randn(100, 512)[:, :KEEP_DIMS]

for i in range(len(latents)):
    images = generator2.predict(latents[i:i+1, :])
    utils.to_pil(images[0]).save(os.path.join(OUTPUT_DIR, f'img-{i}-{KEEP_DIMS}.png'))

file_name, ext = os.path.splitext(os.path.basename(model_path))

generator2.save(os.path.join(OUTPUT_DIR, file_name + ext))
