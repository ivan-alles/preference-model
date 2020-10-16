# Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import sys
import os

import numpy as np
import tensorflow as tf
import PIL.Image

from tasks import utils

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


model_path = sys.argv[1]

generator = tf.keras.models.load_model(model_path)
generator.summary()
utils.print_sizes(generator)

for i, layer in enumerate(generator.layers):
    w = None
    if '.Conv' in str(layer):
        w = layer.kernel.numpy().reshape(-1, layer.kernel.shape[-1])
    elif '.Dense' in str(layer):
        w = layer.kernel.numpy()
    if w is None:
        continue
    u, s, vh = np.linalg.svd(w)
    print(layer.name)
    print(s)
