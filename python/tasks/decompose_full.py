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

w = generator._layers[2].weights[0].numpy()

u, s, vh = np.linalg.svd(w)

KEEP_DIMS = 150

w1 = u[:, :KEEP_DIMS] * s[:KEEP_DIMS]
w2 = vh[:KEEP_DIMS, :]

is_close = np.allclose(w, np.dot(w1, w2), atol=0.5)
print(f'is close {is_close}, reduction {(w1.size + w2.size) / w.size * 100:0.2f}%')

dense1 = tf.keras.layers.Dense(KEEP_DIMS, input_shape=[512],
                               use_bias=False,
                               name='Dense1',
                               kernel_initializer=tf.keras.initializers.Constant(w1)
                               )
dense2 = tf.keras.layers.Dense(8192, input_shape=[KEEP_DIMS],
                               use_bias=False,
                               name='Dense2',
                               kernel_initializer=tf.keras.initializers.Constant(w2)
                               )

layers = generator._layers[:2] + [dense1, dense2] + generator._layers[3:]
generator2 = tf.keras.Sequential(layers)

generator2.summary()

latents = np.random.RandomState(1000).randn(1000, *generator.input.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

images = generator2.predict(latents)

for i in range(len(latents)):
    utils.to_pil(images[i]).save(os.path.join(OUTPUT_DIR, f'preview-{i}.png'))

file_name, ext = os.path.splitext(os.path.basename(model_path))

generator2.save(os.path.join(OUTPUT_DIR, file_name + ext))
