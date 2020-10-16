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

# Remove upsampling layers to reduce model output size from 1024 to 256
layers = generator._layers[:40] + generator._layers[41:45] + generator._layers[46:]
generator2 = tf.keras.Sequential(layers)

generator2.summary()

latents = np.random.RandomState(1000).randn(1000, *generator.input.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

images = generator2.predict(latents)

for i in range(len(latents)):
    utils.to_pil(images[i]).save(os.path.join(OUTPUT_DIR, f'preview-{i}.png'))

file_name, ext = os.path.splitext(os.path.basename(model_path))

generator2.save(os.path.join(OUTPUT_DIR, file_name + ext))
