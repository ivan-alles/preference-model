# Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import sys
import os

import numpy as np
import tensorflow as tf
import PIL.Image

model_path = sys.argv[1]

generator = tf.keras.models.load_model(model_path)
generator.summary()

for i in range(len(generator.layers)):
    layer = generator.layers[i]
    output_size = np.prod(layer.output.shape[1:])
    print(f'{i} {layer} output size {output_size}')
    for w in layer.weights:
        weight_size = np.prod(w.shape)
        print(f'weight size {weight_size}')

# Replace Dense layer and 2 upsampling layers
layers = generator.layers[:2] + [tf.keras.layers.RepeatVector(16)] + \
    generator.layers[3:40] + generator._layers[41:45] + generator._layers[46:]
generator2 = tf.keras.Sequential(layers)

generator2.summary()

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *generator.input.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

images = generator2.predict(latents)

# Convert to bytes in range [0, 255]
images = np.rint(np.clip(images, 0, 1) * 255.0).astype(np.uint8)

os.makedirs('.temp', exist_ok=True)
# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('.temp/img%d.png' % idx)

generator2.save(os.path.join('.temp', os.path.basename(model_path)))