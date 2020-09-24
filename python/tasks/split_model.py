"""
Current generator has the following structure:

    input (512) -> start -> (size, channels) repeated (size * 2, channels / 2) -> to_rgb (size, 3)

    where repeated is the following:

    (size, channels) -> upsampling (size * 2, channels) -> conv (size * 2, channels / 2) -> pixel_norm ->
    conv (size * 2, channels / 2) -> pixel_norm

    We can split the model between any repeated layers into 3 parts to generate previews of desired PREVIEW_SIZE :

    (512) -> common -> (PREVIEW_SIZE, channels)
    (PREVIEW_SIZE, channels) -> to_rgb_preview (PREVIEW_SIZE, 3)
    (PREVIEW_SIZE, channels) -> full (1024, 3)

    to_rgb_preview is a convolution layer that needs new training.
"""
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
tf.keras.utils.plot_model(generator,
                          to_file=os.path.join(OUTPUT_DIR, 'generator.svg'),
                          dpi=50, show_shapes=True)

utils.print_sizes(generator)

# Remove upsampling layers to reduce model output size from 1024 to 256
layers = generator._layers[:39] + generator._layers[50:]
generator2 = tf.keras.Sequential(layers)

generator2.summary()

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *generator.input.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

images = generator2.predict(latents)

# Convert to bytes in range [0, 255]
images = np.rint(np.clip(images, 0, 1) * 255.0).astype(np.uint8)

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save(os.path.join(OUTPUT_DIR, f'img{idx}.png'))

generator2.save(os.path.join(OUTPUT_DIR, os.path.basename(model_path)))