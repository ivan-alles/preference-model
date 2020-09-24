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

PREVIEW_SIZE = 256
OUTPUT_DIR = 'output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = sys.argv[1]

generator = tf.keras.models.load_model(model_path)
generator.summary()
tf.keras.utils.plot_model(generator,
                          to_file=os.path.join(OUTPUT_DIR, 'generator.svg'),
                          dpi=50, show_shapes=True)
utils.print_sizes(generator)

split_idx = None

for i, layer in enumerate(generator.layers):
    if '.UpSampling2D' in str(layer) and layer.input.shape[1] == PREVIEW_SIZE:
        split_idx = i
        break

print(f'Splitting at index {split_idx}')

common = tf.keras.Sequential(generator._layers[:split_idx])
full = tf.keras.Sequential(generator._layers[split_idx:])

tf.keras.utils.plot_model(common,
                          to_file=os.path.join(OUTPUT_DIR, 'common.svg'),
                          dpi=50, show_shapes=True)

tf.keras.utils.plot_model(full,
                          to_file=os.path.join(OUTPUT_DIR, 'full.svg'),
                          dpi=50, show_shapes=True)



# Test
latents = np.random.RandomState(1000).randn(1000, *generator.input.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

intermediate = common.predict(latents)
images = full.predict(intermediate)

# Convert to bytes in range [0, 255]
images = np.rint(np.clip(images, 0, 1) * 255.0).astype(np.uint8)

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save(os.path.join(OUTPUT_DIR, f'img{idx}.png'))

file_name, ext = os.path.splitext(model_path)

common.save(os.path.join(OUTPUT_DIR, file_name + '-common' + ext))
full.save(os.path.join(OUTPUT_DIR, file_name + '-full' + ext))

