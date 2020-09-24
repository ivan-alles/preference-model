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

import cv2
import numpy as np
import tensorflow as tf
import PIL.Image

from tasks import utils

PREVIEW_SIZE = 256
OUTPUT_DIR = 'output'

def to_pil(numpy_image):
    return PIL.Image.fromarray(np.rint(np.clip(numpy_image, 0, 1) * 255.0).astype(np.uint8), 'RGB')

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

rng = np.random.RandomState(1)

for epoch_i in range(10):
    for batch_i in range(1000):
        inputs = rng.standard_normal(size=(16, 512))
        outputs_full = generator.predict(inputs)
        targets = []
        for i in range(len(outputs_full)):
            image = cv2.resize(outputs_full[i], (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_CUBIC)
            targets.append(image)
            if batch_i == 0:
                to_pil(targets[-1]).save(os.path.join(OUTPUT_DIR, f'target-{i:02d}.png'))
        targets = np.stack(targets)
    print('hello')


# Test
latents = np.random.RandomState(1000).randn(1000, *generator.input.shape[1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

intermediate = common.predict(latents)
images = full.predict(intermediate)


# Save images as PNG.
for i in range(images.shape[0]):
    to_pil(images[i]).save(os.path.join(OUTPUT_DIR, f'image-{i}.png'))

file_name, ext = os.path.splitext(model_path)

common.save(os.path.join(OUTPUT_DIR, file_name + '-common' + ext))
full.save(os.path.join(OUTPUT_DIR, file_name + '-full' + ext))
