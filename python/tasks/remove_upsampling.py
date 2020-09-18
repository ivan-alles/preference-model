import sys
import os

import numpy as np
import tensorflow as tf
import PIL.Image

model_path = sys.argv[1]

generator = tf.keras.models.load_model(model_path)
generator.summary()

# Remove upsampling layers to reduce model output size from 1024 to 256
layers = generator._layers[:40] + generator._layers[41:45] + generator._layers[46:]
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