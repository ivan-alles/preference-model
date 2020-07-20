"""
    Test generator inference from a saved tf.keras model.
"""

import numpy as np
import tensorflow as tf

class Generator:
    def __init__(self):
        self._generator = tf.keras.models.load_model('karras2018iclr-celebahq-1024x1024.tf')

    def generate(self, latents, minibatch_size = 8):
        outputs = []
        for i in range(0, len(latents), minibatch_size):
            outputs.append(generator.predict(latents[i:i + minibatch_size]))

        images = np.concatenate(outputs)
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0, 255]
        images = images.moveaxis(images, 1, 3)  # NCHW => NHWC

        return images

