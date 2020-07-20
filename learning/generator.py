"""
    Test generator inference from a saved tf.keras model.
"""

import numpy as np
import tensorflow as tf

class Generator:
    def __init__(self, model_path):
        self._generator = tf.keras.models.load_model(model_path)

    def generate(self, latents, batch_size=8):
        outputs = []
        for i in range(0, len(latents), batch_size):
            outputs.append(self._generator.predict(latents[i:i + batch_size]))

        images = np.concatenate(outputs)
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0, 255]
        images = np.moveaxis(images, 1, 3)  # NCHW => NHWC

        return images

