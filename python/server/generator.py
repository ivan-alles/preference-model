# Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

"""
    Test generator inference from a saved tf.keras model.
"""

import numpy as np
import tensorflow as tf

class Generator:
    def __init__(self, model_path):
        self._model = tf.keras.models.load_model(model_path)

    def generate(self, latents, batch_size=8):
        outputs = []
        for i in range(0, len(latents), batch_size):
            outputs.append(self._model.predict(latents[i:i + batch_size]))

        images = np.concatenate(outputs)
        images = np.rint(np.clip(images, 0, 1) * 255.0).astype(np.uint8)

        return images

