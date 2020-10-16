# Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import PIL
import numpy as np

def print_sizes(model):
    """
    Prints the sizes of intermediate tensors and weights.
    """

    for i in range(len(model.layers)):
        layer = model.layers[i]
        output_shape = layer.output.shape[1:]
        output_size = np.prod(output_shape)
        print(f'{i} {type(layer)} {layer.name} output {output_shape}, {output_size}')
        for w in layer.weights:
            weight_size = np.prod(w.shape)
            print(f'weight {w.shape} {weight_size}')

def to_pil(numpy_image):
    return PIL.Image.fromarray(np.rint(np.clip(numpy_image, 0, 1) * 255.0).astype(np.uint8), 'RGB')
