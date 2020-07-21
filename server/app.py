import base64
import io
import uuid

from flask import Flask, jsonify
from flask_cors import CORS

import numpy as np
from PIL import Image


rng = np.random.RandomState(1)

def get_response_image():
    # pil_img = Image.open("D:\ivan\projects\progressive_growing_of_gans\img8.png", mode='r')  # reads the PIL image
    image = np.empty((256, 256, 3), dtype=np.uint8)
    image = rng.uniform(0, 255, 3)
    pil_img = Image.fromarray(image, 'RGB')
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    encoded_img = encoded_img.replace('\n', '')
    style = f"width: 200px; height: 200px; background-size: cover; background-image: url(data:image/png;base64,{encoded_img});"
    return style

# TODO(ia): replace ids by latents.
IMAGES = [
    {
        'id': uuid.uuid4().hex,
        'image': get_response_image()
    },
    {
        'id': uuid.uuid4().hex,
        'image': get_response_image()
    },
    {
        'id': uuid.uuid4().hex,
        'image': get_response_image()
    }
]

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/books', methods=['GET'])
def all_books():
    response_object = {'status': 'success', 'books': IMAGES}
    return jsonify(response_object)

if __name__ == '__main__':
    app.run()
