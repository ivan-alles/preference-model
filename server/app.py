import base64
import io
import uuid

from flask import Flask, jsonify
from flask_cors import CORS

import numpy as np
from PIL import Image


rng = np.random.RandomState()

def get_image_data():
    image = np.broadcast_to(rng.uniform(0, 255, 3).astype(np.uint8), (256, 256, 3))
    pil_img = Image.fromarray(image, 'RGB')
    # pil_img = Image.open("D:\ivan\projects\progressive_growing_of_gans\img8.png", mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = base64.b64encode(byte_arr.getvalue()).decode('ascii')
    encoded_img = "data:image/png;base64," + encoded_img
    return encoded_img

# TODO(ia): replace ids by latents.
PICTURES = [
    {
        'id': uuid.uuid4().hex,
        'data': get_image_data(),
        'latents': [1.2, 2, 3]
    },
    {
        'id': uuid.uuid4().hex,
        'data': get_image_data(),
        'latents': [4.4, 5, 6]
    },
    {
        'id': uuid.uuid4().hex,
        'data': get_image_data(),
        'latents': [7, 8.8, 9.1]
    }
]

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/images', methods=['GET'])
def images():
    response_object = {'status': 'success', 'images': PICTURES}
    return jsonify(response_object)

if __name__ == '__main__':
    app.run()
