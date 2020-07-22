import base64
import io
import uuid

import flask
from flask_cors import CORS

import numpy as np
from PIL import Image


rng = np.random.RandomState()

def encode_image(image):
    """
    Convert an image to the format accepted by a browser.
    :param image: image as numpy array.
    :return: an encoded image.
    """
    pil_img = Image.fromarray(image, 'RGB')
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = base64.b64encode(byte_arr.getvalue()).decode('ascii')
    encoded_img = "data:image/png;base64," + encoded_img
    return encoded_img


# instantiate the app
app = flask.Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/images', methods=['GET'])
def images():
    """
    Generate the next portion of images based on current settings.
    :return: an HTTP response containing a list of image objects.
    """
    num_images = 3
    images = []
    for i in range(num_images):
        image = np.broadcast_to(rng.uniform(0, 255, 3).astype(np.uint8), (256, 256, 3))
        image_object = {
            'id': uuid.uuid4().hex,
            'data': encode_image(image),
            'latents': rng.randn(512).tolist()
        }
        images.append(image_object)
    response_object = {'status': 'success', 'images': images}
    return flask.jsonify(response_object)


@app.route('/learn', methods=['POST'])
def learn():
    """
    Learn user preferences from likes.
    :return: an HTTP response.
    """
    post_data = flask.request.get_json()
    response_object = {'status': 'success'}
    return flask.jsonify(response_object)


if __name__ == '__main__':
    app.run()

