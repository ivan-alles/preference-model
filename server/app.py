import base64
import io

import flask
from flask_cors import CORS

import numpy as np
from PIL import Image

from server import preference_model

rng = np.random.RandomState(0)

#  Set to true for a fast startup and responses. Is useful to test the client.
DUMMY_IMAGES = False

if not DUMMY_IMAGES:
    from server import generator
    generator = generator.Generator('karras2018iclr-celebahq-1024x1024.tf')


preference_model = preference_model.SphericalLinearPreferenceModel(rng=rng)

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
    count = int(flask.request.args['count'])
    variance = int(flask.request.args['variance'])
    latents = preference_model.generate(count, variance)
    if DUMMY_IMAGES:
        images = np.broadcast_to(rng.uniform(0, 255, (count, 1, 1, 3)).astype(np.uint8),
                                 (count, 256, 256, 3))
    else:
        images = generator.generate(latents)

    image_objects = []
    for i in range(len(images)):
        image = images[i]
        image_object = {
            'picture': encode_image(image),
            'latents': latents[i].tolist()  # rng.randn(512).tolist()
        }
        image_objects.append(image_object)
    response_object = {'status': 'success', 'images': image_objects}
    return flask.jsonify(response_object)


@app.route('/learn', methods=['POST'])
def learn():
    """
    Learn user preferences from likes.
    :return: an HTTP response.
    """
    post_data = flask.request.get_json()
    latents = np.array(post_data)
    preference_model.train(latents)
    response_object = {'status': 'success'}
    return flask.jsonify(response_object)


if __name__ == '__main__':
    app.run()

