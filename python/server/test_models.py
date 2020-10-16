# Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import PIL.Image

from server import generator
from server import preference_model


def save_images():
    for i in range(len(images)):
        PIL.Image.fromarray(images[i], 'RGB').save(f'img{epoch}-{i}.png')


generator = generator.Generator('karras2018iclr-celebahq-1024x1024.tf')
preference_model = preference_model.PreferenceModel()

epoch = 0
latents = preference_model.generate(10)
images = generator.generate(latents)
save_images()

epoch = 1
preference_model.train(latents[[3, 5, 7], :])
latents = preference_model.generate(10)
images = generator.generate(latents)
save_images()

epoch = 2
preference_model.train(latents[[1, 2, 9], :])
latents = preference_model.generate(10)
images = generator.generate(latents)
save_images()

epoch = 3
preference_model.train(latents[[2, 4], :])
latents = preference_model.generate(10)
images = generator.generate(latents)
save_images()

epoch = 4
preference_model.train(latents[[9], :])
latents = preference_model.generate(10)
images = generator.generate(latents)
save_images()

