/* eslint-disable no-unused-vars */
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

// TODO(ia): temp code begin.
import axios from 'axios';
const PICTURES_URL = 'http://localhost:5000/images';
const LEARN_URL = 'http://localhost:5000/learn';
// TODO(ia): temp code end.

const MODEL_URL = '/karras2018iclr-celebahq-1024x1024.tfjs/model.json';

class Generator {
  constructor() {
    this.model = null;
  }

  async init() {
    console.log(`Loading model ${MODEL_URL} ...`)
    this.model = await loadGraphModel(MODEL_URL);
    console.log('Model loaded')
  }

  async generate(latents) {
    console.log(`Generator.generate(), this.model ${this.model}`)

    let output = this.model.predict(latents)
    let picture = tf.clipByValue(output, 0, 1.0);
    picture = picture.squeeze(); // TODO(ia): only 1 image is supported now. Extend to many.

    // TODO(ia): we draw the picture and then convert it into PNG.
    // This is probably not efficient. We can optimize this by d
    // directly on a canvas in Vue and save CPU time.
    let canvas = document.createElement("canvas");
    await tf.browser.toPixels(picture, canvas);
    let pictureData = canvas.toDataURL("image/png");
    return pictureData;
  }
}

class PreferenceModel {
  constructor(shape=512) {
      this.shape = shape;
      // Learnable parameters
      this.trainingExamples = [];
  }

  isRandom() {
      return this.trainingExamples.length == 0;
  }

  train(self, trainingExamples) {
    this.trainingExamples = trainingExamples;
    // this.r0 = -5;
  }

  generate(size, variance) {
      if (this.isRandom) {
        const latents = tf.randomNormal([size, 512]);
        return latents
      }

      const latents = [size, variance];
      return latents;
  }
}

class Engine {
  constructor () {
    console.log('Client Engine')
    this.generator = new Generator();
    this.preferenceModel = new PreferenceModel();
    this.initDone = false; // TODO(ia): this shall be no more necessary

    // TODO(ia): temp code begin.
    this.firstCall = true;
    this.isRandom = true;
    // TODO(ia): temp code end.
  }

  async init() {
    await this.generator.init();
    this.initDone = true;
    console.log('Engine.init() done')
  }

  async getPictures(count, variance) {
    
    const latents = this.preferenceModel.generate(count, variance);
    const picture = await this.generator.generate(latents);
    const data = {
        "picture": picture,
        "latents": []
    }
    return [data];

    // // TODO(ia): temp code begin.
    // if(this.firstCall) {
    //   this.firstCall = false;
    //   // reset to random pictures.
    //   await axios.post(LEARN_URL, []);
    // }

    // let result = await axios.get(PICTURES_URL, {
    //     params: {
    //         count: count,
    //         variance: variance
    //       }
    //     }
    //   );
    // return result.data.images;
    // // TODO(ia): temp code end.
  }

  async learn(likes) {
    if(!this.initDone)
      return;
    // TODO(ia): temp code begin.
    this.isRandom = likes.length === 0;
    await axios.post(LEARN_URL, likes);  }
    // TODO(ia): temp code end.
}


export { Engine }