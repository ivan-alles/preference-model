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

    const input = tf.randomNormal([1, 512]);
    let output = this.model.predict(input)

    output = tf.clipByValue(output, 0, 1.0);
    output = output.squeeze();

    var canvas = document.getElementById("testCanvas")
    await tf.browser.toPixels(output, canvas);
    console.log('done');
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
      const output = [size, variance];
      return output;
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
    console.log(`Engine.getPictures() started, this.initDone ${this.initDone}`)
    if(this.initDone)
      await this.generator.generate([]);

    // TODO(ia): temp code begin.
    if(this.firstCall) {
      this.firstCall = false;
      // reset to random pictures.
      await axios.post(LEARN_URL, []);
    }

    let result = await axios.get(PICTURES_URL, {
        params: {
            count: count,
            variance: variance
          }
        }
      );
    return result.data.images;
    // TODO(ia): temp code end.
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