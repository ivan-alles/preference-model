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

  /**
   * Generate pictures from latents.
   *
   * @param latents a [count, 512] tensor of latents.
   * @returns an array of pictures.
   */
  async generate(latents) {
    console.log(`Generator.generate(), this.model ${this.model}`)

    const count = latents.shape[0];

    let output = this.model.predict(latents);
    output = tf.clipByValue(output, 0, 1.0);
    // Convert to an array of tensors of shapes [1, H, W, 3]
    const pictures = tf.split(output, count, 0); 
    const pictureData = [];

    for(let i = 0; i < count; ++i) {
      // TODO(ia): we draw the picture and then convert it into PNG.
      // This is probably not efficient. We can optimize this by d
      // directly on a canvas in Vue and save CPU time.
      let canvas = document.createElement("canvas");
      const picture = tf.squeeze(pictures[i]);
      await tf.browser.toPixels(picture, canvas);
      pictureData.push(canvas.toDataURL("image/png"));
    }
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

  generate(count, variance) {
      if (this.isRandom) {
        // Sample uniform random points on n-sphere.
        // See https://mathworld.wolfram.com/HyperspherePointPicking.html
        // Do not normalize the length, as we will only work with the angles.
        return tf.randomNormal([count, 512]);
      }
      const latents = [count, variance];
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
    
    const latentsTensor = this.preferenceModel.generate(count, variance);
    const pictures = await this.generator.generate(latentsTensor);
    const latents = tf.split(latentsTensor, count, 0); // Convert to array of 1d tensors.

    const result = [];
    for(let i = 0; i < count; ++i) {
      result.push(
        {
          "picture": pictures[i],
          "latents": latents[i]
        }
      );
    }

    return result;

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