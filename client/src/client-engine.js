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

/**
 * Sample from a symmetric Dirichlet distribution of dimension k and parameter a,
 * scaled around its mean.
 *
 * @param shape output shape, the output will have a shape of [shape, k]. 
 * @param k dimensionality.
 * @param a concentration parameter in [eps, +inf]. eps shall be > 0.01 or so to avoid nans.
 * @param scale scale: scale factor.
 * @returns a tensor of shape [shape, k].
 */
function scaledDirichlet(shape, k, a, scale=1) {
  // Use the gamma distribution to sample from a Dirichlet distribution.
  // See https://en.wikipedia.org/wiki/Dirichlet_distribution

  const y = tf.randomGamma([...shape, k], a);
  const x = tf.div(y, y.sum(-1, true));
  const mean = 1 / k;
  const d = tf.add(tf.mul(tf.sub(x, mean), scale), mean);

  // y = rng.gamma(np.full(k, a), size=size + (k,))
  // x = y / y.sum(axis=-1, keepdims=True)
  // mean = 1. / k
  // return (x - mean) * scale + mean

  return d;
}

/**
 * Converts n-dimensional spherical coordniates to cartesian coordinates. 
 * The resulting vectors will be on a unit sphere.
 * See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates.
 *
 * @param phi a tensor [size, n-1] of angles.
 *  phi[0:n-2] vary in range [0, pi], 
 *  phi[n-1] in [0, 2*pi] or in [-pi, pi].
 * @returns a tensor [size, n] of n-dimensional unit vectors.
 */
function sphericalToCartesian(phi) {
    const size = phi.shape[0];  // Number of vectors
    const n = phi.shape[1] + 1; // Dimentionality
    const sin = tf.sin(phi);

    let prodSin = tf.ones([size, n]);
    for(let i = 1; i < n; ++i) {
      const shifted = tf.concat([
        tf.ones([size, i]), 
        sin.slice([0, 0], [size, n - i])
      ], 1);
      prodSin = tf.mul(prodSin, shifted);
    }
    const cos = tf.cos(phi);
    const x = tf.mul(
      prodSin,
      tf.concat([cos, tf.ones([size, 1])], 1)
    );

    return x;
}

// TODO(ia): some functions here are exported only for tests.
// How to avoid this namespace clutter?
export { Engine, scaledDirichlet, sphericalToCartesian }