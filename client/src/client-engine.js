/* eslint-disable no-unused-vars */
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

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
      this.trainingExamples = tf.tensor([]);
  }

  get isRandom() {
    console.log("isRandom", this.trainingExamples);
    const r = this.trainingExamples.shape[0] === 0;
    console.log("isRandom end", this.trainingExamples);
    return r;
  }
  
  /**
   * Train model on given training examples.
   * @param trainingExamples a tensor [size, this.shape]. If size === 0 
   * (regardless of the shape), revert to uniform random generator.
   */
  train(trainingExamples) {
    console.log("train");
    this.trainingExamples = trainingExamples;
    // this.r0 = -5;
  }

  generate(count, variance) {
    if (this.isRandom) {
      console.log('Generate random');
      // Sample uniform random points on n-sphere.
      // See https://mathworld.wolfram.com/HyperspherePointPicking.html
      // Do not normalize the length, as we will only work with the angles.
      return tf.randomNormal([count, 512]);
    }

    // For variance from 0 to 4
    // a, scale, std
    const params = [
        (10, 1, .02),
        (3, 1, .03),
        (1, 1, .05),    // Middle range - a uniform dist. in convex combination of training examples
        (1, 1.5, .07),  // Start going outside of the convex combination of training examples
        (1.2, 2.0, .1)  // Concentrate in the middle, as the values at the boarder have little visual difference
    ][variance];

    const k = this.trainingExamples.shape[0];
    let phi = cartesianToSpherical(this.trainingExamples);
    // Convert [0, pi] -> [-pi, pi] for all but the last column (it's already so)
    const cols = phi.shape[1];
    const a = tf.tensor(Array(cols - 1).fill(2).concat(1), [1, cols]);
    const b = tf.tensor(Array(cols - 1).fill(Math.PI).concat(0), [1, cols]);
    phi = phi.mul(a).sub(b);

    let outputPhi = null;
    if (k == 1) {
      console.log('Generate from one training example');
      // Only one training example, it will be varied later by a normal "noise".
      outputPhi = phi.broadcastTo([count, phi.shape[1]]);
    }
    else {
      throw Error("Not implemented");
    }

    // Convert back [-pi, pi] -> [0, pi].
    outputPhi = outputPhi.add(b).div(a);

    console.log('sphericalToCartesian()', outputPhi.shape);
    const output = sphericalToCartesian(outputPhi);
    console.log('generate() end');
    return output;
  }
}

/**
 * Combines alogrithms to implement application logic.
 * Converts the data between UI (plain javascript) to internal representations (tf.tensors).
 */
class Engine {
  constructor () {
    console.log('Client Engine')
    this.generator = new Generator();
    this.preferenceModel = new PreferenceModel();
    this.initDone = false; // TODO(ia): this shall be no more necessary
  }

  async init() {
    await this.generator.init();
    this.initDone = true;
    console.log('Engine.init() done')
  }

  async getPictures(count, variance) {
    
    const latentsTensor = this.preferenceModel.generate(count, variance);
    const latents = await latentsTensor.array();
    const pictures = await this.generator.generate(latentsTensor);

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
  }

  async learn(likes) {
    this.preferenceModel.train(tf.tensor(likes));
  }
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
 * Converts spherical to cartesian coordinates in n dimensions. 
 * The resulting vectors will be on a unit sphere.
 * See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates.
 *
 * @param phi a tensor [size, n-1] of angles.
 *  phi[0:n-2] is in [0, pi], 
 *  phi[n-1] is in [-pi, pi].
 * @returns a tensor [size, n] of n-dimensional unit vectors.
 */
function sphericalToCartesian(phi) {
    const size = phi.shape[0];  // Number of vectors
    const n = phi.shape[1] + 1; // Dimentionality
    const sin = tf.sin(phi).arraySync();
    let prodSinBuffer = tf.buffer([size, n]);
    for(let i = 0; i < size; ++i) {
      let p = 1;
      prodSinBuffer.set(p, i, 0);
      for(let j = 0; j < n-1; ++j) {
        p *= sin[i][j];
        prodSinBuffer.set(p, i, j+1);
      }
    }
    const cos = tf.cos(phi);
    const x = tf.mul(
      prodSinBuffer.toTensor(),
      tf.concat([cos, tf.ones([size, 1])], 1)
    );
    return x;
}

/**
 * Converts cartesian to spherical coordinates in n dimensions. 
 * The resulting r is ignored, as if the input were on the unit sphere.
 * See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates.
 *
 * @param x a tensor [size, n] of cartesian n-dimensional vectors.
 * @returns a tensor [size, n-1] of angles phi, such that 
 * phi[0:n-2] is in [0, pi], phi[n-1] is in [-pi, pi].
 */
function cartesianToSpherical(x) {
  const n = x.shape[1]; // Dimentionality

  const x2 = tf.reverse(tf.square(x), 1);
  const cn = tf.reverse(tf.sqrt(tf.cumsum(x2, 1)), 1).slice([0, 0], [-1,  n - 1]);
  // First n-1 columns and the last column.
  const xParts = x.split([n-1, 1], 1);
  // Elements <= EPSILON will be set to 0.
  const EPSILON = 1e-10;
  const xn = xParts[0].div(cn.add(EPSILON));

  let phi = tf.acos(xn);
  // Reset elements <= EPSILON to 0.
  phi = phi.mul(cn.greater(EPSILON));
  const phiParts = phi.split([n-2, 1], 1);

  /*
  The description in wikipedia boils down to changing the sign of the  phi_(n-1) 
  (using 1-based indexing) if and only if
  1. there is no k such that x_k != 0 and all x_i == 0 for i > k
  and
  2. x_n < 0
  */

  // -1 if last column of x < 0, otherwise 1
  let s = xParts[1].less(0).mul(-2).add(1); 
  phiParts[1] = phiParts[1].mul(s);
  
  return tf.concat(phiParts, 1);
}

// TODO(ia): some functions here are exported only for tests.
// How to avoid this namespace clutter?
export { Engine, PreferenceModel, scaledDirichlet, sphericalToCartesian, cartesianToSpherical }