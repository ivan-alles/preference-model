/* eslint-disable no-unused-vars */
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

let MODEL_URLS = 
{   
  common: '/karras2018iclr-celebahq-1024x1024-common.tfjs/model.json',
  preview: '/karras2018iclr-celebahq-1024x1024-preview.tfjs/model.json',
  full: '/karras2018iclr-celebahq-1024x1024-full.tfjs/model.json'
}

class Generator {
  constructor(logger) {
    this.logger = logger;
    this.models = {};
  }

  async init() {
    for (let [key, url] of Object.entries(MODEL_URLS)) {
      if (process.env.NODE_ENV === 'production' ) {
        url = '/preference-model' + url;      
      }
      console.log(`Loading model ${url} ...`)
      this.models[key] = await loadGraphModel(url);
      console.log(`Model loaded`)
    }
  }

  /**
   * Generate pictures from latents. It will try to create images for all model listed in modelNames. 
   * In case of an error it will either raise an exception or return the current result with 
   * failed or missing images set to 'ERROR'.
   *
   * @param {Tensor} latents a [size, 512] tensor of latents.
   * @param {string[]} modelNames model names ('preview', 'full').
   * @returns {Object[]} an array of objects with fields preview and full. If there was a error in 
   * generation of a single image, it value is set to 'ERROR'.
   */
  async generate(latents, modelNames) {
    const result = [];
    for(let i = 0; i < latents.shape[0]; ++i) {
      const element = {};
      for(const modelName of modelNames) {
        element[modelName] = 'ERROR';
      }

      result.push(element);
    }

    let intermediate = null;
    try {
      intermediate = this.models['common'].predict(latents);
      // let canvas = document.createElement('canvas');
      for(const modelName of modelNames) {
        if(!this.models[modelName]) {
          // We could have delete a model due to an error.
          continue;
        }

        const pictures = await this.makePictures(modelName, intermediate);
        for(let i = 0; i < result.length; ++i) {
          result[i][modelName] = pictures[i];
        }
      }
    }
    catch(error) {
      //this.logger.logException('Generator.generate', error);
      console.error(error);
    }
    finally {
      tf.dispose(intermediate);
    }
    return result; 
  }

  /**
   * Generate pictures as data URL from an intermediate tensor.
   *
   * @param {string} modelName name of the model ('preview' or 'full').
   * @param {Tensor} intermediate an intermediate tensor.
   * @returns an array of images.
   */
  async makePictures(modelName, intermediate) {
    let pictures = null;
    let result = new Array(intermediate.shape[0]).fill('ERROR');
    try {
      pictures = tf.tidy(() => {
        let output = this.models[modelName].predict(intermediate);
        output = tf.clipByValue(output, 0, 1.0);
        // Convert to an array of tensors of shapes [H, W, 3]
        return output.unstack(0);
      });
      const canvas = new OffscreenCanvas(pictures[0].shape[1], pictures[0].shape[0]);
      for (let i = 0; i < pictures.length; ++i) {
        await tf.browser.toPixels(pictures[i], canvas);
        // Use JPEG compression as potentially more compact.
        // The performance with the default quality is better than PNG.
        // result[i] = canvas.toDataURL('image/jpg');

        result[i] = canvas.transferToImageBitmap();
      }
    }
    catch(error) {
      // this.logger.logException('Generator.makePictures', error);
      console.error(error);
      // If generation for a model failed once, it will usually always fail.
      // Remove the model to avoid repetivie failures.
      this.models[modelName].dispose();
      delete this.models[modelName];
      console.log(`Removed model '${modelName}'`);
    }
    finally {
      tf.dispose(pictures);
    }
    return result;
  }
}

class PreferenceModel {
  constructor(shape=512) {
      this.shape = shape;

      // For variance from 0 to 4
      // a, scale, std
      this.VARIANCE_PARAMS = [
        [10, 1, .02],
        [3, 1, .03],
        [1, 1, .05],    // Middle range - a uniform dist. in convex combination of training examples
        [1, 1.5, .07],  // Start going outside of the convex combination of training examples
        [1.2, 2.0, .1]  // Concentrate in the middle, as the values at the boarder have little visual difference
      ];
      this.trainingExamples = null;
  }

  init() {
    this.trainingExamples = tf.tensor([]);
    const cols = this.shape - 1;

    // Convert [0, pi] <-> [-pi, pi] for all but the last column of angles
    // (it's already so)
    this.const2 = tf.tensor(Array(cols - 1).fill(2).concat(1), [1, cols]);
    this.constPi = tf.tensor(Array(cols - 1).fill(Math.PI).concat(0), [1, cols]);
  }

  get isRandom() {
    return this.trainingExamples === null || this.trainingExamples.shape[0] === 0;
  }
  
  /**
   * Train model on given training examples.
   * @param {Tensor} trainingExamples a tensor [size, this.shape]. If size === 0 
   * (regardless of the shape), revert to uniform random generator.
   */
  train(trainingExamples) {
    tf.dispose(this.trainingExamples);
    this.trainingExamples = trainingExamples;
  }

  /**
   * Generate new pictures.
   * @param {number} size number of pictures.
   * @param {number} variance an index to select a value from VARIANCE_PARAMS.
   */
  generate(size, variance) {
    if (this.isRandom) {
      // Sample uniform random points on n-sphere.
      // See https://mathworld.wolfram.com/HyperspherePointPicking.html
      // Do not normalize the length, as we will only work with the angles.
      return tf.randomNormal([size, this.shape]);
    }
    const varianceParams = this.VARIANCE_PARAMS[variance];

    const k = this.trainingExamples.shape[0];
    let phi = cartesianToSpherical(this.trainingExamples);
    // [0, pi] -> [-pi, pi]
    phi = phi.mul(this.const2).sub(this.constPi);

    let outputPhi = null;
    if (k == 1) {
      // Only one training example, it will be varied later by a normal 'noise'.
      outputPhi = phi.broadcastTo([size, phi.shape[1]]);
    }
    else {
      // Random coefficients of shape (size, k, 1)
      const r = scaledDirichlet([size], k, varianceParams[0], varianceParams[1]).expandDims(2);
      // Sines and cosines of shape (size, k, 511)
      let sin = tf.sin(phi).broadcastTo([size, ...phi.shape]);
      let cos = tf.cos(phi).broadcastTo([size, ...phi.shape]);
      sin = sin.mul(r);
      cos = cos.mul(r);
      // Linear combinations of shape (size, 511)
      outputPhi = tf.atan2(sin.sum(1), cos.sum(1));
    }

    outputPhi = outputPhi.add(tf.randomNormal(outputPhi.shape, 0, varianceParams[2]));

    // [-pi, pi] -> [0, pi].
    outputPhi = outputPhi.add(this.constPi).div(this.const2);
    const output = sphericalToCartesian(outputPhi);
    return output;
  }
}

/**
 * Combines alogrithms to implement application logic.
 * Converts the data between UI (plain javascript data) to internal representations (tf.tensor).
 */
export class Engine {
  constructor(logger) {
    this.logger = logger;
  }

  async init() {
    // Do tf initialization here, before any usage of it.
    console.log('32-bit capable', tf.ENV.getBool('WEBGL_RENDER_FLOAT32_CAPABLE'))
    console.log('32-bit enabled', tf.ENV.getBool('WEBGL_RENDER_FLOAT32_ENABLED'))
    
    // Switch off the convolution algo conv2dWithIm2Row() with very high memory requirements.
    tf.ENV.set('WEBGL_CONV_IM2COL', false)

    if (process.env.NODE_ENV === 'production' ) {
      console.log('Production mode');
      tf.enableProdMode();
    }
    else {
      console.log('Development mode');
    }

    this.generator = new Generator(this.logger);
    this.preferenceModel = new PreferenceModel();

    await this.generator.init();
    this.preferenceModel.init();
  }

  /**
   * Create new pictures.
   * @param {number} size number of pictures.
   * @param {number} variance a number from 0 to 4 controlling the variance of the pictures.
   * @param {string[]} modelNames model names ('preview', 'full').
   * @returns {Object} a object with fields latents, preview, full.
   */
  async createPictures(size, variance, modelNames) {
    console.log(size, variance, modelNames);
    // console.log('tf.memory', tf.memory());
    let latentsTensor = null;
    try {
      const latentsTensor = tf.tidy(() => this.preferenceModel.generate(size, variance));
      return await this.generatePicturesFromTensor(latentsTensor, modelNames);
    }
    finally {
      tf.dispose(latentsTensor);
    }
  }

  /**
   * Generate pictures from latents.
   * @param {number[][]} latents array of latent vectors.
   * @param {string[]} modelNames model names ('preview', 'full').
   * @returns {Object} a object with fields latents, preview, full.
   */
  async generatePictures(latents, modelNames) {
    let latentsTensor = null;
    try {
      const latentsTensor = tf.tensor(latents);
      return await this.generatePicturesFromTensor(latentsTensor, modelNames);
    }
    finally {
      tf.dispose(latentsTensor);
    }
  }

  /**
   * Generate pictures from a tensor of latents.
   * @param {Tensor} latentsTensor
   * @param {string[]} modelNames model names ('preview', 'full').
   * @returns {Object} a object with fields latents, preview, full.
   */
  async generatePicturesFromTensor(latentsTensor, modelNames) {
    const result = await this.generator.generate(latentsTensor, modelNames);
    const latents = await latentsTensor.array();
    for(let i = 0; i < latentsTensor.shape[0]; ++i) {
      result[i].latents = latents[i];
    }
    return result;
  }

  /**
   * Learn likes.
   * @param {number[][]} likes array of liked latents.
   */
  async learn(likes) {
    // No need to dispose this tensor, it will be stored in the preference model.
    this.preferenceModel.train(tf.tensor(likes));
  }

  /**
   * Check if random pictures are generated.
   * @readonly
   */
  get isRandom() {
    return this.preferenceModel.isRandom;
  }
}

/**
 * Sample from a symmetric Dirichlet distribution of dimension k and parameter a,
 * scaled around its mean.
 *
 * @param {array} shape output shape, the output will have a shape of [shape, k]. 
 * @param {number} k dimensionality.
 * @param {number} a concentration parameter in [eps, +inf]. eps shall be > 0.01 or so to avoid nans.
 * @param {number} scale scale: scale factor.
 * @returns {Tensor} a tensor of shape [shape, k].
 */
function scaledDirichlet(shape, k, a, scale=1) {
  // Use the gamma distribution to sample from a Dirichlet distribution.
  // See https://en.wikipedia.org/wiki/Dirichlet_distribution

  const y = tf.randomGamma([...shape, k], a);
  const x = tf.div(y, y.sum(-1, true));
  const mean = 1 / k;
  const d = tf.add(tf.mul(tf.sub(x, mean), scale), mean);

  return d;
}

/**
 * Converts spherical to cartesian coordinates in n dimensions. 
 * The resulting vectors will be on a unit sphere.
 * See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates.
 *
 * @param {Tensor} phi a tensor [size, n-1] of angles.
 *  phi[0:n-2] is in [0, pi], 
 *  phi[n-1] is in [-pi, pi].
 * @returns {Tensor} a tensor [size, n] of n-dimensional unit vectors.
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
 * @param {Tensor} x a tensor [size, n] of cartesian n-dimensional vectors.
 * @returns {Tensor} a tensor [size, n-1] of angles phi, such that 
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

let engine = null;

self.addEventListener('message', async (event) => {
  console.log(event);
  if(engine === null) {
    engine = new Engine();
    await engine.init();
  }
  const pictures = await engine.createPictures(event.data.size, event.data.variance, event.data.modelNames);
  self.postMessage({pictures});
});


// Export for tests only.
export const testables = {PreferenceModel, sphericalToCartesian, cartesianToSpherical, scaledDirichlet};