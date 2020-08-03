// TODO(ia): temp code begin.
import axios from 'axios';
const PICTURES_URL = 'http://localhost:5000/images';
const LEARN_URL = 'http://localhost:5000/learn';
// TODO(ia): temp code end.

class Generator {
  constructor() {
    self.model = null;
  }

  generate(latents) {
    let pictures = [latents];
    return pictures;
  }
}

class PreferenceModel {
  constructor(shape=512) {
      self.shape = shape;
      // Learnable parameters
      self.trainingExamples = [];
  }

  isRandom() {
      return this.trainingExamples.length == 0;
  }

  train(self, trainingExamples) {
    self.trainingExamples = trainingExamples;
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

    // TODO(ia): temp code begin.
    this.firstCall = true;
    this.isRandom = true;
    // TODO(ia): temp code end.
  }

  async getPictures(count, variance) {

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
    // TODO(ia): temp code begin.
    this.isRandom = likes.length === 0;
    await axios.post(LEARN_URL, likes);  }
    // TODO(ia): temp code end.
}


export { Engine }