import axios from 'axios';

const PICTURES_URL = 'http://localhost:5000/images';
const LEARN_URL = 'http://localhost:5000/learn';

class Engine {
    constructor () {
      this.firstCall = true;
      this.isRandom = true;
    }

    async init() {
      // Nothing to do.
    }

    async getPictures(size, variance) {
      if(this.firstCall) {
        this.firstCall = false;
        // reset to random pictures.
        await axios.post(LEARN_URL, []);
      }

      let result = await axios.get(PICTURES_URL, {
          params: {
              size: size,
              variance: variance
            }
          }
        );
      return result.data.images;
    }

    async learn(likes) {
      this.isRandom = likes.length === 0;
      await axios.post(LEARN_URL, likes);
    }

}


export { Engine }
