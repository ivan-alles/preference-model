import axios from 'axios';

const PICTURES_URL = 'http://localhost:5000/images';
const LEARN_URL = 'http://localhost:5000/learn';

class Engine {
    constructor () {
      this.firstCall = true;
      this.isRandom = true;
    }

    async getPictures(count, variance) {
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
    }

    async learn(likes) {
      this.isRandom = likes.length === 0;
      console.log('learn');
      console.log(this.isRandom);
      await axios.post(LEARN_URL, likes);
    }

}


export { Engine }
