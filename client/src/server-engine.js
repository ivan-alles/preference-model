import axios from 'axios';

class Engine {
    async getImages() {
      const path = 'http://localhost:5000/images';
      let result = await axios.get(path);
      return result.data.images;
    }

    async learn(likes) {
      const path = 'http://localhost:5000/learn';
      await axios.post(path, likes);
    }
}


export { Engine }
