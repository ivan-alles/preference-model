
class Generator {
  constructor() {
    self.model = null;
  }

  generate(latents) {
    let pictures = [latents];
    return pictures;
  }
}

class Engine {
  constructor () {
    this.generator = new Generator();
  }

  async getPictures(count, variance) {
    return count + variance;
  }

  async learn(likes) {
    return likes;
  }

}


export { Engine }