<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>Learn What You Like From Your Likes</h1>
        <button @click="getImages()" type="button">More pictures</button>
        <button @click="learn()" type="button">Learn from likes</button>
        <button @click="resetLearning()" type="button">Reset learning</button>
        <table class="table table-hover">
          <tbody>
            <tr v-for="(image, index) in images" :key="index">
              <td>
                <div class="image-box">
                  <img :src="image.data" class="image">
                  <span v-if="image.liked">
                    <div class="fa fa-heart liked image-button" @click="toggleLike(image)"></div>
                  </span>
                  <span v-else>
                    <div class="fa fa-heart-o image-button" @click="toggleLike(image)"></div>
                  </span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script>
import { Engine } from '@/server-engine'

export default {
  data() {
    return {
      images: [],
    };
  },

  methods: {
    async getImages() {
        const images = await this.engine.getImages();
        for(let image of images) {
          this.images.unshift({
            data: image.data,
            latents: image.latents,
            liked: false,
          });
        }
    },
    async learn() {
      const likes = [];
      for(let image of this.images) {
        if (image.liked) {
          likes.push(image.latents);
          image.liked = false;
        }
      }
      if (likes.length === 0) {
        return;
      }
      await this.engine.learn(likes);
      await this.getImages();
    },
    async resetLearning() {
        const likes = [];
        await this.engine.learn(likes);
        await this.getImages();
    },
    toggleLike(image) {
      this.image = image;
      image.liked = !image.liked;
    },
  },
  created() {
    this.images = [];
    this.engine = new Engine();
    this.getImages();
  },
};

</script>

<style scoped>
.image {
  width: 200px;
  height: 200px;
}

.image-box 
{ 
  position: relative; /* To help the image + text element to get along with the rest of the page*/ 
} 

.image-button 
{ 
  position: absolute;
  bottom: 5px;
  left: 5px;
  width: 100%; 
}

.fa {
  font-size: 50px;
  cursor: pointer;
  user-select: none;
  color: red;
}

.fa:hover {
  color: red;
}

.liked {
  color: red;
}

.liked:hover {
  color: red;
}

button {
  margin: 0 0.5rem 0 0;
}

</style>
