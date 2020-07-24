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
                <img :src="image.data">
              </td>
              <td v-if="image.liked">
                <div class="fa fa-heart liked" @click="toggleLike(image)"/>
              </td>
              <td v-else>
                <div class="fa fa-heart-o" @click="toggleLike(image)"/>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script>
import { Backend } from '@/server-backend.js'

export default {
  data() {
    return {
      images: [],
    };
  },

  methods: {
    async getImages() {
        const images = await this.backend.getImages();
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
      await this.backend.learn(likes);
      await this.getImages();
    },
    async resetLearning() {
        const likes = [];
        await this.backend.learn(likes);
        await this.getImages();
    },
    toggleLike(image) {
      this.image = image;
      image.liked = !image.liked;
    },
  },
  created() {
    this.images = [];
    this.backend = new Backend();
    this.getImages();
  },
};

</script>

<style scoped>
img {
  width: 200px;
  height: 200px;
}

.fa {
  font-size: 50px;
  cursor: pointer;
  user-select: none;
  color: gray;
}

.fa:hover {
  color: darkgray;
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
