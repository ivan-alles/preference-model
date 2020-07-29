<template>
  <b-container>
    <h1>Learn What You Like From Your Likes</h1>
    <b-button @click="learnFromLikes()" :disabled="disableLearnFromLikes" variant="primary">Learn from likes</b-button>
    <b-button @click="forgetLearning()" variant="secondary">Forget learning</b-button>
    <b-button v-if="pollImages" @click="togglePollImages()" variant="secondary">Pause pictures</b-button>
    <b-button v-else @click="togglePollImages()" variant="secondary">More pictures</b-button>
    <b-button @click="deleteAllImages()" variant="secondary" >Delete all pictures</b-button>
    <b-container>
        <b-row>
          <b-col sm="1">
            <label>Variance</label>
          </b-col>
          <b-col sm="3">
            <b-form-input v-model="varianceSlider" type="range" min="0" max="8"></b-form-input>
          </b-col>
        </b-row>
    </b-container>  
    <div class="flex-container">
      <div v-for="(image, index) in images" :key="index" class="image-box">
        <img :src="image.data" class="image">
          <span v-if="image.liked">
            <b-icon icon="heart-fill" @click="toggleLike(image)" class="image-button"></b-icon>
          </span>
          <span v-else>
            <b-icon icon="heart" @click="toggleLike(image)" class="image-button"></b-icon>
          </span>
      </div>
    </div>
  </b-container>
</template>

<script>
import { Engine } from '@/server-engine'

export default {
  data() {
    return {
      images: [],
      varianceSlider: 4,
      pollImages: true,
      pollImagesIntervalId: null,
    };
  },
  computed: {
    disableLearnFromLikes: function () {
        let num_likes = 0;
        for(let image of this.images) {
        if (image.liked) {
          num_likes ++;
        }
      }
      return num_likes == 0;
    }
  },  

  methods: {
    togglePollImages() {
      this.pollImages = !this.pollImages;
    },

    async getImages() {
        const VARIANCES = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];
        const variance = VARIANCES[this.varianceSlider];
        const images = await this.engine.getImages(1, variance);
        for(let image of images) {
          this.images.unshift({
            data: image.data,
            latents: image.latents,
            liked: false,
          });
        }
    },
    async learnFromLikes() {
      const likes = [];
      const liked_latents = [];
      for(let image of this.images) {
        if (image.liked) {
          likes.push(image);
          liked_latents.push(image.latents);
          image.liked = false;
        }
      }
      if (liked_latents.length === 0) {
        return;
      }
      await this.engine.learn(liked_latents);
      await this.getImages();
      for(let image of likes) {
          image.liked = false;
      }      
    },
    async forgetLearning() {
        const likes = [];
        await this.engine.learn(likes);
        await this.getImages();
    },
    deleteAllImages() {
        this.images = [];
    },    
    toggleLike(image) {
      this.image = image;
      image.liked = !image.liked;
    },
  },
  created() {
    this.images = [];
    this.engine = new Engine();
    // this.getImages();
    this.pollImagesIntervalId = setInterval(() => {
        if(this.pollImages) {
          this.getImages()
        }
      }, 1000)
  },
  beforeDestroy () {
    clearInterval(this.pollImagesIntervalId)
  },
};

</script>

<style scoped>

.flex-container {
  display: flex;
  flex-wrap: wrap;
}

.image {
  width: 200px;
  height: 200px;
}

.image-box 
{ 
  margin: 5px;
  /* For like button positioning to work. */
  position: relative; 
} 

.image-button 
{ 
  width: 40px; 
  height: 40px;
  position: absolute;
  bottom: 5px;
  left: 70px;
  width: 100%; 
  color: red;
}

.image-button:hover {
  color: red;
}

.liked {
  color: red;
}

button {
  margin: 0 0.5rem 0 0;
}

</style>
