<template>
  <b-container>
    <div>
      <h1>Learn What You Like From Your Likes</h1>
    </div>
    <div id="stickyHeader">
      <b-button @click="learnFromLikes()" :disabled="disableLearnFromLikes" variant="primary">Learn from likes</b-button>
      <b-button @click="forgetLearning()" variant="secondary">Forget learning</b-button>
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
    </div>
    <div class="flex-container content">
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
    async getImages(count=1) {
        if(document.documentElement.scrollTop + window.innerHeight < document.documentElement.offsetHeight - 210) {
          return;
        }

        const VARIANCES = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];
        const variance = VARIANCES[this.varianceSlider];
        const images = await this.engine.getImages(count, variance);
        for(let image of images) {
          this.images.push({
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
  mounted() {
    this.images = [];
    this.engine = new Engine();
    this.getImages();
    this.pollImagesIntervalId = setInterval(() => {
        this.getImages();
      }, 1000)
  },

  beforeDestroy () {
    clearInterval(this.pollImagesIntervalId)
  },
};

window.onscroll = function() {myFunction()};

function myFunction() {
  var header = document.getElementById("stickyHeader");
  var sticky = header.offsetTop;

  if (window.pageYOffset > sticky) {
    header.classList.add("sticky");
  } else {
    header.classList.remove("sticky");
  }
}

</script>

<style scoped>

.sticky {
  position: fixed;
  top: 0;
  width: 100%;
  z-index: 100;
  /* A counter-measure for transparent background. TODO(ia): why the BG is transparent? */
  background-color: #ffffff;
}

.sticky + .content {
  margin-top: 130px;
  z-index: 10;
}

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
