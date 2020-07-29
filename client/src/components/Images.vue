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
        <div v-if="image.kind === cellKind.IMAGE" >
          <img :src="image.data" class="picture">
            <span v-if="image.liked">
              <b-icon icon="heart-fill" @click="toggleLike(image)" class="image-button"></b-icon>
            </span>
            <span v-else>
              <b-icon icon="heart" @click="toggleLike(image)" class="image-button"></b-icon>
            </span>
        </div>
        <div v-else-if="image.kind === cellKind.LIKES">
          Likes
          <div v-for="(picture, index) in image.pictures" :key="index" >
            <img :src="picture" class="like-picture">
          </div>
        </div>        
      </div>
    </div>
  </b-container>
</template>

<script>
import { Engine } from '@/server-engine'

const cellKind = {
    IMAGE: 'image',
    LIKES: 'likes',
    RANDOM: 'random'
}

export default {
  data() {
    return {
      images: [],
      varianceSlider: 4,
      pollImagesIntervalId: null,
      cellKind, // Make this enum accessible in Vue code
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
            kind: cellKind.IMAGE,
            data: image.data,
            latents: image.latents,
            liked: false,
          });
        }
    },
    async learnFromLikes() {
      const likes = [];
      for(let image of this.images) {
        if (image.kind === cellKind.IMAGE && image.liked) {
          likes.push(image);
        }
      }
      if (likes.length === 0) {
        return;
      }
      const liked_latents = [];
      const liked_pictures = [];

      for(let like of likes) {
          like.liked = false;
          liked_latents.push(like.latents);
          liked_pictures.push(like.data)
      }
      this.images.push({
          kind: cellKind.LIKES,
          pictures: liked_pictures
      });

      await this.engine.learn(liked_latents);
      await this.getImages();
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

.image-box 
{ 
  width: 200px;
  height: 200px;
  margin: 5px;
  /* For like button positioning to work. */
  position: relative; 
} 

.picture {
    height: 100%;
    width: 100%; 
    object-fit: contain;
}

.like-picture {
    height: 40px;
    width: 40px; 
    object-fit: contain;
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
