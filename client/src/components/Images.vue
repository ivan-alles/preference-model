<template>
  <b-container>
    <div>
      <h1>Learn What You Like From Your Likes</h1>
    </div>
    <div id="stickyHeader">
      <b-button @click="learnFromLikes()" :disabled="findLikes().length == 0" variant="primary">Learn from likes</b-button>
      <b-button @click="forgetLearning()" variant="secondary">Forget learning</b-button>
      <b-button @click="deleteAllPictures()" variant="secondary" >Delete all pictures</b-button>
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
      <div v-for="(cell, index) in cells" :key="index" class="cell">
        <div v-if="cell.kind === cellKind.PICTURE" >
          <img :src="cell.picture" class="picture">
            <span v-if="cell.liked">
              <b-icon icon="heart-fill" @click="toggleLike(cell)" class="like-button"></b-icon>
            </span>
            <span v-else>
              <b-icon icon="heart" @click="toggleLike(cell)" class="like-button"></b-icon>
            </span>
        </div>
        <div v-else-if="cell.kind === cellKind.LIKES">
          Learning from
          <div v-for="(picture, index) in cell.pictures" :key="index" class="likes-picture-div">
            <img :src="picture" class="likes-picture">
          </div>
        </div>   
        <div v-else-if="cell.kind === cellKind.RANDOM">
          Random
        </div>                
      </div>
    </div>
  </b-container>
</template>

<script>
import { Engine } from '@/server-engine'

const cellKind = {
    PICTURE: 'picture',
    LIKES: 'likes',
    RANDOM: 'random'
}

export default {
  data() {
    return {
      cells: [],
      varianceSlider: 4,
      pollPicturesIntervalId: null,
      cellKind, // Make this enum accessible in Vue code
    };
  },
  computed: {
  },  

  methods: {
    findLikes() {
      let likes = this.cells.filter(cell => cell.kind === cellKind.PICTURE && cell.liked);
      return likes;
    },

    async getPictures(count=1) {
        if(document.documentElement.scrollTop + window.innerHeight < document.documentElement.offsetHeight - 210) {
          return;
        }
        const VARIANCES = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16];
        const variance = VARIANCES[this.varianceSlider];
        const enginePictures = await this.engine.getPictures(count, variance);
        for(let enginePicture of enginePictures) {
          this.cells.push({
            kind: cellKind.PICTURE,
            picture: enginePicture.picture,
            latents: enginePicture.latents,
            liked: false,
          });
        }
    },

    async learnFromLikes() {
      const likes = this.findLikes();

      if (likes.length === 0) {
        return;
      }

      const liked_latents = [];
      const liked_pictures = [];

      for(let like of likes) {
          like.liked = false;
          liked_latents.push(like.latents);
          liked_pictures.push(like.picture)
      }

      this.cells.push({
          kind: cellKind.LIKES,
          pictures: liked_pictures
      });

      await this.engine.learn(liked_latents);
      await this.getPictures();
    },
    async forgetLearning() {
        this.cells.push({
          kind: cellKind.RANDOM
        });
        await this.engine.learn([]);
        await this.getPictures();
    },

    deleteAllPictures() {
        this.cells = [];
    },    
    toggleLike(cell) {
      cell.liked = !cell.liked;
    },
  },
  created() {
    this.engine = new Engine();
    this.cells = [];
    // TODO(ia): this is not always true, as the server engine has its state.
    // this.cells.push({
    //   kind: cellKind.RANDOM
    // });    
  },

  mounted() {
    this.pollPicturesIntervalId = setInterval(() => {
        this.getPictures();
      }, 1000)
  },

  beforeDestroy () {
    clearInterval(this.pollPicturesIntervalId)
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

.cell 
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

.likes-picture-div {
    display: table-cell;
}

.likes-picture {
    max-width: 100%;
}

.like-button 
{ 
  width: 40px; 
  height: 40px;
  position: absolute;
  bottom: 5px;
  left: 70px;
  width: 100%; 
  color: red;
}

.like-button:hover {
  color: red;
}

.liked {
  color: red;
}

button {
  margin: 0 0.5rem 0 0;
}

</style>
