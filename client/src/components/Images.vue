<template>
  <b-container>
    <div>
      <h1>Make a Person of Your Dreams</h1>
    </div>
    <div id="stickyHeader">
      <span id="learn-wrapper" class="d-inline-block" tabindex="0">
        <b-button  @click="triggerLearning()" :disabled="! isLearningEnabled()" variant="primary">
          <b-icon icon="heart"></b-icon>
          Learn
        </b-button>
      </span>
      <b-tooltip target="learn-wrapper" :delay="{ show: 500, hide: 50 }">
        <template v-if="isLearningEnabled()">
          Learn from likes
        </template>
        <template v-else>
          Like some pictures to learn from them
        </template>      
      </b-tooltip>
      <span id="random-wrapper" class="d-inline-block" tabindex="0">
        <b-button @click="triggerRandom()" variant="secondary" :disabled="isRandom()">
          <b-icon icon="dice6" ></b-icon>
          Random
        </b-button>
      </span>
      <b-tooltip target="random-wrapper" :delay="{ show: 500, hide: 50 }">
        <template v-if="! isRandom()">
          Forget learning and make random pictures
        </template>
        <template v-else>
          Already making random pictures
        </template>      
      </b-tooltip>      
      <b-button id="delete-all-button" @click="deleteAllPictures()" variant="secondary">
        <b-icon icon="trash" ></b-icon>
        Delete all
      </b-button>
      <b-tooltip target="delete-all-button" :delay="{ show: 500, hide: 50 }">
          Delete all pictures
      </b-tooltip>
      <b-container>
        <b-row>
          <b-col sm="1">
            <label>Variance</label>
          </b-col>
          <b-col sm="3" id="variance-slider">
            <b-form-input v-model="varianceSlider" type="range" min="0" max="4" :disabled="isRandom()"></b-form-input>
          </b-col>
          <!-- This does not work well, the tooltip remains visible all the time.
          TODO(ia): try again after layout rework, delete if still not working.
          <b-tooltip target="variance-slider" delay="{ show: 1000, hide: 50 }">
            <template v-if="! isRandom()">
              Vary pictures less or more
            </template>
            <template v-else>
              Cannot vary random pictures, like some pictures and learn first
            </template>      
          </b-tooltip>           
          -->
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
        <div v-else-if="cell.kind === cellKind.LIKES" @click="relike(cell)">
          <h4>
            <b-icon icon="dice6" ></b-icon>
            Likes
          </h4>
          <div v-for="(picture, index) in cell.pictures" :key="index" class="likes-picture-div">
            <img :src="picture" class="likes-picture">
          </div>
        </div>   
        <div v-else-if="cell.kind === cellKind.RANDOM">
          <h4>
            <b-icon icon="dice6" ></b-icon>
            Random
          </h4>
        </div>                
      </div>
    </div>
  </b-container>
</template>

<script>

// import { Engine } from '@/server-engine'
import { Engine } from '@/client-engine'

const cellKind = {
    PICTURE: 'picture',
    LIKES: 'likes',
    RANDOM: 'random'
}

export default {
  data() {
    return {
      cells: [],
      varianceSlider: 2,
      pollPicturesIntervalId: null,
      cellKind, // Make this enum accessible in Vue code
    };
  },
  computed: {
  },  

  methods: {
    isRandom: function() {
      return this.engine.isRandom;
    },

    findLikes() {
      let likes = this.cells.filter(cell => cell.kind === cellKind.PICTURE && cell.liked);
      return likes;
    },

    isLearningEnabled() {
      // TODO(ia): shall this be a computed property?
      return this.findLikes().length != 0;
    },

    /**
    * Generates pictures in the background.
    */
    async getPicturesTask() {
        await this.engine.init();
        for(;;) {
          try {
            if(document.documentElement.scrollTop + window.innerHeight < document.documentElement.offsetHeight - 210) {
              await sleep(50);
              continue;
            }
            if(this.isLearningTriggered) {
              this.isLearningTriggered = false;
              await this.learn();
            }
            if(this.isRandomTriggered) {
              this.isRandomTriggered = false;
              await this.random();
            }            
            const enginePictures = await this.engine.getPictures(1, this.varianceSlider);
            for(let enginePicture of enginePictures) {
              this.cells.push({
                kind: cellKind.PICTURE,
                picture: enginePicture.picture,
                latents: enginePicture.latents,
                liked: false,
              });
            }
            await sleep(200);
          }
          catch(err) {
            console.error(err, err.stack);
          }
        }
    },

    triggerLearning() {
      this.isLearningTriggered = true;
    },

    triggerRandom() {
      this.isRandomTriggered = true;
    },

    async learn() {
      const likes = this.findLikes();

      if (likes.length === 0) {
        return;
      }

      const latents = [];
      const pictures = [];

      for(let like of likes) {
          like.liked = false;
          latents.push(like.latents);
          pictures.push(like.picture)
      }

      this.cells.push({
          kind: cellKind.LIKES,
          likes: likes,
          pictures: pictures // TODO(ia): pictures are redundant, we can take them from likes.
      });

      await this.engine.learn(latents);
    },

    async random() {
        this.cells.push({
          kind: cellKind.RANDOM
        });
        await this.engine.learn([]);
    },

    deleteAllPictures() {
        this.cells = [];
    },    

    toggleLike(cell) {
      cell.liked = !cell.liked;
    },

    relike(cell) {
      for(let like of cell.likes) {
          like.liked = true;
      }
    }
  },
  created() {
    this.cells = [];
    this.isRandomTriggered = true;
    this.isLearningTriggered = false;
    this.engine = new Engine();
  },

  mounted() {
    this.getPicturesTask();
  },

  beforeDestroy () {
    clearInterval(this.pollPicturesIntervalId)
  },
};

window.onscroll = function() {stickyHeader()};

function stickyHeader() {
  var header = document.getElementById("stickyHeader");
  var sticky = header.offsetTop;

  if (window.pageYOffset > sticky) {
    header.classList.add("sticky");
  } else {
    header.classList.remove("sticky");
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
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

.cell { 
  width: 200px;
  height: 200px;
  margin: 5px;
  text-align: center;
  /* For like button positioning to work. */
  position: relative; 
  border: 1px solid var(--secondary);
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
