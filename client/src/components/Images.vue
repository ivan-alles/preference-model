<!--
  Error handling: the practice have shown that in case of an error we cannot recover. Only reloading the page helps.
  This is what we implement in the code.
-->
<template>
  <b-container>
    <div>
      <h1>Make a Person of Your Dreams</h1>
    </div>
    <template v-if="state === stateKind.WORKING">
      <template v-if="fullPicture === null">
        <div id="stickyHeader">
          <span id="learn-wrapper" class="d-inline-block" tabindex="0">
            <b-button  @click="triggerLearning()" :disabled="! isLearningEnabled" variant="primary">
              <b-icon icon="heart"></b-icon>
              Learn
            </b-button>
          </span>
          <b-tooltip target="learn-wrapper" :delay="{ show: 500, hide: 50 }">
            <template v-if="isLearningEnabled">
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
            </b-row>
          </b-container>  
        </div>
        <div class="flex-container content">
          <div v-for="(cell, index) in cells" :key="index" class="cell">
            <template v-if="cell.kind === Picture.kind.PICTURE" >
              <img :src="cell.picture" class="picture" @click="showFullPicture(cell)">
              <span v-if="cell.liked">
                <b-icon icon="heart-fill" @click="toggleLike(cell)" class="like-button liked"></b-icon>
              </span>
              <span v-else>
                <b-icon icon="heart" @click="toggleLike(cell)" class="like-button"></b-icon>
              </span>
            </template>
            <template v-else-if="cell.kind === Picture.kind.LIKES">
              <h4>
                <b-icon icon="heart" @click="relike(cell)"></b-icon>
                Likes
              </h4>
              <div class="likes-picture-row" @click="relike(cell)">
                <div v-for="(picture, index) in cell.pictures" :key="index" class="likes-picture-col">
                  <img :src="picture" class="likes-picture">
                </div>
              </div>
            </template>   
            <template v-else-if="cell.kind === Picture.kind.IN_PROGRESS">
              <h4>
                <b-spinner variant="secondary" label="Dreaming"></b-spinner>
                Dreaming
              </h4>
            </template>            
            <template v-else-if="cell.kind === Picture.kind.RANDOM">
              <h4>
                <b-icon icon="dice6" ></b-icon>
                Random
              </h4>
            </template>   
          </div>
        </div>
      </template>
      <template v-else>
        <b-button @click="closeFullPicture()" variant="primary">
          <b-icon icon="arrow-left-short" ></b-icon>
          Continue
        </b-button>
        <ShareNetwork
            network="VK"
            :url="shareUrl()"
            :title="shareTitle()"
          >
          <b-button variant="secondary">
            <font-awesome-icon :icon="['fab', 'vk']" size="lg" ></font-awesome-icon>
          </b-button>
        </ShareNetwork>
        <img :src="fullPicture.picture" class="full-picture">
      </template>
    </template>
    <template v-if="state === stateKind.INIT">
      <h4>
        <b-spinner variant="secondary" label="Loading"></b-spinner>
        Loading
      </h4>
    </template>
    <template v-if="state === stateKind.ERROR">
      <h4 class="error">
        <b-icon icon="exclamation-circle-fill" variant="danger"></b-icon>
        Error
      </h4>
      <p>
      This app works on a desktop with an NVIDIA graphic card. Other devices may not be supported.
      </p>
      <b-button @click="reload()" variant="primary">
        <b-icon icon="bootstrap-reboot"></b-icon>
          Reload
        </b-button>
    </template>

  </b-container>
</template>

<script>

// import { Engine } from '@/server-engine'
import { Engine } from '@/client-engine'
import { float32ArrayToBase64, base64ToFloat32Array } from '@/utils'

class Picture {

  static kind = {
    PICTURE: 'PICTURE',
    IN_PROGRESS: 'IN_PROGRESS',
    LIKES: 'LIKES',
    RANDOM: 'RANDOM',
  }
}

console.log('Picture', Picture.kind)

const stateKind = {
    INIT: 'INIT',       // Loading models, etc.
    WORKING: 'WORKING', // Generating pictures
    EXIT: 'EXIT',       // App finished.
    ERROR: 'ERROR',     // Fatal error, cannot work.
}

class GoogleAnalyticsLogger {
  constructor(ga) {
    this.ga = ga;
  }

  log(category, action, label, value=1) {
    console.log(category, action, label, value);
    this.ga.event({
      eventCategory: category,
      eventAction: action,
      eventLabel: label,
      eventValue: value
    });
  }

  logException(action, exception, value=1) {
    console.error(exception);
    this.ga.event({
      eventCategory: 'LogError',
      eventAction: action,
      eventLabel: exception.stack,
      eventValue: value
    });
  }
}

export default {
  data() {
    return {
      state: stateKind.INIT,
      cells: [],
      varianceSlider: 2,
      fullPicture: null,
    };
  },
  computed: {
    findLikes() {
      let likes = this.cells.filter(cell => cell.kind === Picture.kind.PICTURE && cell.liked);
      return likes;
    },

    isLearningEnabled() {
      return this.findLikes.length != 0;
    },
  },  

  methods: {

    isRandom: function() {
      return this.engine.isRandom;
    },

    /**
    * Generates pictures in the background.
    */
    async getPicturesTask() {
      try {
        await this.engine.init();

        if('show' in this.$route.query) {
          const showParam = decodeURIComponent(this.$route.query['show']);
          console.log(showParam);
          const latents = base64ToFloat32Array(showParam);
          const enginePictures = await this.engine.generatePictures([latents], 'full');
          this.cells.push({
            picture: enginePictures[0].picture,
            latents: enginePictures[0].latents,
            liked: false,
            kind: Picture.kind.PICTURE
          });
          this.fullPicture = {
            picture: enginePictures[0].picture,
            latents: enginePictures[0].latents
          }
        }       

        this.state = stateKind.WORKING;
      }
      catch(err) {
        console.error(err);
        this.state = stateKind.ERROR;
        return;
      }

      while(this.state != stateKind.EXIT) {
        await sleep(50);

        if(!this.isActive || this.fullPicture !== null) {
          continue;
        }
        
        // Are images below the bottom of the screen?
        if(document.documentElement.scrollTop + window.innerHeight < document.documentElement.offsetHeight - 210) {
          continue;
        }

        try {
          if(this.isLearningTriggered) {
            this.isLearningTriggered = false;
            await this.learn();
          }
          else if(this.isRandomTriggered) {
            this.isRandomTriggered = false;
            await this.random();
          }  
        }
        catch(err) {
          console.error(err);
          this.state = stateKind.ERROR;
          return;
        }        

        const size = 1;
        let newCells = [];
        for(let i = 0; i < size; ++i) {
          const cell = { kind: Picture.kind.IN_PROGRESS };
          newCells.push(cell);
          this.cells.push(cell);
        }

        try {
          const enginePictures = await this.engine.createPictures(size, this.varianceSlider, 'preview');
          for(let i = 0; i < size; ++i) {
            newCells[i].picture = enginePictures[i].picture;
            newCells[i].latents = enginePictures[i].latents;
            newCells[i].liked = false;
            newCells[i].kind = Picture.kind.PICTURE;
          }
        }
        catch(err) {
          this.state = stateKind.ERROR;
          this.logger.logException('Images.getPicturesTask.createPictures', err);
          return;
        }
      }
    },

    triggerLearning() {
      this.isLearningTriggered = true;
    },

    triggerRandom() {
      this.isRandomTriggered = true;
    },

    reload() {
      location.reload();
    },

    async learn() {
      const likes = this.findLikes;

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
          kind: Picture.kind.LIKES,
          likes: likes,
          pictures: pictures
      });

      await this.engine.learn(latents);
    },

    async random() {
        this.cells.push({
          kind: Picture.kind.RANDOM
        });
        await this.engine.learn([]);
    },

    deleteAllPictures() {
        this.cells = [];
    },    

    toggleLike(cell) {
      cell.liked = !cell.liked;
    },

    showFullPicture(cell) {
      this.fullPicture = {
        picture: cell.picture,
        latents: cell.latents,
      }
    },

    closeFullPicture() {
      this.fullPicture = null;
    },

    relike(cell) {
      for(let like of cell.likes) {
          like.liked = true;
      }
    },

    shareUrl() {
      const url = window.location.href + '?show=' + 
        encodeURIComponent(float32ArrayToBase64(this.fullPicture.latents));
      return url;
    },

    shareTitle() {
      return "Make a Person of Your Dreams";
    },
  },

  created() {
    // Make globals accessible in Vue rendering code
    this.stateKind = stateKind;
    this.Picture = Picture;

    this.logger = new GoogleAnalyticsLogger(this.$ga);
    this.isActive = true;
    this.isRandomTriggered = true;
    this.isLearningTriggered = false;
    this.engine = new Engine(this.logger);
  },

  mounted() {
    this.getPicturesTask();
  },

  beforeDestroy () {
    this.state = stateKind.EXIT;
  },

  watch: {
    $route(to, from) { // eslint-disable-line
      // Activate this component when the router points to it.
      this.isActive = to.name === "Home";
    }
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
  /* A counter-measure against the default transparent background. */
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
  border-radius: 4px;
  box-shadow: 2px 2px 4px #0004;
} 

.cell h4 { 
  margin-top: 5px;
  margin-bottom: 10px;
}

.picture {
    height: 100%;
    width: 100%; 
    object-fit: contain;
    border-radius: 4px;
    cursor: zoom-in;
}

.likes-picture-row {
    display: flex;
    justify-content: center;
}

.likes-picture-col {
    /* Set to a value in [0.5, 1) for 1-picture case to fit the container. */
    flex: 0.65;
}

.likes-picture {
    width: 100%;
}

.like-button 
{ 
  width: 40px; 
  height: 40px;
  position: absolute;
  bottom: 5px;
  left: 70px;
  width: 100%; 
  color: #f97878;
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

.full-picture {
  border-radius: 4px;
  box-shadow: 2px 2px 4px #0004;
  margin-top: 10px;  
}

.error {
  color: var(--danger);
}

</style>
