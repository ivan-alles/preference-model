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
          <div v-for="(picture, index) in pictures" :key="index" class="picture">
            <template v-if="picture.kind === null" >
              <template v-if="picture.preview !== null">
                <img :src="picture.preview" class="preview-picture" @click="showFullPicture(picture)">
                <span v-if="picture.liked">
                  <b-icon icon="heart-fill" @click="toggleLike(picture)" class="like-button liked"></b-icon>
                </span>
                <span v-else>
                  <b-icon icon="heart" @click="toggleLike(picture)" class="like-button"></b-icon>
                </span>
              </template>
              <template v-else>
                <h4>
                  <b-spinner variant="secondary" label="Dreaming"></b-spinner>
                  Dreaming
                </h4>
              </template>               
            </template>
            <template v-else-if="picture.kind === Picture.kind.LIKES">
              <h4>
                <b-icon icon="heart" @click="relike(picture)"></b-icon>
                Likes
              </h4>
              <div class="likes-picture-row" @click="relike(picture)">
                <div v-for="(picture, index) in picture.deprecatedLikedPictures" :key="index" class="likes-picture-col">
                  <img :src="picture" class="likes-picture">
                </div>
              </div>
            </template>             
            <template v-else-if="picture.kind === Picture.kind.RANDOM">
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
        <template v-if="fullPicture.full !== null">
          <img :src="fullPicture.full" class="full-picture">
        </template>
        <template v-else>
          <img :src="fullPicture.preview" class="full-picture">
        </template>
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
  kind; // TODO(ia): remove this
  latents;
  preview;
  full = null;
  liked = false;

  constructor(latents, preview, kind=null) {
    this.latents = latents;
    this.preview = preview;
    this.kind = kind;
  }

  // TODO(ia): remove this
  static kind = {
    LIKES: 'LIKES',
    RANDOM: 'RANDOM',
  }
}

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
      pictures: [],
      varianceSlider: 2,
      fullPicture: null,
    };
  },
  computed: {
    findLikes() {
      let likes = this.pictures.filter(picture => picture.kind === null && picture.liked);
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
          const enginePictures = await this.engine.generatePictures([latents], ['preview', 'full']);
          const picture = new Picture(enginePictures[0]);
          this.pictures.push(picture);
          this.fullPicture = picture;
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

        if(!this.isActive) {
          continue;
        }

        if(this.fullPicture !== null) {
          if(this.fullPicture.full === null) {
            const enginePictures = await this.engine.generatePictures([this.fullPicture.latents], ['full']);
            this.fullPicture.full = enginePictures[0].full;
          }
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
        let newPictures = [];
        for(let i = 0; i < size; ++i) {
          const picture = new Picture(null, null);
          newPictures.push(picture);
          this.pictures.push(picture);
        }

        try {
          const enginePictures = await this.engine.createPictures(size, this.varianceSlider, ['preview']);
          for(let i = 0; i < size; ++i) {
            newPictures[i].preview = enginePictures[i].preview;
            newPictures[i].latents = enginePictures[i].latents;
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
          pictures.push(like.preview)
      }

      const likesPicture = new Picture(null, null, Picture.kind.LIKES);
      likesPicture.likes = likes;
      likesPicture.deprecatedLikedPictures = pictures;
      this.pictures.push(likesPicture);
      
      await this.engine.learn(latents);
    },

    async random() {
        this.pictures.push(new Picture(null, null, Picture.kind.RANDOM));
        await this.engine.learn([]);
    },

    deleteAllPictures() {
        this.pictures = [];
    },    

    toggleLike(picture) {
      picture.liked = !picture.liked;
    },

    showFullPicture(picture) {
      this.fullPicture = picture;
    },

    closeFullPicture() {
      this.fullPicture = null;
    },

    relike(picture) {
      for(let like of picture.likes) {
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

.picture { 
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

.picture h4 { 
  margin-top: 5px;
  margin-bottom: 10px;
}

.preview-picture {
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
  width: 100%; 
  height: 100%; 
}

.error {
  color: var(--danger);
}

</style>
