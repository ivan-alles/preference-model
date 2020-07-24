<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>Learn What You Like From Your Likes</h1>
        <button @click="getImages()" type="button" class="btn btn-primary">More pictures</button>
        <button @click="learn()" type="button" class="btn btn-primary">Learn from likes</button>
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
import axios from 'axios';

export default {
  data() {
    return {
      images: [],
    };
  },

  methods: {
    getImages() {
      const path = 'http://localhost:5000/images';
      axios.get(path)
        .then((res) => {
          res.data.images.forEach((item) => {
            const image = {
              data: item.data,
              latents: item.latents,
              liked: false,
            };
            this.images.unshift(image);
          });
        })
        .catch((error) => {
          console.error(error);
        });
    },
    learn() {
      const path = 'http://localhost:5000/learn';
      const likes = [];
      this.images.forEach((item) => {
        if (item.liked) {
          likes.push(item.latents);
          item.liked = false;
        }
      });
      if (likes.length === 0) {
        return;
      }
      axios.post(path, likes)
        .then(() => {
          this.getImages();
        })
        .catch((error) => {
          console.error(error);
        });
    },
    toggleLike(image) {
      this.image = image;
      image.liked = !image.liked;
    },
  },
  created() {
    this.images = [];
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
