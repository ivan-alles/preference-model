<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>Make Your Dream Come True</h1>
        <table class="table table-hover">
          <tbody>
            <tr v-for="(image, index) in images" :key="index">
              <td>
                <img :src="image.data">
              </td>
              <td>
                <div onclick="onLike(this)" class="fa fa-heart-o"/>
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
          this.images = res.data.images;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
  },
  created() {
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

.like {
  color: red;
}

.like:hover {
  color: red;
}
</style>
