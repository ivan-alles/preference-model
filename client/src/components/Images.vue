<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>Preference Model</h1>
        <table class="table table-hover">
          <thead>
            <tr>
              <th scope="col">Image</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(image, index) in images" :key="index">
              <td>
                <img :src="image.data" width="200px" height="200px">
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
