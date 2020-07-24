<template>
  <div class="home">
    <Images msg="Welcome to Your Vue.js App"/>
  </div>
</template>

<script>
// @ is an alias to /src
import Images from '@/components/Images.vue'

export default {
  name: 'Home',
  components: {
    Images
  }
}
</script>
