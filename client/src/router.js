import Vue from 'vue';
import Router from 'vue-router';
import Images from './components/Images.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'Images',
      component: Images,
    },
  ],
});
