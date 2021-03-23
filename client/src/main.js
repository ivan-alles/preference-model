// Copyright 2016-2020 Ivan Alles. See also the LICENSE file. 

import Vue from 'vue'
import App from './App.vue'
import router from './router'
import VueGtag from 'vue-gtag'


Vue.config.productionTip = false

Vue.use(VueGtag, {
  config: { id: 'G-M6ZS8H863N' }
})

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
