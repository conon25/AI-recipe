import Vue from "vue";
import App from "./App.vue";
import router from "./router/index";
import vuetify from "./plugins/vuetify";
import store from "./store";
import "./registerServiceWorker";

Vue.config.productionTip = false;
new Vue({
  render: h => h(App),
  vuetify,
  router,
  store
}).$mount("#app");
