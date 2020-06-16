import Vue from "vue";
import Vuex from "vuex";

Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    data: null
  },
  mutations: {
    setValue(state, recipedata) {
      state.data = recipedata.recipeInfoArr;
    }
  },
  actions: {
    recipeInfo(context, recipedata) {
      context.commit("setValue", recipedata);
    }
  }
});
