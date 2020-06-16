import Vue from "vue";
import Router from "vue-router";

import RecipeMain from "../view/RecipeMain";
import RecipeList from "../view/RecipeList";
import RecipeDetail from "../view/RecipeDetail";

Vue.use(Router);

export default new Router({
  mode: "history",
  routes: [
    {
      path: "/",
      name: "RecipeMain",
      component: RecipeMain
    },
    {
      path: "/RecipeList",
      name: "RecipeList",
      component: RecipeList
    },
    {
      path: "/RecipeDetail/:id",
      name: "RecipeDetail",
      component: RecipeDetail
    }
  ],
  scrollBehavior(to, from, savedPosition) {
    return { x: 0, y: 0 };
  }
});
