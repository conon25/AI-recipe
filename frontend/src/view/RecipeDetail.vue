<template>
  <div class="recipe-detail-container">
    <div :style="{ backgroundImage: `url(${imagePath})` }" class="thumbnail"></div>
    <RecipeMaterial :recipeMaterial="recipeMaterial" />
    <RecipeSequence :recipeSequence="recipeSequence" />
  </div>
</template>
<script>
import http from "../services/http-common.js";
import RecipeMaterial from "../components/Recipe/Meterial/RecipeMaterial";
import RecipeSequence from "../components/Recipe/Sequence/RecipeSequence";

export default {
  name: "RecipeDetail",
  components: {
    RecipeMaterial,
    RecipeSequence
  },
  data: () => ({
    recipeMaterial: null,
    recipeSequence: null,
    imagePath: null
  }),
  methods: {
    getRecipeSequence() {
      http.get(`/recipes/processinfo/${this.$route.params.id}`).then(res => {
        this.recipeSequence = res.data;
      });
    },

    getRecipeMaterial() {
      http.get(`/recipes/materialinfo/${this.$route.params.id}`).then(res => {
        this.recipeMaterial = res.data;
      });
    },

    getImgPath() {
      http.get(`/recipes/basicinfo/${this.$route.params.id}`).then(res => {
        this.imagePath = res.data[0].basic_imgurl;
      });
    }
  },
  beforeMount() {
    console.log(this.$route.params.id);
    this.getRecipeSequence();
    this.getRecipeMaterial();
    this.getImgPath();
  }
};
</script>

<style>
.recipe-detail-container {
  width: 100%;
  height: 100vh;
  padding: 70px 0 0 0;
}
.thumbnail {
  width: 100%;
  height: 50%;
  background-repeat: no-repeat;
  background-size: cover;
  background-position: center;
}
</style>
