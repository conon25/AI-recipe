<template>
  <div class="recipe-list-container">
    <div v-if="recipeInfoArr.length !== 0">
      <v-row no-gutters>
        <v-col v-for="recipeInfo in recipeInfoArr" :key="recipeInfo.id" cols="12">
          <RecipeCard :recipeInfo="recipeInfo" />
        </v-col>
      </v-row>
    </div>
    <div v-else>
      <RecipeNotFound />
    </div>
  </div>
</template>

<script>
import http from "../services/http-common";
import RecipeCard from "../components/Recipe/List/RecipeCard";
import RecipeNotFound from "../components/Recipe/RecipeNotFound";
export default {
  name: "RecipeList",
  components: {
    RecipeCard,
    RecipeNotFound
  },
  computed: {
    recipeInfoArr() {
      return this.$store.state.data;
    }
  },
  mounted: () => {
    // document.querySelector('header').style.background = 'red'
    // document.querySelector('header').classList.remove('test-red')
    // document.querySelector('header').classList.add('test-black')
    // console.log(this.recipeInfoArr);
  }
};
</script>

<style scoped>
.recipe-list-container {
  width: 100%;
  height: 100vh;
  padding-top: 70px;
}
</style>
