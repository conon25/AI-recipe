<template>
  <div class="recipe-container" @click="recipeDetail(recipeInfo)">
    <div class="recipe-card" :style="{ backgroundImage: `url(${recipeInfo.basic_imgurl})` }"></div>
    <v-container class="recipe-info">
      <p class="recipe-name">
        {{ recipeInfo.basic_name }}
        <span class="recipe-detail">{{ recipeInfo.basic_kcal }}</span>
      </p>
      <p class="recipe-detail">{{ recipeInfo.basic_intro }}</p>
    </v-container>
  </div>
</template>

<script>
export default {
  name: "RecipeCard",
  data: () => ({}),
  props: {
    recipeInfo: {
      type: Object,
      required: true
    }
  },
  methods: {
    recipeDetail(recipeInfo) {
      this.$router.push({
        name: "RecipeDetail",
        params: { id: recipeInfo.basic_code }
      });
    }
  }
};
</script>

<style scoped src="./RecipeCard.css">