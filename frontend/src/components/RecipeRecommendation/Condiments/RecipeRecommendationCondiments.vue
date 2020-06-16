<template>
  <div>
    <main>
      <p>집에있는 조미료를 선택해주세요</p>
      <v-row class="row">
        <v-col
          v-for="condiment in Object.keys(condiments)"
          :key="condiment"
          :class="condiment"
          @click="paintingSeasoning(condiment)"
          cols="3"
        >
          <div class="condiments-icon" :style="getCondimentsIconPath(condiments[condiment])"></div>
          <div class="condiments-title">{{ condiment }}</div>
        </v-col>
      </v-row>
    </main>
  </div>
</template>

<script>
export default {
  name: "RecipeRecommendationCondiments",
  data: () => ({
    condiments: {
      소금: "salt",
      밀가루: "flour",
      올리고당: "syrup",
      식용유: "water",
      버터: "butter",
      식초: "soy",
      케찹: "ketchup",
      레몬소스: "lemon",
      마요네즈: "mayonnaise",
      부침가루: "flour",
      후추: "salt",
      참기름: "oil",
      고춧가루: "pepper",
      굴소스: "sauce",
      간장: "soy",
      카레가루: "flour",
      물엿: "syrup",
      // 물: "water",
      들기름: "oil",
      설탕: "salt"
    }
  }),
  methods: {
    getCondimentsIconPath(condiment) {
      let img = require("@/assets/ingredients/" + condiment + ".svg");
      return "mask-image: url(" + img + "); ";
    },

    paintingSeasoning(condiment) {
      const yellowgreen = document.querySelector(
        "." + condiment + " .condiments-icon"
      ).style.backgroundColor;

      document.querySelector(
        "." + condiment + " .condiments-icon"
      ).style.backgroundColor =
        yellowgreen === "yellowgreen" ? "gray" : "yellowgreen";

      this.sendSelectedSeasoning();
    },

    sendSelectedSeasoning() {
      const seasoningData = document.querySelectorAll(".condiments-icon");
      let data = [];
      seasoningData.forEach((v, i) => {
        if (v.style.backgroundColor === "yellowgreen") {
          if (data === undefined) {
            data = Object.keys(this.condiments)[i];
          } else {
            data.push(Object.keys(this.condiments)[i]);
          }
        }
      });
      this.$emit("condimentsData", data);
    }
  }
};
</script>

<style scoped src="./RecipeRecommendationCondiments.css">