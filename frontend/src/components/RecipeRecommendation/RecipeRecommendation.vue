<template>
  <div class="container">
    <Loading v-if="loading" />
    <div class="image" :style="{ backgroundImage: `url(${imageUrl}` }"></div>
    <RecipeRecommendationMaterial
      :materials="materials"
      :percentages="percentages"
      class="content"
    />

    <RecipeRecommendationCondiments
      @condimentsData="getCondiments"
      class="content"
    />
    <v-btn
      class="search-recipe-button"
      color="light-green"
      @click="searchRecipe"
      >레시피 검색하기</v-btn
    >
  </div>
</template>

<script>
import http from "@/services/http-common.js";
import Loading from "@/components/Loading";
import RecipeRecommendationMaterial from "./Meterial/RecipeRecommendationMaterial";
import RecipeRecommendationCondiments from "./Condiments/RecipeRecommendationCondiments";

export default {
  name: "RecipeRecommendation",
  components: {
    Loading,
    RecipeRecommendationMaterial,
    RecipeRecommendationCondiments
  },

  data: () => ({
    loading: false,
    condiments: [],
    selectmaterials: []
  }),
  props: {
    materials: {
      type: Array
    },
    imageUrl: {
      type: String
    },
    percentages: {
      type: Array
    }
  },
  methods: {
    getCondiments(data) {
      this.condiments = data;
    },

    searchRecipe() {
      this.percentages.forEach((v, i) => {
        if (v > 80) {
          this.selectmaterials.push(this.materials[i]);
        }
      });

      const data = {
        // 상위 컴포넌트에서 가져온 것 (냉장고에서 찾은 재료)
        materials: this.selectmaterials,

        // 하위 컴포넌트에서 가져온 것 (사용자가 선택한 양념정보)
        condiments: this.condiments
      };
      if (data.condiments.length === 0) {
        alert("앙념을 1개이상 선택해 주세요.");
      } else if (data.materials.length === 0) {
        alert("80%이상 판별된 재료가 없습니다.");
      } else {
        this.loading = !this.loading;
        http.post("/recipes/get_dishes/", data).then(res => {
          this.loading = !this.loading;
          const payload = {
            // 찾은 레시피정보 vuex에 저장
            recipeInfoArr: res.data
          };
          this.loading = !this.loading;

          this.$store.dispatch("recipeInfo", payload);
          this.$router.push("/RecipeList");
        });
      }
    }
  }
};
</script>

<style scoped>
.container {
  width: 100%;
  height: 100%;
  margin-top: 70px;
  text-align: center;
}
.search-recipe-button {
  margin-top: 30px;
  font-family: "Cute Font", cursive;
}
.image {
  width: 100%;
  height: 240px;
  background-color: black;
  background-size: contain;
  background-position: center;
}
.content {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
