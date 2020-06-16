<template>
  <div>
    <main>
      <v-row class="row">
        <v-col v-for="(material, i) in materials" :key="i" cols="4" class="col">
          <div class="material-color-icon-percent">
            <div :style="getMeterialColorIconPercent(material, i)">
              <img class="meterial-color-icon" :src="getMaterialColorIconPath(material, i)" />
            </div>
          </div>
          <img class="material-icon" :src="getMaterialIconPath(material, i)" />

          <div class="container-title" :style="getMaterialIconStyle(material, i)">
            {{ material }}
            <br />
            ({{ percentages[i] }}%)
          </div>
        </v-col>
      </v-row>
    </main>
  </div>
</template>

<script>
export default {
  name: "RecipeRecommendationMaterial",
  props: {
    materials: {
      type: Array
    },
    percentages: {
      type: Array
    }
  },
  data: () => ({
    materialItem: {
      감자: "potato",
      고추: "chili",
      사과: "apple",
      스팸: "spam",
      양파: "onion",
      계란: "egg"
    }
  }),
  methods: {
    getMeterialColorIconPercent(material, i) {
      let height = 50 * (this.percentages[i] / 100);
      return (
        "position: relative; overflow: hidden; max-height: " + height + "px"
      );
    },

    getMaterialIconPath(material, i) {
      let img = null;
      img = require("@/assets/ingredients/" +
        this.materialItem[material] +
        ".svg");
      return img;
    },

    getMaterialColorIconPath(material, i) {
      let img = null;
      img = require("@/assets/ingredients/" +
        this.materialItem[material] +
        "_color.svg");

      return img;
    },

    getMaterialIconStyle(material, i) {
      let style = null;
      let img = null;
      if (this.percentages[i] > 80) {
        img = require("@/assets/ingredients/" +
          this.materialItem[material] +
          ".svg");
        style = "font-weight: bold; color:red";
      } else {
        style = "color: gray";
      }

      return style;
    }
  }
};
</script>
<style scoped src="./RecipeRecommendationMaterial.css">