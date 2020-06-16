<template>
  <div>
    <div class="sequence-container">
      <div class="sequence-container-title">
        <b>조리순서</b>
        <span>Steps</span>
      </div>
      <div v-for="(sequence, i) in recipeSequence" :key="i">
        <div class="sequence_step" :style="getSequnceNumber(i + 1)">
          <div>{{ sequence.process_info }}</div>
          <img :src="sequence.process_imgurl" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "RecipeSequence",
  props: {
    recipeSequence: {
      type: Array
    },
    recipeCarousel: {
      type: Array
    }
  },
  methods: {
    getSequnceNumber(index) {
      return (
        "background: url(https://recipe1.ezmember.co.kr/img/icon_step_" +
        index +
        ".gif) no-repeat; background-size: 36px;"
      );
    }
  }
};
</script>

<style scope src="./RecipeSequence.css">