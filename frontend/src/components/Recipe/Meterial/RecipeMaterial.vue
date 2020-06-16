<template>
  <div>
    <div class="meterial-container">
      <div class="meterial-container-title">
        <b>재료</b>
        <span>Material</span>
      </div>
      <div class="meterial-list">
        <li v-for="(material, i) in recipeMaterial" :key="i">
          {{ material.material_name }}
          <span class="meterial-unit">{{ material.material_volume }}</span>
        </li>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "RecipeMaterial",
  props: {
    recipeMaterial: {
      type: Array
    }
  }
};
</script>

<style scoped src="./RecipeMeterial.css">
