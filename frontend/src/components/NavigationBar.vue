<template>
  <header>
    <h1 @click="home">
      <a>냉장고를 부탁해</a>
    </h1>
  </header>
</template>

<script>
export default {
  name: "NavigationBar",
  methods: {
    home() {
      if (this.$router.history.current.name === "RecipeMain") {
        location.reload();
      } else {
        this.$router.push("/");
      }
    }
  }
};
</script>
<style>
header {
  height: 70px;
  width: 100%;
  max-width: 600px;
  background: white;
  position: fixed;
  top: 0;
  z-index: 100;
  border-bottom: 1px solid #cacaca;
  box-shadow: 1px 2px 4px rgba(180, 180, 180, 0.3);
}
header h1 {
  display: inline-block;
  position: absolute;
  left: 24px;
}
header h1 a {
  text-decoration: none;
  display: inline-block;
  height: 70px;
  line-height: 70px;
  font-size: 25px;
  font-family: "Gugi", cursive;
  color: rgba(0, 0, 0, 0.9) !important;
}
.header-transparent {
  background: transparent;
  border-bottom: none;
  box-shadow: none;
}
.header-transparent a {
  color: white !important;
}
@media (max-width: 850px) {
  header nav li {
    padding: 0 5px;
  }
  header nav li a {
    font-size: 12px;
  }
}
</style>
