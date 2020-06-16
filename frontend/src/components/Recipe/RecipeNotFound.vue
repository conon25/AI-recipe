<template>
  <div class="videomain">
    <video
      id="video"
      playsinline
      muted
      autoplay
      loop
      data-silent="true"
      src="https://cdn.dribbble.com/users/1693462/screenshots/3504905/icon-404.gif?vid=1"
    ></video>
    <p>레시피가 존재하지 않습니다.</p>
  </div>
</template>

<script>
export default {};
</script>

<style scoped>
p {
  font-size: 1.4em;
  font-family: "Cute Font", cursive;
  position: absolute;
  left: 50%;
  bottom: 30%;
  z-index: 100;
  transform: translateX(-50%);
}

@media screen and (min-width: 400px) {
  p {
    font-size: 2em;
    font-family: "Cute Font", cursive;
    position: absolute;
    left: 50%;
    bottom: 25%;
    z-index: 100;
    transform: translateX(-50%);
  }
}

video {
  position: absolute;
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  margin: auto;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
}
</style>

