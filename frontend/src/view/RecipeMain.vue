<template>
  <RecipeRecommendation
    v-if="showMaterialPage"
    :materials="materials"
    :uploadedImage="uploadedImage"
    :percentages="percentages"
    :imageUrl="imageUrl"
  />
  <div class="main-container" v-else>
    <Loading v-if="loading" />
    <div class="possible">
      <span>현재 판별가능한 재료 목록</span>
      <div class="possibleList">감자, 계란, 고추, 사과, 스팸, 양파</div>
    </div>
    <div class="wrapper">
      <div class="logo" :class="{ rotate: rotate }">
        <label class="white-round front" for="ex_file" v-if="!rotate">
          <v-icon class="center">mdi-camera</v-icon>
        </label>
        <div class="white-round back" v-if="rotate">
          <div class="center">
            <v-icon>mdi-image-area</v-icon>
            <p>
              사진촬영 완료
              <br />하단 버튼을 눌러 <br />재료를 탐색하세요
            </p>
          </div>
        </div>
        <input type="file" @change="onFileChange" id="ex_file" />
      </div>
      <div class="intro">
        <div class="intro-txt" v-if="!rotate">
          딥러닝을 활용해 사진 속
          <br />식재료를 분석해 레시피를 <br />추천해드립니다.
        </div>
        <!-- <div class="explain"> 설명을 써 말어...
          Multi-lavel <br>
          Mask R-CNN
        </div>
        <v-icon v-if="rotate">mdi-help-box</v-icon>-->
        <div class="intro-button" @click="onMultiLabel" v-if="rotate">
          <span>
            Multi-
            <br />label
          </span>
        </div>
        <div class="intro-button" @click="onMaskRCNN" v-if="rotate">
          <span>
            Mask
            <br />R-CNN
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
<script>
import Loading from "../components/Loading";
import RecipeRecommendation from "../components/RecipeRecommendation/RecipeRecommendation";
import http from "../services/http-common.js";

export default {
  name: "RecipeMain",
  components: {
    Loading,
    RecipeRecommendation
  },

  data() {
    return {
      materials: [],
      percentages: [],
      uploadedImage: new FormData(),
      imageUrl: "",
      loading: false,
      showMaterialPage: false,
      rotate: false
    };
  },
  methods: {
    onMaskRCNN() {
      this.loading = !this.loading;
      http
        .post("/recipes/mask_rcnn/", this.uploadedImage)
        .then(res => {
          this.loading = !this.loading;
          this.changeBackground();
          this.materials = res.data.materials;
          this.percentages = res.data.percentages;
          this.showMaterialPage = !this.showMaterialPage;
          this.removeTransparentClass();
          console.log(res.data);
        })
        .catch(e => {
          alert(e);
          this.loading = false;
          (this.rotate = !this.rotate), (this.uploadedImage = new FormData());
        });
    },
    onFileChange(e) {
      var files = e.target.files || e.dataTransfer.files;
      if (!files.length) return;
      this.uploadedImage.append("file", files[0]);
      this.rotate = !this.rotate;
    },
    onMultiLabel() {
      this.loading = !this.loading;
      http
        .post("/recipes/image_upload/", this.uploadedImage)
        .then(res => {
          this.loading = !this.loading;
          this.changeBackground();
          this.materials = res.data.materials;
          this.percentages = res.data.percentages;
          this.showMaterialPage = !this.showMaterialPage;
          this.removeTransparentClass();
          console.log(res.data);
        })
        .catch(e => {
          alert(e);
          this.loading = false;
          (this.rotate = !this.rotate), (this.uploadedImage = new FormData());
        });
    },
    changeBackground() {
      const image = this.uploadedImage.get("file");
      var reader = new FileReader();
      reader.readAsDataURL(image);
      reader.onloadend = () => {
        this.imageUrl = reader.result;
      };
    },
    addTransparentClass() {
      const navClassList = document.querySelector("header").classList;
      navClassList.add("header-transparent");
    },
    removeTransparentClass() {
      const navClassList = document.querySelector("header").classList;
      if (navClassList.contains("header-transparent")) {
        navClassList.remove("header-transparent");
      }
    }
  },
  mounted() {
    this.addTransparentClass();
    window.addEventListener("scroll", this.onScroll);
  },
  beforeDestroy() {
    this.removeTransparentClass();
    window.removeEventListener("scroll", this.onScroll);
  }
};
</script>
<style scoped>
* {
  box-sizing: border-box;
}
.main-container {
  width: 100%;
  height: 100%;
  background-image: url("https://i.pinimg.com/564x/b9/26/90/b92690a0d83b7c15a7d00b5f97a1a170.jpg");
  background-size: cover;
  background-position: center;
  padding-top: 70px;
  position: relative;
  /* display: flex; */
  /* justify-content: center;
  align-items: center; */
}
.main-container::before {
  content: " ";
  width: 100%;
  height: 100%;
  position: absolute;
  top: 0;
  left: 0;
  background: -moz-radial-gradient(
    center,
    ellipse cover,
    rgba(0, 0, 0, 0) 0%,
    rgba(0, 0, 0, 0.4) 100%
  );
  background: -webkit-radial-gradient(
    center,
    ellipse cover,
    rgba(0, 0, 0, 0) 0%,
    rgba(0, 0, 0, 0.4) 100%
  );
  background: radial-gradient(
    ellipse at center,
    rgba(0, 0, 0, 0) 0%,
    rgba(0, 0, 0, 0.4) 100%
  );
  filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#00000000', endColorstr='#66000000',GradientType=1 );
}
.possible {
  position: absolute;
  left: 20px;
  bottom: 20px;
  color: white;
  text-decoration: underline;
}
.possibleList {
  width: 200px;
  height: 70px;
  padding: 10px;
  position: absolute;
  top: -70px;
  display: none;
  background-color: rgba(255, 255, 255, 0.75);
  border-radius: 5px;
  color: #555;
  font-weight: 700;
}
.possible:hover .possibleList {
  display: block;
}
.wrapper {
  width: 55%;
  height: 80%;
  margin: 50px auto 0 auto;
  perspective: 1000px;
  /* border: 1px blue solid; */
}
.logo {
  width: 100%;
  padding-bottom: 100%;
  position: relative;
  /* border: 1px solid red; */

  /* transition: 1s;
  transform-style: preserve-3d; */
}
/* .rotate {
  transform: rotateY(-180deg);
} */
.white-round {
  position: absolute;
  display: inline-block;
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.7);
  border-radius: 50%;
  box-shadow: 2px 4px 8px gray;
  /* backface-visibility: hidden; */
}
.white-round i {
  color: rgb(32, 32, 32);
  font-size: 140px;
}
.white-round p {
  margin-top: -10px;
  text-align: center;
  font-size: 22px;
  font-family: "Jua", sans-serif;
  font-weight: bold;
}
/* .front {
  z-index: 1;
  cursor: pointer;
}
.back {
  transform: rotateY(180deg);
} */
.logo input {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: rgba(0, 0, 0, 0.815);
  text-align: center;
}
.intro {
  width: 100%;
  margin-top: 80px;
  font-size: 20px;
  font-weight: 700;
  color: rgba(0, 0, 0, 0.9);
  /* border: 1px solid green; */
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  position: relative;
}
.intro-txt {
  color: rgb(255, 255, 255);
  font-family: "Gugi", cursive;
  text-align: center;
  font-size: 28px;
  padding: 12px;
  margin: 0 auto;
  background: -moz-radial-gradient(
    center,
    ellipse cover,
    rgba(0, 0, 0, 0.65) 0%,
    rgba(0, 0, 0, 0) 90%,
    rgba(0, 0, 0, 0) 91%
  ); /* FF3.6-15 */
  background: -webkit-radial-gradient(
    center,
    ellipse cover,
    rgba(0, 0, 0, 0.65) 0%,
    rgba(0, 0, 0, 0) 90%,
    rgba(0, 0, 0, 0) 91%
  ); /* Chrome10-25,Safari5.1-6 */
  background: radial-gradient(
    ellipse at center,
    rgba(0, 0, 0, 0.65) 0%,
    rgba(0, 0, 0, 0) 90%,
    rgba(0, 0, 0, 0) 91%
  ); /* W3C, IE10+, FF16+, Chrome26+, Opera12+, Safari7+ */
  filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#a6000000', endColorstr='#00000000',GradientType=1 ); /* IE6-9 fallback on horizontal gradient */
}
.explain {
  position: absolute;
  width: 100%;
  height: 80%;
  padding: 10px;
  display: block;
  background-color: rgb(83, 96, 173);
  border: solid 1px #3f51b5;
  color: white;
  border-radius: 10px;
  z-index: 2;
}
.intro i {
  position: absolute;
  right: 0;
  bottom: 0;
  font-size: 40px;
  color: #e8f5e9;
}
.intro i:hover .intro i {
  color: red;
}
.intro-button {
  width: 120px;
  height: 120px;
  margin: 50px 0 50px 0;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: rgba(255, 255, 255, 0.7);
  font-size: 24px;
  color: black;
  border-radius: 10px;
  box-shadow: 2px 4px 8px gray;
  cursor: pointer;
}
@media (max-width: 500px) {
  .wrapper {
    width: 70%;
  }
  .white-round i {
    font-size: 130px;
  }
  .white-round p {
    font-size: 18px;
  }
  .intro-button {
    width: 100px;
    height: 100px;
    font-size: 20px;
  }
  .intro-txt {
    font-size: 20px;
  }
}
</style>
