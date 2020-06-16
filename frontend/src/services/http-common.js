import axios from "axios";

export default axios.create({
  baseURL: "https://i02b102.p.ssafy.io/",
  // baseURL: "http://211.213.225.87:8085/",
  headers: {
    "Content-type": "application/json"
  }
});
