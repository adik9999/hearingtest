const express = require("express");
//const bodyParser = require("body-parser");
const exp = express();
//const request = require("request");
//const https = require("https");
//exp.use(express.static("public"));
exp.get("/", function(req, res) {
  res.sendFile(__dirname + "/index.html");
});
exp.use(bodyParser.urlencoded({
  extended: true
}));
exp.post("/", function(req, res) {
  var first_name = req.body.fname;
  exp.listen(process.env.PORT||3000, function() {
  console.log("l@3000");
});
