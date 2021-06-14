const express = require("express");
const cors = require("cors");
const http = require("http");

const app = express();

app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.use(express.json());
app.get("/", function (req, res) {
  res.json({
    status: "OK",
  });
});

app.get("/sendimg", function (req, res) {
  var fs = require("fs");
  var file = "../result.txt";
  var imageSplitted = "../imageSplitted";
  if (!fs.existsSync(imageSplitted)) {
    fs.mkdirSync(imageSplitted);
  }
  var imgData = req.body.imgsrc;
  var base64Data = imgData.replace(/^data:image\/png;base64,/, "");
  require("fs").writeFile(
    "../image/out.png",
    base64Data,
    "base64",
    function (err, data) {
      if (err) {
        console.log("err", err);
      }
      console.log("success");

      const { exec, execSync } = require("child_process");

      execSync("py ../img_slicer.py", (error, stdout, stderr) => {
        if (error) {
          console.log(`error: ${error.message}`);
          return;
        }
        if (stderr) {
          console.log(`stderr: ${stderr}`);
          return;
        }
        console.log(`stdout: ${stdout}`);
      });

      execSync("py ../tf.py ", (error, stdout, stderr) => {
        if (error) {
          console.log(`error: ${error.message}`);
          return;
        }
        if (stderr) {
          console.log(`stderr: ${stderr}`);
          return;
        }
        console.log(`stdout: ${stdout}`);
      });

      fs.readFile(file, "utf8", function (err, data) {
        if (err) throw err;
        console.log(data);
        res.json({
          result: data,
        });
      });
    }
  );
});
app.listen(3000, function () {
  console.log("Listening");
});
