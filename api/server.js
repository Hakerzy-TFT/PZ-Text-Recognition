const express = require("express");
const cors = require("cors");
const http = require('http');

const hostname = '127.0.0.1';
const port = 3000;
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
// app.post("/sendimg", function(req, res) {
//     var img = req.body.imgsrc;
     
    
//   });

  app.get("/aa", function(req, res) {
    var fs = require('fs');
    var dir = '../letters'
    

    fs.mkdir(dir, function(err) {
        if (err) {
          console.log(err)
        } else {
          console.log("New directory successfully created.")
        }
      })
    var imgData = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAABkAQAAAACAsFvaAAAB8UlEQVR4nO2YMW7bQBBF/1CErcKwWKowQB4hNwiPoiP4AA68pW8VpkrKlO6ygF2kpAUXEiBrXNAMl9oZkuMEAQxwGu3+weMfrXaHCxHDGomZmJEZ+R8IUWZ32b5N6uaDqBxFuGgm9wCAPeBHkJz9YyjdTSlsfQwUdp+HT2oCAOdfA+VIVVfYN1IQfAqUw1kwqeAUZBkmymDsUClILxHiq4NWWBonmsiE3dFIiZcJBqmFaVEIWoMISwkAOEriiEtpR+wuwhKPujgzsrO7iCs5jNQ68qIgXkfElQFwJokN8ltBxGgQL1YwhLyUZpfncyvC2dqIPCVqq9P2WI1FT6q6oX5ebnrSshvmCpKzU+pCJmWEPbYnatVlFaclZAe0TTn105AwWwui0PRWXeOnZ7MLvtiRTWFGMuFYDCE74GLvYp055za+MzPvVu30gZmrLtuG4JKFkyKubKzB5nszAleakWtvRi5rMxL3WAEp+lNncRGP8T9/i63e6RI3BgVh4O1mE38hBTn+efYhi5HT+1r6REQpAKQVgDs/6FIBCN92PwHhvCThT9f1SXIAki3wYxEhcYdJ8l8td0tSjx1c5A2w8MOIO0leAVfxg+LCgkvbQrz99lwysbxBZFq8D/F/4zINTtSr6DSXaUHzXwQzMiMfH3kFoDmNZ05v3aQAAAAASUVORK5CYII=';
var base64Data = imgData.replace(/^data:image\/png;base64,/, "");

require("fs").writeFile("../image/out.png", base64Data, 'base64', 
function(err, data) {
if (err) {
    console.log('err', err);
}
console.log('success');
res.json({
    status: "success",
  });

  const { exec } = require("child_process");

  exec("py ../img_slicer.py", (error, stdout, stderr) => {
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

  exec("py ../tf.py ", (error, stdout, stderr) => {
    if (error) {
        console.log(`error: ${error.message}`);
        return;
    }
    if (stderr) {
        console.log(`stderr: ${stderr}`);
        return;
    }
    console.log(`stdout: ${stdout}`);
    fs.readFile('../sth.txt', 'utf8', function(err, data) {
        if (err) throw err;
        console.log(data);
    });

    
});

fs.rm('../sth.txt', { recursive:true }, (err) => {
    if(err){
        // File deletion failed
        console.error(err.message);
        return;
    }
    console.log("File deleted successfully");
})





});
     
    
  });
  app.listen(3000, function () {
    console.log("Listening");
  });
