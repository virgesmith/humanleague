#!/usr/bin/nodejs

if (process.argv.length !== 3) {
  console.log("usage: nodejs server.js <port>");
  return;
}

var port = process.argv[2]

var express = require('express');

var app = express();

var humanleague = require("./build/Release/humanleague.node");

// register entry points
app.get('/sobolSequence', function(req, res) {
  console.log(req.query.args);
  // convert object back to string...(seems a bit perverse?)
  var result = humanleague.sobolSequence(req.query.args);
  res.status(200).send(result);
}); 

// app.get('/synthPop', function(req, res) {
//   console.log(req.query.args);
//   var result = humanleague.synthPop(req.query.args);
//   res.status(200).send(result);
// });


// Catch unrecognised incoming requests
app.get('*', function(req, res) {
  res.status(404).send(JSON.stringify({ "error": "route not found"}));
});

// Handle errors
app.use(function(err, req, res, next) {
  if (req.xhr) {
    res.status(500).send(JSON.stringify({ "error": "internal server error"}));
  } else {
    next(err);
  }
});

console.log("humanleague server running at port " + port);
app.listen(port);

