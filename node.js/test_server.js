#!/usr/bin/nodejs

if (process.argv.length != 3) {
  console.log("usage: nodejs test_server.js <host:port>");
  return;
}

var hostport = process.argv[2]


var request = require('request');

args = { dim: 7, length: 10 };

// Args are always passed as "args=<string>" as this avoids all numeric values being implicitly converted to strings
url = "http://" + hostport + "/sobolSequence?args=" + JSON.stringify(args);
console.log(url);

request(encodeURI(url), function(err, resp, res) {
  // TODO some error checking
  res = JSON.parse(res);
  console.log(res);
});

// args = { marginals: [[1,1,1,1],[1,2,1]] };

// // Args are always passed as "args=<string>" as this avoids all numeric values being implicitly converted to strings
// url = "http://" + hostport + "/synthPop?args=" + JSON.stringify(args);
// console.log(url);
// request(encodeURI(url), function(err, resp, res) {
//   // TODO some error checking
//   res = JSON.parse(res);
//   console.log(res);
// });



