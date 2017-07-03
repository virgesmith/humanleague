#!/usr/bin/nodejs

if (process.argv.length != 3) {
  console.log("usage: nodejs test_server.js <host:port>");
  return;
}

var hostport = process.argv[2]


var request = require('request');

//var req1 = { dim: 2, length: 10 };

url = "http://" + hostport + "/sobolSequence?dim=2&length=10"
console.log(url);

request(encodeURI(url), function(err, resp, res) {
  // TODO some error checking
  //res = JSON.parse(res);
  console.log(res);
});

