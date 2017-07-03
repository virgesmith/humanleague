#!/usr/bin/nodejs

var humanleague_api = require("./build/Release/humanleague.node");

var seq = humanleague_api.sobolSequence(JSON.stringify({dim: 2, length: 10}));
console.log(JSON.parse(seq));

seq = humanleague_api.synthPop(JSON.stringify({marginals:[[1,1,1,1],[1,2,1]]}));
console.log(JSON.parse(seq));


