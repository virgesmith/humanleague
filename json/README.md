# humanleague JSON API

[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html) 

JSON API to humanleague. 

## Dependencies

- node.js
- node-gyp
- [C++ JSON parser](http://github.com/nlohmann/json) - place in 3rdParty directory at same level as humanleague base directory
- npm
- npm packages: request, express 

## Build

```
node-gyp configure # before first build only
node-gyp build
```

## Test 
#### Local
(does not require express or request modules)
```
nodejs test.js
```
#### http service
Server
```
nodejs server.js <port>
```
Client
```
nodejs test_server.js <hostname:port>
```
