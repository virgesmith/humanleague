# humanleague JSON API

[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html) 

## Dependencies

- node.js
- node-gyp
- [C++ JSON parser](http://github.com/nlohmann/json) - place in 3rdParty directory at same level as humanleague base directory
- npm
- npm packages: request, express 

## Build

```
user@host:~/dev/humanleague/json$ node-gyp configure # before first build only
user@host:~/dev/humanleague/json$ node-gyp build
```

## Test 
#### Local
(does not require express or request modules)
```
user@host:~/dev/humanleague/json$ nodejs test.js
```
#### http service
Start server
```
user@host:~/dev/humanleague/json$ nodejs server.js <port>
```
Client (localhost)
```
user@host:~/dev/humanleague/json$ nodejs test_server.js localhost:<port>
```
Client (remote)
```
user@otherhost:~/dev/humanleague/json$ nodejs test_server.js <host:port>
```
Integrated localhost client-server test
```
user@otherhost:~/dev/humanleague/json$ ./run_server_test.sh
```



