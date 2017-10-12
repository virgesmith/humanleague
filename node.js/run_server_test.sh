#!/bin/bash

server_log=server.log
port=3000

# start server
nodejs server.js $port > $server_log 2>&1 &
server_pid=$!
echo server: pid = $server_pid port = $port log = $server_log
echo waiting for server...
sleep 1

# run tests
echo running tests
nodejs test_server.js localhost:$port

echo killing server
kill $server_pid

