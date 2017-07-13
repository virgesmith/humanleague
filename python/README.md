# humanleague Python API

[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html) 

## Dependencies

- Python 3.0 or greater, with numpy and dev tools installed

```
user@host:~/dev/humanleague/python$ sudo apt install python3-dev python3-numpy
```

## Build

```
user@host:~/dev/humanleague/python$ ./setup.py build
user@host:~/dev/humanleague/python$ ./setup.py install
```
Latter command may require admin rights.
On linux ensure you have group write access to /usr/local/lib, or run as root.

## Test 
```
user@host:~/dev/humanleague/python$ ./test.py
```
