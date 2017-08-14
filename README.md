# humanleague

[![Build Status](https://travis-ci.org/virgesmith/humanleague.png?branch=master)](https://travis-ci.org/virgesmith/humanleague)
[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html) 

python and Rcpp package for generating synthetic integer populations from integer marginal data, using quasirandom without-replacement sampling. 

- Supports up to 12 dimensions.
- Produces statistical data to ascertain the degeneracy of the population.

Also provides functionality to integerise marginals expressed as discrete probabilities, and for direct generation of quasirandom (Sobol) sequences.


## R installation

```
> devtools::install_github("virgesmith/humanleague")
```

## python installation

Requires Python 3, with numpy installed
```
pip install git+https://github.com/virgesmith/humanleague.git@master
```
### Build and test (from local repo)
```
user@host:~/dev/humanleague/python$ ./setup.py test
```
### Install (from local repo)
```
user@host:~/dev/humanleague/python$ ./setup.py install
```
Latter command may require admin rights.
On linux ensure you have group (e.g. staff) write access to /usr/local/lib, or run as root.

