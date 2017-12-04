# humanleague

[![Travis Build Status](https://travis-ci.org/virgesmith/humanleague.png?branch=master)](https://travis-ci.org/virgesmith/humanleague)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/x9oypgryt21ndc3p?svg=true)](https://ci.appveyor.com/project/virgesmith/humanleague)
[![codecov](https://codecov.io/gh/virgesmith/humanleague/branch/master/graph/badge.svg)](https://codecov.io/gh/virgesmith/humanleague)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)

*humanleague* is a python and an R package for microsynthesising populations from marginal and (optionally) seed data. The core code is implemented in C++, and the current release is version 2.

The package contains algorithms that use a number of different microsynthesis techniques:
- [Iterative Proportional Fitting (IPF)](https://en.wikipedia.org/wiki/Iterative_proportional_fitting)
- [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) (no seed population)
- Quasirandom Integer Sampling of IPF (QISI): A combination of the two techniques whereby IPF solutions are used to sample an integer population.

The latter provides a bridge between deterministic reweighting and combinatorial optimisation, offering advantages of both techniques:
- generates high-entropy integral populations 
- can be used to generate multiple populations for sensitivity analysis
- goes some way to address the 'empty cells' issues that can occur in straight IPF
- relatively fast compuation time

The algorithms: 
- support arbitrary dimensionality* for both the marginals and the seed.
- produce statistical data to ascertain the likelihood/degeneracy of the population (where appropriate).

[* excluding the legacy functions retained for backward compatibility with version 1.0.1]

The package also contains the following utility functions:
- a Sobol sequence generator
- functionality to convert fractional to nearest-integer marginals (in 1D). This can also be achieved in multiple dimensions by using the QISI algorithm.
- functionality to 'flatten' a population into a table: this converts a multidimensional array containing the population count for each state into a table listing individuals and their characteristics. 

Version 1.0.1 reflects the work described in the [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) paper.

## R installation
```
> devtools::install_github("virgesmith/humanleague")
```
Or, for the previous version
```
> devtools::install_github("virgesmith/humanleague@1.0.1")
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
The latter command may require admin rights. On linux, `sudo` is unnecessary if you have group (e.g. staff) write access to /usr/local/lib.

### Examples

Consult the package documentation, e.g.
```
> library(humanleague)
> ?humanleague
```
in R, or for python:
```
>>> import humanleague as hl
>>> help(hl)
```
