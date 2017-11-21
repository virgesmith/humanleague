# humanleague

[![Build Status](https://travis-ci.org/virgesmith/humanleague.png?branch=master)](https://travis-ci.org/virgesmith/humanleague)
[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html) 

*humanleague* is a python and an R package for microsynthesising populations from marginal and (optionally) seed data. The core code is implemented in C++, and the current release is version 2.

The package contains algorithms that use a number of different techniques:
- Iterative Proportional Fitting (IPF)
- [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) (no seed population)
- QIS-IPF: A combination of the above whereby (seeded) IPF solutions are sampled to generate integer populations.

The latter provides a bridge between deterministic reweighting and combinatorial optimisation, offering advantages of both techniques:
- generates high-entropy integral populations 
- can be used to generate multiple populations for sensitivity analysis
- goes some way to address the 'empty cells' issues that can occur in straight IPF.
- relatively fast compuation time

The algorithms: 
- supports arbitrary dimensionality* (both marginals and seed).
- produce statistical data to ascertain the likelihood/degeneracy of the population (where appropriate).

[* excluding the legacy functions retained for backward compatibility with version 1.0.1]

The package also contains a number of utility functions, e.g.
- A Sobol sequence generator
- Functionality to convert fractional to nearest-integer marginals (in 1D). This can also be achieved in multiple dimensions by using the QISI algorithm.
- Functionality to `flatten` a population into a table: this converts a multidimensional array containing the population of each state into a table listing individuals and their characteristics. 

Version 1.0.1 reflects the work described in the [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) paper.

## R installation
```
> devtools::install_github("virgesmith/humanleague")
```
Or, for the stable (but limited) version
```
> devtools::install_github("virgesmith/humanleague@1.0.1")
```
Version 1.0.1 reflects the work described in the [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) paper.

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

