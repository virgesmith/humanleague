# humanleague

[![Build Status](https://travis-ci.org/virgesmith/humanleague.png?branch=master)](https://travis-ci.org/virgesmith/humanleague)
[![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html) 

python and R package for microsynthesising populations from marginal and seed data, using a variety of techniques:
- Iterative Proportional Fitting (IPF)
- [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) (no seed population)
- A combination of the above whereby (seeded) IPF solutions are sampled to generate integer populations (QISI).

The latter goes some way to address the 'empty cells' issues that can occur in straight IPF. The combined algorithm is a bridge between the deterministic reweighting and the combinatorial optimisation camps, combining advantages of both (efficiency and degeneracy).

- Supports arbitrary dimensionality (both marginals and seed).
- Produces statistical data to ascertain the degeneracy of the population.

The package also contains a number of utility functions, e.g.
- A Sobol sequence generator
- Functionality to convert fractional to nearest-integer marginals (in 1D). This can also be achieved in multiple dimensions by using the QISI algorithm.

_The package (and thus the documentation) is currently under development_

However, version 1.0.1 is stable, and reflects the work described in the [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) paper.

## R installation
```
> devtools::install_github("virgesmith/humanleague")
```
Or, for the stable (but limited) version
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
Latter command may require admin rights.
On linux ensure you have group (e.g. staff) write access to /usr/local/lib, or run as root.

