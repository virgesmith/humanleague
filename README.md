# humanleague

[![CRAN\_Status\_Badge](http://www.r-pkg.org/badges/version/humanleague)](https://CRAN.R-project.org/package=humanleague)
[![CRAN Downloads](http://cranlogs.r-pkg.org/badges/grand-total/humanleague?color=black)](http://cran.r-project.org/package=humanleague)
[![PyPI version](https://badge.fury.io/py/humanleague.svg)](https://badge.fury.io/py/humanleague)
[![Travis Build Status](https://travis-ci.org/virgesmith/humanleague.png?branch=master)](https://travis-ci.org/virgesmith/humanleague)
[![Appveyor Build status](https://ci.appveyor.com/api/projects/status/x9oypgryt21ndc3p?svg=true)](https://ci.appveyor.com/project/virgesmith/humanleague)
[![codecov](https://codecov.io/gh/virgesmith/humanleague/branch/master/graph/badge.svg)](https://codecov.io/gh/virgesmith/humanleague)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1116318.svg)](https://doi.org/10.5281/zenodo.1116318)
[![status](http://joss.theoj.org/papers/d5aaf6e1c2efed431c506762622473b4/status.svg)](http://joss.theoj.org/papers/d5aaf6e1c2efed431c506762622473b4)

> ## Latest news: 2.1.0 pre-release
> - adds new functionality for multidimensional integerisation. 
> - deletes previously deprecated functionality `synthPop` and `synthPopG`.
> ### Multidimensional integerisation
> Building on the `prob2IntFreq` function - which takes a discrete probability distribution and a count, and returns the closest integer population to the distribution that sums to the count - a multidimensional equivalent `integerise` is introduced.
> 
> In one dimension, for example:
> ```python
> >>> import numpy as np
> >>> import humanleague
> >>> p=np.array([0.1, 0.2, 0.3, 0.4])
> >>> humanleague.prob2IntFreq(p, 11)
> {'freq': array([1, 2, 3, 5]), 'rmse': 0.3535533905932736}
> ```
> produces the optimal (i.e. closest possible) integer population to the discrete distribution.
>  
> The `integerise` function generalises this problem and applies it to higher dimensions: given an n-dimensional array of real numbers where the 1-d marginal sums in every dimension are integral (and thus the total population is too), it attempts to find an integral array that also satisfies these constraints. 

> The QISI algorithm is repurposed to this end. As it is a sampling algorithm it cannot guarantee that a solution is found, and if so, whether the solution is optimal. If it fails this does not prove that a solution does not exist for the given input.

> ```python
> >>> a = np.array([[ 0.3,  1.2,  2. ,  1.5],
>                   [ 0.6,  2.4,  4. ,  3. ],
>                   [ 1.5,  6. , 10. ,  7.5],
>                   [ 0.6,  2.4,  4. ,  3. ]])
> # marginal sums
> >> sum(a)
> array([ 3., 12., 20., 15.])
> >>> sum(a.T)
> array([ 5., 10., 25., 10.])
> # perform integerisation
> >>> r = humanleague.integerise(a)
> >>> r["conv"]
> True
> >>> r["result"]
> array([[ 0,  2,  2,  1],
>        [ 0,  3,  4,  3],
>        [ 2,  6, 10,  7],
>        [ 1,  1,  4,  4]])
> >>> r["rmse"]
> 0.5766281297335398
> # check marginals are preserved
> >>> sum(r["result"]) == sum(a)
> array([ True,  True,  True,  True])
> >>> sum(r["result"].T) == sum(a.T)
> array([ True,  True,  True,  True])
> ```

*humanleague* is a python and an R package for microsynthesising populations from marginal and (optionally) seed data. The core code is implemented in C++, and the current release is version 2.

The package contains algorithms that use a number of different microsynthesis techniques:
- [Iterative Proportional Fitting (IPF)](https://en.wikipedia.org/wiki/Iterative_proportional_fitting)
- [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) (no seed population)
- Quasirandom Integer Sampling of IPF (QISI): A combination of the two techniques whereby the results of repeated IPF algorithms are used to sample an integer population.

The latter provides a bridge between deterministic reweighting and combinatorial optimisation, offering advantages of both techniques:
- generates high-entropy integral populations 
- can be used to generate multiple populations for sensitivity analysis
- goes some way to address the 'empty cells' issues that can occur in straight IPF
- relatively fast compuation time

The algorithms: 
- support arbitrary dimensionality* for both the marginals and the seed.
- produce statistical data to ascertain the likelihood/degeneracy of the population (where appropriate).

The package also contains the following utility functions:
- a Sobol sequence generator
- construct a closest integer population from a discrete univariate probability distribution.
- an algorithm for creating multidimensional integer populations from fractional ones, constrained to the marginal sums in each dimension.  
- 'flatten' a multidimensional population into a table: this converts a multidimensional array containing the population count for each state into a table listing individuals and their characteristics. 

Version 1.0.1 reflects the work described in the [Quasirandom Integer Sampling (QIS)](http://jasss.soc.surrey.ac.uk/20/4/14.html) paper.

## R installation
Official release:
```
> install.packages("humanleague")
```
For development version
```bash
> devtools::install_github("virgesmith/humanleague")
```
Or, for the legacy version
```bash
> devtools::install_github("virgesmith/humanleague@1.0.1")
```
## python installation

Requires Python 3 and numpy. PyPI package:
```bash
python3 -m pip install humanleague --user
```
[Conda-forge package is being worked on]

### Build, install and test (from local cloned repo)
```bash
$ ./setup.py install --user
```
```bash
$ ./setup.py test
```
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
