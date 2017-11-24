---
title: 'humanleague: a C++ microsynthesis package with R and python interfaces'
tags:
  - c++
  - r
  - python
  - microsynthesis
  - sampling
authors:
 - name: Andrew P Smith
   orcid: 0000-0002-9951-6642
   affiliation: 1
affiliations:
 - name: School of Geography and Leeds Institute for Data Analytics, University of Leeds
   index: 1
date: 24 November 2017
bibliography: paper.bib
---

# Summary

humanleague is a microsynthesis package for R and python, with its core implementation in C++. It provides both traditional and novel algorithms for generating synthetic populations from two or more marginal constraints and, optionally, a seed population. The marginal constraints can be of arbitrary dimensionality.

The package provides a fast implementation of the traditional Iterative Proportional Fitting (IPF) algorithm, which generates fractional populations given marginal constraints and a seed population. Where integral populations are preferred, the package also provides two variants of a quasirandom sampling algorithm (QIS) which generate high-entropy 'IPF-like' whole-number populations. The first variant is extremely fast but can only be used where no seed data is provided, and is described in [@smith2017]. The second variant (QIS-I) supports a seed population by sampling from a dynamically-computed IPF population. 

The QIS-I algorithm can also be used to integerise precomputed multidimensional fractional populations. Functions are also provided to integerise discrete univariate propbability distributions, directly generate quasirandom (Sobol) sequences, and to convert populations represented as counts in a multidimensional state array to a tabular form listing individuals.

# References
