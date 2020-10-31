# API Reference
## `humanleague.flatten` (function)

```python
flatten(pop: numpy.ndarray[numpy.int64]) -> list
```


Converts an n-dimensional array of counts into an n-column table with a row for each unit

Args:

pop: The population.

Returns:

A 2-d array of size n by sum(pop). 


## `humanleague.integerise` (function)

```python
integerise(*args, **kwargs)
```
Overloaded function.

```python
 integerise(frac: numpy.ndarray[numpy.float64], pop: int) -> dict
```


Computes the closest integer frequencies given fractional counts and a total population. 

Args:

frac: The fractional counts (must be a 1-d array).

pop: The growth rate

Returns:

The frequencies and the RMS error


```python
 integerise(pop: numpy.ndarray[numpy.float64]) -> dict
```


Tries to construct and integer multidimensional array that has identical marginal sums to the fractional input array (which of course must have 
integer marginal sums). The algorithm may not always find a solution and will return an approximate array in this case.

Args:

pop: The fractional population.

Returns:

A dictionary containing The integral population, the RMS error, and a boolean indicating whether the population matches the marginal sums. 


## `humanleague.ipf` (function)

```python
ipf(seed: numpy.ndarray[numpy.float64], indices: list, marginals: list) -> dict
```


Uses iterative proportional fitting to construct an n-dimensional array from a seed population that matches the specified marginal sums.

seed: The seed population as an array.

indices: A list of the indices in the overall array that each marginal represents 

marginals: A list of arrays containing the marginal sums.

Returns:

A dictionary containing the result, a convergence flag, the total population, the iterations and the error


## `humanleague.prob2IntFreq` (function)

```python
prob2IntFreq(probs: numpy.ndarray[numpy.float64], pop: int) -> dict
```


Computes the closest integer frequencies given fractional counts and a total population. 

Args:

frac: The fractional counts (must be a 1-d array).

pop: The growth rate

Returns:

The frequencies and the RMS error


## `humanleague.qis` (function)

```python
qis(*args, **kwargs)
```
Overloaded function.

```python
 qis(indices: list, marginals: list, skips: int) -> dict
```


Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

indices: A list of the indices in the overall array that each marginal represents 

marginals: A list of arrays containing the marginal sums.

skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

Returns:

A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.


```python
 qis(indices: list, marginals: list) -> dict
```


Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

indices: A list of the indices in the overall array that each marginal represents 

marginals: A list of arrays containing the marginal sums.

skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

Returns:

A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.


## `humanleague.qisi` (function)

```python
qisi(*args, **kwargs)
```
Overloaded function.

```python
 qisi(seed: numpy.ndarray[numpy.float64], indices: list, marginals: list, skips: int) -> dict
```


Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

seed: The dimension of the sequence (between 1 and 1111).

indices: A list of the indices in the overall array that each marginal represents 

marginals: A list of arrays containing the marginal sums.

skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

Returns:

A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.


```python
 qisi(seed: numpy.ndarray[numpy.float64], indices: list, marginals: list) -> dict
```


Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

indices: A list of the indices in the overall array that each marginal represents 

marginals: A list of arrays containing the marginal sums.

skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

Returns:

A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.


## `humanleague.sobolSequence` (function)

```python
sobolSequence(*args, **kwargs)
```
Overloaded function.

```python
 sobolSequence(dim: int, length: int, skips: int) -> numpy.ndarray[numpy.float64]
```


Returns a Sobol' sequence given of supplied dimension and length, optionally skipping values.

dim: The dimension of the sequence (between 1 and 1111).

length: The length of the returned sequence

skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

Returns:

A 2d array containing Sobol sequence values in (0,1). 


```python
 sobolSequence(dim: int, length: int) -> numpy.ndarray[numpy.float64]
```


Returns a Sobol' sequence given of supplied dimension and length, optionally skipping values.

dim: The dimension of the sequence (between 1 and 1111).

length: The length of the returned sequence

skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

Returns:

A 2d array containing Sobol sequence values in (0,1). 


## `humanleague.unittest` (function)

```python
unittest() -> dict
```


Developers only. Runs the C++ unit tests. 


## `humanleague.version` (function)

```python
version() -> str
```


Gets the module version


