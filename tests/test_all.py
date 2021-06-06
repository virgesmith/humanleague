#!/usr/bin/env python3

import numpy as np
import humanleague as hl


def assert_throws(e, f, *args, **kwargs):
  try:
    f(*args, **kwargs)
  except e:
    pass
  else:
    assert False, "expected exception %s not thrown" % e

def assert_close(x, y, tol=1e-8): # ~sqrt(epsilon)
  assert abs(x-y) < tol

def test_version():
  assert hl.__version__

def test_unittest():
  res = hl.unittest()
  print("unit test fails/tests: ", res["nFails"], "/", res["nTests"])
  print(res["errors"])
  assert res["nFails"] == 0

def test_sobolSequence():
  a = hl.sobolSequence(3, 5)
  assert a.size == 15
  assert a.shape == (5, 3)
  assert np.array_equal(a[0, :], [0.5, 0.5, 0.5])

  # invalid args
  assert_throws(ValueError, hl.sobolSequence, 0, 10)
  assert_throws(ValueError, hl.sobolSequence, 100000, 10)
  assert_throws(ValueError, hl.sobolSequence, 1, -10)

def test_integerise():

  # probs not valid
  # r = hl.prob2IntFreq(np.array([0.3, 0.3, 0.2, 0.1]), 10)
  # assert r == "probabilities do not sum to unity"

  # pop not valid
  assert_throws(ValueError, hl.prob2IntFreq, np.array([0.4, 0.3, 0.2, 0.1]), -1)
  #assert r == "population cannot be negative"

  # zero pop
  r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 0)
  assert r["rmse"] == 0.0
  assert np.array_equal(r["freq"], np.array([0, 0, 0, 0]))

  # exact
  r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 10)
  assert r["rmse"] < 1e-15
  assert np.array_equal(r["freq"], np.array([4, 3, 2, 1]))

  # inexact
  r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 17)
  assert r["rmse"] == 0.273861278752583
  assert np.array_equal(r["freq"], np.array([7, 5, 3, 2]))

  # 1-d case
  r = hl.integerise(np.array([2.0, 1.5, 1.0, 0.5]))
  assert r["conv"]

  # multidim integerisation
  # invalid population
  s = np.array([[1.1, 1.0], [1.0, 1.0]])
  assert_throws(RuntimeError, hl.integerise, s)
  # invalid marginals
  s = np.array([[1.1, 1.0], [0.9, 1.0]])
  assert_throws(RuntimeError, hl.integerise, s)

  # use IPF to generate a valid fractional population
  m0 = np.array([111,112,113,114,110], dtype=float)
  m1 = np.array([136,142,143,139], dtype=float)
  s = np.ones([len(m0),len(m1),len(m0)])

  fpop = hl.ipf(s, [np.array([0]),np.array([1]),np.array([2])], [m0,m1,m0])["result"]

  result = hl.integerise(fpop)
  assert result["conv"]
  assert np.sum(result["result"]) == sum(m0)
  assert result["rmse"] < 1.05717

def test_IPF():
  m0 = np.array([52.0, 48.0])
  m1 = np.array([87.0, 13.0])
  m2 = np.array([55.0, 45.0])
  i = [np.array([0]),np.array([1])]

  s = np.ones([len(m0), len(m1)])
  p = hl.ipf(s, i, [m0, m1])
  #print(p)
  assert p["conv"]
  assert p["pop"] == 100.0
  assert np.array_equal(p["result"], np.array([[45.24, 6.76], [41.76, 6.24]]))

  s[0, 0] = 0.7
  p = hl.ipf(s, i, [m0, m1])
  #print(np.sum(p["result"], 0))
  assert p["conv"]
  # check overall population and marginals correct
  assert np.sum(p["result"]) == p["pop"]
  assert np.allclose(np.sum(p["result"], 0), m1)
  assert np.allclose(np.sum(p["result"], 1), m0)

  i = [np.array([0]),np.array([1]),np.array([2])]
  s = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
  p = hl.ipf(s, i, [m0, m1, m2])
  #print(np.sum(p["result"], (0, 1)))
  #print(np.sum(p["result"], (1, 2)))
  #print(np.sum(p["result"], (2, 0)))
  assert p["conv"]
  # check overall population and marginals correct
  assert_close(np.sum(p["result"]), p["pop"])
  assert np.allclose(np.sum(p["result"], (0, 1)), m2)
  assert np.allclose(np.sum(p["result"], (1, 2)), m0)
  assert np.allclose(np.sum(p["result"], (2, 0)), m1)

  # 12D
  s = np.ones([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
  i = [np.array([0]),np.array([1]),np.array([2]),np.array([3]),np.array([4]),np.array([5]),
        np.array([6]),np.array([7]),np.array([8]),np.array([9]),np.array([10]),np.array([11])]
  m = np.array([2048., 2048.])
  p = hl.ipf(s,i,[m, m, m, m, m, m, m, m, m, m, m, m])
  #print(p)
  assert p["pop"] == 4096

  m0 = np.array([52.0, 48.0])
  m1 = np.array([87.0, 13.0])
  m2 = np.array([55.0, 45.0])

  seed = np.ones([len(m0), len(m1)])
  p = hl.ipf(seed, [np.array([0]),np.array([1])], [m0, m1])
  assert np.allclose(np.sum(p["result"], (0)), m1)
  assert np.allclose(np.sum(p["result"], (1)), m0)
  assert p["conv"]
  assert p["iterations"] == 1
  assert p["maxError"] == 0.0
  assert p["pop"] == 100.0
  assert np.array_equal(p["result"], np.array([[45.24, 6.76], [41.76, 6.24]]))

  seed[0, 1] = 0.7
  p = hl.ipf(seed, [np.array([0]),np.array([1])], [m0, m1])
  assert np.allclose(np.sum(p["result"], (0)), m1)
  assert np.allclose(np.sum(p["result"], (1)), m0)
  assert p["conv"]
  assert p["iterations"] < 6
  assert p["maxError"] < 5e-10
  assert p["pop"] == 100.0

  s = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
  p = hl.ipf(s, [np.array([0]),np.array([1]),np.array([2])], [m0, m1, m2])
  #print(np.sum(p["result"], (0, 1)))
  #print(np.sum(p["result"], (1, 2)))
  #print(np.sum(p["result"], (2, 0)))
  assert p["conv"]
  # check overall population and marginals correct
  assert_close(np.sum(p["result"]), p["pop"])
  assert np.allclose(np.sum(p["result"], (0, 1)), m2)
  assert np.allclose(np.sum(p["result"], (1, 2)), m0)
  assert np.allclose(np.sum(p["result"], (2, 0)), m1)

  # 12D
  s = np.ones([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
  m = np.array([2048., 2048.])
  p = hl.ipf(s, [np.array([0]),np.array([1]),np.array([2]),np.array([3]),np.array([4]),np.array([5]),np.array([6]),np.array([7]),np.array([8]),np.array([9]),np.array([10]),np.array([11])],[m, m, m, m, m, m, m, m, m, m, m, m])
  assert p["conv"]
  assert p["pop"] == 4096

def test_QIS():

  # m = np.array([[10,20,10],[10,10,20],[20,10,10]])
  # idx = [np.array([0,1]), np.array([1,2])]
  # r = hl.qis(idx, [m, m])
  # assert false)

  m0 = np.array([52, 48])
  m1 = np.array([10, 77, 13])
  i0 = np.array([0])
  i1 = np.array([1])

  p = hl.qis([i0, i1], [m0, m1])
  #print(p)
  assert p["conv"]
  assert p["chiSq"] < 0.04
  assert p["pValue"] > 0.9
  #.assertLess(p["degeneracy"], 0.04) TODO check the calculation
  assert p["pop"] == 100.0
  assert np.allclose(np.sum(p["result"], 0), m1)
  assert np.allclose(np.sum(p["result"], 1), m0)
  #assert np.array_equal(p["result"], np.array([[5, 40, 7],[5, 37, 6]])))

  m0 = np.array([52, 40, 4, 4])
  m1 = np.array([87, 10, 3])
  m2 = np.array([55, 15, 6, 12, 12])
  i0 = np.array([0])
  i1 = np.array([1])
  i2 = np.array([2])

  p = hl.qis([i0, i1, i2], [m0, m1, m2])
  assert p["conv"]
  assert p["chiSq"] < 73.0 # TODO seems a bit high (probably )
  assert p["pValue"] > 0.0 # TODO this is suspect
  assert p["pop"] == 100.0
  assert np.allclose(np.sum(p["result"], (0, 1)), m2)
  assert np.allclose(np.sum(p["result"], (1, 2)), m0)
  assert np.allclose(np.sum(p["result"], (2, 0)), m1)

  # Test flatten functionality
  table = hl.flatten(p["result"])

  # length is no of dims
  assert len(table) == 3
  # length of element is pop
  assert len(table[0]) == p["pop"]
  # check consistent with marginals
  for i, mi in enumerate(m0):
    assert table[0].count(i) == mi
  for i, mi in enumerate(m1):
    assert table[1].count(i) == mi
  for i, mi in enumerate(m2):
    assert table[2].count(i) == mi


  m0 = np.array([52, 48])
  m1 = np.array([87, 13])
  m2 = np.array([67, 33])
  m3 = np.array([55, 45])
  i0 = np.array([0])
  i1 = np.array([1])
  i2 = np.array([2])
  i3 = np.array([3])

  p = hl.qis([i0, i1, i2, i3], [m0, m1, m2, m3])
  assert p["conv"]
  assert p["chiSq"] < 10
  assert p["pValue"] > 0.002 # TODO this looks suspect too
  assert p["pop"] == 100
  assert np.allclose(np.sum(p["result"], (0, 1, 2)), m3)
  assert np.allclose(np.sum(p["result"], (1, 2, 3)), m0)
  assert np.allclose(np.sum(p["result"], (2, 3, 0)), m1)
  assert np.allclose(np.sum(p["result"], (3, 0, 1)), m2)

  m = np.array([[10,20,10],[10,10,20],[20,10,10]])
  idx = [np.array([0,1]), np.array([1,2])]
  p = hl.qis(idx, [m, m])
  #print(p)
  assert p["conv"]
  assert p["chiSq"] < 10
  assert p["pValue"] > 0.27
  assert p["pop"] == 120
  #print(np.sum(p["result"], 2))
  assert np.allclose(np.sum(p["result"], 2), m)
  assert np.allclose(np.sum(p["result"], 0), m)

def test_QIS_dim_indexing():

  # tricky array indexing - 1st dimension of d0 already sampled, remaining dimension
  # indices on slice of d0 need to be remapped

  m0 = np.ones([4,6,4,4], dtype=int)
  m1 = np.ones([4,4,4], dtype=int) * 6

  ms=hl.qis([np.array([0,1,2,3]),np.array([0,4,5])], [m0,m1])
  assert ms["conv"]

  ms=hl.qis([np.array([0,4,5]),np.array([0,1,2,3])], [m1,m0])
  assert ms["conv"]

  ms=hl.qis([np.array([0,1,2]),np.array([0,3,4,5])], [m1,m0])
  assert ms["conv"]


def test_QISI():
  m0 = np.array([52, 48])
  m1 = np.array([10, 77, 13])
  i0 = np.array([0])
  i1 = np.array([1])
  s = np.ones([len(m0), len(m1)])

  p = hl.qisi(s, [i0, i1], [m0, m1])
  #print(p)
  assert p["conv"]
  assert p["chiSq"] < 0.04
  assert p["pValue"] > 0.9
  #.assertLess(p["degeneracy"], 0.04) TODO check the calculation
  assert p["pop"] == 100.0
  assert np.allclose(np.sum(p["result"], 0), m1)
  assert np.allclose(np.sum(p["result"], 1), m0)
  #assert np.array_equal(p["result"], np.array([[5, 40, 7],[5, 37, 6]])))

  m0 = np.array([52, 40, 4, 4])
  m1 = np.array([87, 10, 3])
  m2 = np.array([55, 15, 6, 12, 12])
  i0 = np.array([0])
  i1 = np.array([1])
  i2 = np.array([2])
  s = np.ones([len(m0), len(m1), len(m2)])

  p = hl.qisi(s, [i0, i1, i2], [m0, m1, m2])
  assert p["conv"]
  assert p["chiSq"] < 70 # seems a bit high
  assert p["pValue"] > 0.0 # seems a bit low
  assert p["pop"] == 100.0
  assert np.allclose(np.sum(p["result"], (0, 1)), m2)
  assert np.allclose(np.sum(p["result"], (1, 2)), m0)
  assert np.allclose(np.sum(p["result"], (2, 0)), m1)

  m0 = np.array([52, 48])
  m1 = np.array([87, 13])
  m2 = np.array([67, 33])
  m3 = np.array([55, 45])
  i0 = np.array([0])
  i1 = np.array([1])
  i2 = np.array([2])
  i3 = np.array([3])
  s = np.ones([len(m0), len(m1), len(m2), len(m3)])

  p = hl.qisi(s, [i0, i1, i2, i3], [m0, m1, m2, m3])
  assert p["conv"]
  assert p["chiSq"] < 5.5
  assert p["pValue"] > 0.02
  assert p["pop"] == 100.0
  assert np.allclose(np.sum(p["result"], (0, 1, 2)), m3)
  assert np.allclose(np.sum(p["result"], (1, 2, 3)), m0)
  assert np.allclose(np.sum(p["result"], (2, 3, 0)), m1)
  assert np.allclose(np.sum(p["result"], (3, 0, 1)), m2)

  # check dimension consistency check works
  s = np.ones([2,3,7,5])
  m1 = np.ones([2,3], dtype=int) * 5 * 7
  m2 = np.ones([3,5], dtype=int) * 7 * 2
  m3 = np.ones([5,7], dtype=int) * 2 * 3
  assert_throws(RuntimeError, hl.qisi, s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1, m2, m3])

  assert_throws(RuntimeError, hl.ipf, s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1.astype(float), m2.astype(float), m3.astype(float)])

  s = np.ones((2,3,5))
  assert_throws(RuntimeError, hl.qisi, s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1, m2, m3])

  assert_throws(RuntimeError, hl.ipf, s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1.astype(float), m2.astype(float), m3.astype(float)])

