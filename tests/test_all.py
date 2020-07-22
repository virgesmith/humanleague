#!/usr/bin/env python3

import numpy as np
import humanleague as hl

import unittest

class Test(unittest.TestCase):

  def test_unittest(self):
    res = hl.unittest()
    print("unit test fails/tests: ", res["nFails"], "/", res["nTests"])
    print(res["errors"])
    self.assertTrue(res["nFails"] == 0)

  def test_apitest(self):
    res = hl.apitest()
    #print("api test: ", res)
    self.assertTrue(res is None)

  def test_sobolSequence(self):
    a = hl.sobolSequence(3, 5)
    self.assertTrue(a.size == 15)
    self.assertTrue(a.shape == (5, 3))
    self.assertTrue(np.array_equal(a[0, :], [0.5, 0.5, 0.5]))

    # invalid dimension
    a = hl.sobolSequence(0, 10)
    self.assertTrue(isinstance(a, str))
    a = hl.sobolSequence(100000, 10)
    self.assertTrue(isinstance(a, str))
    
    # invalid length
    a = hl.sobolSequence(1, -10)
    self.assertTrue(isinstance(a, str))

  def test_integerise(self):

    # probs not valid
    r = hl.prob2IntFreq(np.array([0.3, 0.3, 0.2, 0.1]), 10)
    self.assertTrue(r == "probabilities do not sum to unity")

    # pop not valid
    r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), -1)
    self.assertTrue(r == "population cannot be negative")

    # zero pop
    r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 0)
    self.assertTrue(r["rmse"] == 0.0)
    self.assertTrue(np.array_equal(r["freq"], np.array([0, 0, 0, 0])))

    # exact
    r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 10)
    self.assertTrue(r["rmse"] == 0.0)
    self.assertTrue(np.array_equal(r["freq"], np.array([4, 3, 2, 1])))

    # inexact
    r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 17)
    self.assertAlmostEqual(r["rmse"], np.sqrt(0.075))
    self.assertTrue(np.array_equal(r["freq"], np.array([7, 5, 3, 2])))

    # 1-d case
    r = hl.integerise(np.array([2.0, 1.5, 1.0, 0.5]))
    self.assertTrue(r["conv"])

    # multidim integerisation
    # invalid population
    s = np.array([[1.1, 1.0], [1.0, 1.0]])
    r = hl.integerise(s)
    self.assertEqual(r, "Marginal or total value 4.100000 is not an integer (within tolerance 0.000100)")
    # invalid marginals
    s = np.array([[1.1, 1.0], [0.9, 1.0]])
    r = hl.integerise(s)
    self.assertEqual(r, "Marginal or total value 2.100000 is not an integer (within tolerance 0.000100)")

    # use IPF to generate a valid fractional population
    m0 = np.array([111,112,113,114,110], dtype=float)
    m1 = np.array([136,142,143,139], dtype=float)
    s = np.ones([len(m0),len(m1),len(m0)])

    fpop = hl.ipf(s, [np.array([0]),np.array([1]),np.array([2])], [m0,m1,m0])["result"]

    result = hl.integerise(fpop)
    self.assertTrue(result["conv"])
    self.assertEqual(np.sum(result["result"]), sum(m0))
    self.assertTrue(result["rmse"] < 1.05717)

  def test_IPF(self):
    m0 = np.array([52.0, 48.0])
    m1 = np.array([87.0, 13.0])
    m2 = np.array([55.0, 45.0])
    i = [np.array([0]),np.array([1])]

    s = np.ones([len(m0), len(m1)])
    p = hl.ipf(s, i, [m0, m1])
    #print(p)
    self.assertTrue(p["conv"])
    self.assertEqual(p["pop"], 100.0)
    self.assertTrue(np.array_equal(p["result"], np.array([[45.24, 6.76], [41.76, 6.24]])))

    s[0, 0] = 0.7
    p = hl.ipf(s, i, [m0, m1])
    #print(np.sum(p["result"], 0))
    self.assertTrue(p["conv"])
    # check overall population and marginals correct
    self.assertEqual(np.sum(p["result"]), p["pop"])
    self.assertTrue(np.allclose(np.sum(p["result"], 0), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], 1), m0))

    i = [np.array([0]),np.array([1]),np.array([2])]
    s = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    p = hl.ipf(s, i, [m0, m1, m2])
    #print(np.sum(p["result"], (0, 1)))
    #print(np.sum(p["result"], (1, 2)))
    #print(np.sum(p["result"], (2, 0)))
    self.assertTrue(p["conv"])
    # check overall population and marginals correct
    self.assertAlmostEqual(np.sum(p["result"]), p["pop"]) # default is 7d.p.
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1)), m2))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 0)), m1))

    # 12D
    s = np.ones([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    i = [np.array([0]),np.array([1]),np.array([2]),np.array([3]),np.array([4]),np.array([5]),
         np.array([6]),np.array([7]),np.array([8]),np.array([9]),np.array([10]),np.array([11])]
    m = np.array([2048., 2048.])
    p = hl.ipf(s,i,[m, m, m, m, m, m, m, m, m, m, m, m])
    #print(p)
    self.assertTrue(p["pop"] == 4096)

    m0 = np.array([52.0, 48.0])
    m1 = np.array([87.0, 13.0])
    m2 = np.array([55.0, 45.0])

    seed = np.ones([len(m0), len(m1)])
    p = hl.ipf(seed, [np.array([0]),np.array([1])], [m0, m1])
    self.assertTrue(np.allclose(np.sum(p["result"], (0)), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], (1)), m0))
    self.assertTrue(p["conv"])
    self.assertEqual(p["iterations"], 1)
    self.assertEqual(p["maxError"], 0.0)
    self.assertEqual(p["pop"], 100.0)
    self.assertTrue(np.array_equal(p["result"], np.array([[45.24, 6.76], [41.76, 6.24]])))

    seed[0, 1] = 0.7
    p = hl.ipf(seed, [np.array([0]),np.array([1])], [m0, m1])
    self.assertTrue(np.allclose(np.sum(p["result"], (0)), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], (1)), m0))
    self.assertTrue(p["conv"])
    self.assertLess(p["iterations"], 6)
    self.assertLess(p["maxError"], 5e-10)
    self.assertEqual(p["pop"], 100.0)

    s = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    p = hl.ipf(s, [np.array([0]),np.array([1]),np.array([2])], [m0, m1, m2])
    #print(np.sum(p["result"], (0, 1)))
    #print(np.sum(p["result"], (1, 2)))
    #print(np.sum(p["result"], (2, 0)))
    self.assertTrue(p["conv"])
    # check overall population and marginals correct
    self.assertAlmostEqual(np.sum(p["result"]), p["pop"]) # default is 7d.p.
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1)), m2))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 0)), m1))

    # 12D
    s = np.ones([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    m = np.array([2048., 2048.])
    p = hl.ipf(s, [np.array([0]),np.array([1]),np.array([2]),np.array([3]),np.array([4]),np.array([5]),np.array([6]),np.array([7]),np.array([8]),np.array([9]),np.array([10]),np.array([11])],[m, m, m, m, m, m, m, m, m, m, m, m])
    self.assertTrue(p["conv"] == True)
    self.assertTrue(p["pop"] == 4096)

  def test_QIS(self):

    # m = np.array([[10,20,10],[10,10,20],[20,10,10]])
    # idx = [np.array([0,1]), np.array([1,2])]
    # r = hl.qis(idx, [m, m])
    # self.assertTrue(false)

    m0 = np.array([52, 48])
    m1 = np.array([10, 77, 13])
    i0 = np.array([0])
    i1 = np.array([1])

    p = hl.qis([i0, i1], [m0, m1])
    #print(p)
    self.assertTrue(p["conv"])
    self.assertLess(p["chiSq"], 0.04)
    self.assertGreater(p["pValue"], 0.9)
    #self.assertLess(p["degeneracy"], 0.04) TODO check the calculation
    self.assertEqual(p["pop"], 100.0)
    self.assertTrue(np.allclose(np.sum(p["result"], 0), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], 1), m0))
    #self.assertTrue(np.array_equal(p["result"], np.array([[5, 40, 7],[5, 37, 6]])))

    m0 = np.array([52, 40, 4, 4])
    m1 = np.array([87, 10, 3])
    m2 = np.array([55, 15, 6, 12, 12])
    i0 = np.array([0])
    i1 = np.array([1])
    i2 = np.array([2])

    p = hl.qis([i0, i1, i2], [m0, m1, m2])
    self.assertTrue(p["conv"])
    self.assertLess(p["chiSq"], 73.0) # TODO seems a bit high (probably )
    self.assertGreater(p["pValue"], 0.0) # TODO this is suspect
    self.assertEqual(p["pop"], 100.0)
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1)), m2))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 0)), m1))

    # Test flatten functionality
    table = hl.flatten(p["result"])

    # length is no of dims
    self.assertTrue(len(table) == 3)
    # length of element is pop
    self.assertTrue(len(table[0]) == p["pop"])
    # check consistent with marginals
    for i, mi in enumerate(m0):
      self.assertTrue(table[0].count(i) == mi)
    for i, mi in enumerate(m1):
      self.assertTrue(table[1].count(i) == mi)
    for i, mi in enumerate(m2):
      self.assertTrue(table[2].count(i) == mi)


    m0 = np.array([52, 48])
    m1 = np.array([87, 13])
    m2 = np.array([67, 33])
    m3 = np.array([55, 45])
    i0 = np.array([0])
    i1 = np.array([1])
    i2 = np.array([2])
    i3 = np.array([3])

    p = hl.qis([i0, i1, i2, i3], [m0, m1, m2, m3])
    self.assertTrue(p["conv"])
    self.assertLess(p["chiSq"], 10)
    self.assertGreater(p["pValue"], 0.002) # TODO this looks suspect too
    self.assertEqual(p["pop"], 100)
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1, 2)), m3))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2, 3)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 3, 0)), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], (3, 0, 1)), m2))

    m = np.array([[10,20,10],[10,10,20],[20,10,10]])
    idx = [np.array([0,1]), np.array([1,2])]
    p = hl.qis(idx, [m, m])
    #print(p)
    self.assertTrue(p["conv"])
    self.assertLess(p["chiSq"], 10)
    self.assertGreater(p["pValue"], 0.27)
    self.assertEqual(p["pop"], 120)
    #print(np.sum(p["result"], 2))
    self.assertTrue(np.allclose(np.sum(p["result"], 2), m))
    self.assertTrue(np.allclose(np.sum(p["result"], 0), m))

  def test_QIS_dim_indexing(self):

    # tricky array indexing - 1st dimension of d0 already sampled, remaining dimension
    # indices on slice of d0 need to be remapped

    m0 = np.ones([4,6,4,4], dtype=int)
    m1 = np.ones([4,4,4], dtype=int) * 6

    ms=hl.qis([np.array([0,1,2,3]),np.array([0,4,5])], [m0,m1])
    self.assertTrue(ms["conv"])

    ms=hl.qis([np.array([0,4,5]),np.array([0,1,2,3])], [m1,m0])
    self.assertTrue(ms["conv"])

    ms=hl.qis([np.array([0,1,2]),np.array([0,3,4,5])], [m1,m0])
    self.assertTrue(ms["conv"])


  def test_QISI(self):
    m0 = np.array([52, 48])
    m1 = np.array([10, 77, 13])
    i0 = np.array([0])
    i1 = np.array([1])
    s = np.ones([len(m0), len(m1)])

    p = hl.qisi(s, [i0, i1], [m0, m1])
    #print(p)
    self.assertTrue(p["conv"])
    self.assertLess(p["chiSq"], 0.04)
    self.assertGreater(p["pValue"], 0.9)
    #self.assertLess(p["degeneracy"], 0.04) TODO check the calculation
    self.assertEqual(p["pop"], 100.0)
    self.assertTrue(np.allclose(np.sum(p["result"], 0), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], 1), m0))
    #self.assertTrue(np.array_equal(p["result"], np.array([[5, 40, 7],[5, 37, 6]])))

    m0 = np.array([52, 40, 4, 4])
    m1 = np.array([87, 10, 3])
    m2 = np.array([55, 15, 6, 12, 12])
    i0 = np.array([0])
    i1 = np.array([1])
    i2 = np.array([2])
    s = np.ones([len(m0), len(m1), len(m2)])

    p = hl.qisi(s, [i0, i1, i2], [m0, m1, m2])
    self.assertTrue(p["conv"])
    self.assertLess(p["chiSq"], 70) # seems a bit high
    self.assertGreater(p["pValue"], 0.0) # seems a bit low
    self.assertEqual(p["pop"], 100.0)
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1)), m2))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 0)), m1))

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
    self.assertTrue(p["conv"])
    self.assertLess(p["chiSq"], 5.5)
    self.assertGreater(p["pValue"], 0.02)
    self.assertEqual(p["pop"], 100)
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1, 2)), m3))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2, 3)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 3, 0)), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], (3, 0, 1)), m2))

    # check dimension consistency check works
    s = np.ones([2,3,7,5])
    m1 = np.ones([2,3], dtype=int) * 5 * 7
    m2 = np.ones([3,5], dtype=int) * 7 * 2
    m3 = np.ones([5,7], dtype=int) * 2 * 3
    p = hl.qisi(s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1, m2, m3])
    self.assertEqual(p, "seed dimensions [2, 3, 7, 5] are inconsistent with that implied by marginals ([2, 3, 5, 7])")

    p = hl.ipf(s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1.astype(float), m2.astype(float), m3.astype(float)])
    self.assertEqual(p, "seed dimensions [2, 3, 7, 5] are inconsistent with that implied by marginals ([2, 3, 5, 7])")

    s = np.ones((2,3,5))
    p = hl.qisi(s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1, m2, m3])
    self.assertEqual(p, "seed dimensions 3 is inconsistent with that implied by marginals (4)")

    p = hl.ipf(s, [np.array([0,1]), np.array([1,2]), np.array([2,3])], [m1.astype(float), m2.astype(float), m3.astype(float)])
    self.assertEqual(p, "seed dimensions 3 is inconsistent with that implied by marginals (4)")

if __name__ == "__main__":
  unittest.main()