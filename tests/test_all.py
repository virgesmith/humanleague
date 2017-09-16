#!/usr/bin/env python3

import humanleague as hl
import numpy as np

from unittest import TestCase

class Test(TestCase):

  def test_sobolSequence(self):
    a = hl.sobolSequence(3, 5)
    self.assertTrue(a.size == 15)
    self.assertTrue(a.shape == (5, 3))
    self.assertTrue(np.array_equal(a[0, :], [0.5, 0.5, 0.5]))

  def test_synthPop(self):

    p = hl.synthPop([np.array([4, 2]), np.array([1, 2, 3])])
    print(p)
    self.assertTrue(p["conv"])

    p = hl.synthPop([np.array([4, 1]), np.array([1, 2, 3])])
    self.assertTrue(p == "invalid marginals")

    p = hl.synthPop([[4, 2], [1, 2, 3]])
    self.assertTrue(p == "input should be a list of numpy integer arrays")

    p = hl.synthPop([np.array([1.0]), np.array([1, 2, 3, 4, 5, 6])])
    self.assertTrue(p == 'python array contains invalid type: 12 when expecting 7')

    p = hl.synthPop([np.array([4, 2]), np.array([1, 2, 3]), np.array([3, 3])])
    self.assertTrue(p["conv"])

  def test_synthPopG(self):

    p = hl.synthPopG(np.array([4, 2]), np.array([1, 2, 3]), np.array([[1.0, 0.9, 0.8], [0.5, 0.6, 0.7]]))
    self.assertTrue(p["conv"])
    self.assertTrue(p["pop"] == 6)

  def test_integerise(self):

    # probs not valid
    r = hl.prob2IntFreq(np.array([0.3, 0.3, 0.2, 0.1]), 10)
    self.assertTrue(r == "probabilities do not sum to unity")

    # pop not valid
    r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 0)
    self.assertTrue(r == "population must be strictly positive")

    # exact
    r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 10)
    self.assertTrue(r["var"] == 0.0)
    self.assertTrue(np.array_equal(r["freq"], np.array([4, 3, 2, 1])))

    # inexact
    r = hl.prob2IntFreq(np.array([0.4, 0.3, 0.2, 0.1]), 17)
    self.assertAlmostEqual(r["var"], 0.075)
    self.assertTrue(np.array_equal(r["freq"], np.array([7, 5, 3, 2])))

  def test_IPF(self):
    m0 = np.array([52.0, 48.0])
    m1 = np.array([87.0, 13.0])
    m2 = np.array([55.0, 45.0])

    s = np.ones([len(m0), len(m1)])
    p = hl.ipf(s, [m0, m1])
    #print(p)
    self.assertTrue(p["conv"])
    self.assertTrue(p["pop"] == 100)
    self.assertTrue(np.array_equal(p["result"], np.array([[45.24, 6.76], [41.76, 6.24]])))

    s[0, 0] = 0.7
    p = hl.ipf(s, [m0, m1])
    #print(np.sum(p["result"], 0))
    self.assertTrue(p["conv"])
    # check overall population and marginals correct
    self.assertTrue(np.sum(p["result"]) == p["pop"])
    self.assertTrue(np.allclose(np.sum(p["result"], 0), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], 1), m0))

    s = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    p = hl.ipf(s, [m0, m1, m2])
    print(np.sum(p["result"], (0, 1)))
    print(np.sum(p["result"], (1, 2)))
    print(np.sum(p["result"], (2, 0)))
    self.assertTrue(p["conv"])
    # check overall population and marginals correct
    self.assertTrue(np.sum(p["result"]) == p["pop"])
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1)), m2))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 0)), m1))

    # 12D
    s = np.ones([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    m = np.array([2048., 2048.])
    p = hl.ipf(s,[m, m, m, m, m, m, m, m, m, m, m, m])
    print(p)
    self.assertTrue(p["pop"] == 4096)

  def test_QSIPF(self):
    m0 = np.array([52, 48]) 
    m1 = np.array([10, 77, 13])

    # nonunity seed
    s = np.ones([len(m0), len(m1)])
    s[0,0] = 0.7
    s[1,1] = 1.3
    p = hl.qsipf(s, [m0, m1])
    self.assertTrue(p["conv"])
    self.assertTrue(p["chiSq"] < 0.1) # ~0.091
    self.assertTrue(p["pop"] == 100)
    self.assertTrue(np.allclose(np.sum(p["result"], 0), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], 1), m0))
    #self.assertTrue(np.array_equal(p["result"], np.array([[5, 40, 7],[5, 37, 6]])))

    m0 = np.array([52, 40, 4, 4]) 
    m1 = np.array([87, 10, 3])
    m2 = np.array([55, 15, 6, 12, 12])

    s = np.ones([len(m0), len(m1), len(m2)])
    p = hl.qsipf(s, [m0, m1, m2])
    self.assertTrue(p["conv"])
    self.assertTrue(p["chiSq"] < 70) # TODO seems a bit high (probably )
    self.assertTrue(p["pop"] == 100)
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1)), m2))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 0)), m1))

    m0 = np.array([52, 48]) 
    m1 = np.array([87, 13])
    m2 = np.array([67, 33])
    m3 = np.array([55, 45])

    s = np.ones([len(m0), len(m1), len(m2), len(m3)])
    p = hl.qsipf(s, [m0, m1, m2, m3])
    self.assertTrue(p["conv"])
    self.assertTrue(p["chiSq"] < 5.25) # 
    self.assertTrue(p["pop"] == 100)
    self.assertTrue(np.allclose(np.sum(p["result"], (0, 1, 2)), m3))
    self.assertTrue(np.allclose(np.sum(p["result"], (1, 2, 3)), m0))
    self.assertTrue(np.allclose(np.sum(p["result"], (2, 3, 0)), m1))
    self.assertTrue(np.allclose(np.sum(p["result"], (3, 0, 1)), m2))