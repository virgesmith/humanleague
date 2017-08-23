#!/usr/bin/env python3

import humanleague as hl
import numpy as np

from unittest import TestCase

class Test(TestCase):

  # just to ensure test harness works
  def test_init(self):
    self.assertTrue(True)

  def test_sobolSequence(self):
    a = hl.sobolSequence(3,5)
    self.assertTrue(a.size == 15)
    self.assertTrue(a.shape == (5,3))
    self.assertTrue(np.array_equal(a[0,:], [ 0.5, 0.5, 0.5]))

  def test_synthPop(self):

    p = hl.synthPop([[4,2],[1,2,3]])
    self.assertTrue(p["conv"])

    p = hl.synthPop([[1.0],[1,2,3,4,5,6]])
    self.assertTrue(p == 'object is not an int')

    p = hl.synthPop(["a",[1,2,3,4,5,6]])
    self.assertTrue(p == 'object is not a list')

    p = hl.synthPop([[4,2],[1,2,3],[3,3]])
    self.assertTrue(p["conv"])

  def test_synthPopG(self):
  
    p = hl.synthPopG(np.array([4,2]),np.array([1,2,3]),np.array([[1.0, 0.9, 0.8],[0.5, 0.6, 0.7]]))
    self.assertTrue(p["conv"])
    self.assertTrue(p["pop"] == 6)
    
  def test_integerise(self):

    # probs not valid
    r = hl.prob2IntFreq(np.array([0.3,0.3,0.2,0.1]),10)
    self.assertTrue(r == "probabilities do not sum to unity")

    # pop not valid
    r = hl.prob2IntFreq(np.array([0.4,0.3,0.2,0.1]),0)
    self.assertTrue(r == "population must be strictly positive")

    # exact  
    r = hl.prob2IntFreq(np.array([0.4,0.3,0.2,0.1]),10)
    self.assertTrue(r["var"] == 0.0)
    self.assertTrue(np.array_equal(r["freq"], np.array([4,3,2,1])))

    # inexact
    r = hl.prob2IntFreq(np.array([0.4,0.3,0.2,0.1]),17)
    self.assertAlmostEqual(r["var"], 0.075)
    self.assertTrue(np.array_equal(r["freq"], np.array([7,5,3,2])))


