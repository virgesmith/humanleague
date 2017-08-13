#!/usr/bin/env python3

import humanleague as hl
import numpy as np

# temporary
import sys
def check(cond):
  if not cond:
    sys.exit(1)

#TODO proper unit testing
# best solution is probably to move from distutils to setuptools in setup.py
from unittest import TestCase

class Test(TestCase):

  # just to ensure test harness works
  def test_init(self):
    self.assertTrue(True)

  def test_false(self):
    self.assertTrue(False)

#t = Test()

a = hl.sobolSequence(3,5)
check(a.size == 15)
check(a.shape == (5,3))
check(np.array_equal(a[0,:], [ 0.5, 0.5, 0.5]))

#print(hl.sobolSequence(6,10))

p = hl.synthPop([[4,2],[1,2,3]])
#print(p)
check(p["conv"])

p = hl.synthPop([[1.0],[1,2,3,4,5,6]])
check(p == 'object is not an int')

p = hl.synthPop(["a",[1,2,3,4,5,6]])
check(p == 'object is not a list')

p = hl.synthPop([[4,2],[1,2,3],[3,3]])
check(p["conv"])

print(hl.synthPopR([4,2],[1,2,3],0.0))
print(hl.synthPopR([4,2],[1,2,3],1.0))
print(hl.synthPopR([10,10,10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10,10,10],0.0))
print(hl.synthPopR([21,0],[1,2,3,4,5,6],3.1))

# TODO...
#print(hl.synthPopG(np.array([4,2]),np.array([1,2,3]),np.array([[1.0, 0.9, 0.8],[0.5, 0.6, 0.7]])))

print(hl.numpytest(np.array([[2,3,4],[6,7,8]])))
#print(hl.numpytest([[2,3,4],[6,7,8]]))    self.assertTrue(True)




