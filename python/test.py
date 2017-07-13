#!/usr/bin/python3

import humanleague
import numpy as np

print(humanleague.sobolSequence(3,5))

print(humanleague.sobolSequence(6,10))

print(humanleague.synthPop([[4,2],[1,2,3]]))
print(humanleague.synthPop([[1.0],[1,2,3,4,5,6]]))
print(humanleague.synthPop(["a",[1,2,3,4,5,6]]))
print(humanleague.synthPop([[4,2],[1,2,3],[3,3]]))

print(humanleague.synthPopR([4,2],[1,2,3],0.0))
print(humanleague.synthPopR([4,2],[1,2,3],1.0))
print(humanleague.synthPopR([10,10,10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10,10,10],0.0))
print(humanleague.synthPopR([21,0],[1,2,3,4,5,6],3.1))
print(humanleague.numpytest(np.array([[2,3,4],[6,7,8]])))
#print(humanleague.numpytest([[2,3,4],[6,7,8]]))
