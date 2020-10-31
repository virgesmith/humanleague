
import numpy as np
import humanleague as hl

print(hl.version())

p = np.array([1.1, 0.2, 0.3, 0.4])
n = 11

print(hl.prob2IntFreq(p, n))
#print(hl.prob2IntFreq(p, -n))

print(hl.sobolSequence(1, 5))
print(hl.sobolSequence(2, 5))
print(hl.sobolSequence(3, 5))

a = np.array([[ 0.3,  1.2,  2. ,  1.5],
              [ 0.6,  2.4,  4. ,  3. ],
              [ 1.5,  6. , 10. ,  7.5],
              [ 0.6,  2.4,  4. ,  3. ]])
# marginal sums
print(sum(a))
#array([ 3., 12., 20., 15.])
print(sum(a.T))
#array([ 5., 10., 25., 10.])
# perform integerisation
r = hl.integerise(a)
print(r)