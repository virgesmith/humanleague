
import numpy as np
import humanleague as hl

print(hl.version())

p = np.array([1.1, 0.2, 0.3, 0.4])
n = 11

print(hl.prob2IntFreq(p, n))
#print(hl.prob2IntFreq(p, -n))

print(hl.sobol(1, 5))
print(hl.sobol(2, 5))
print(hl.sobol(3, 5))

