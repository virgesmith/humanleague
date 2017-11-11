import numpy as np
import humanleague as hl

m = np.array([[10,20,10],[10,10,20],[20,10,10]])
idx = [np.array([0,1]), np.array([1,2])]
r = hl.qis(idx, [m, m])

print(r)