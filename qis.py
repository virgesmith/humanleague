import numpy as np
import humanleague as hl

# THIS CRASHES
m = np.array([[10,20,5,5],[10,5,5,20],[10,10,10,10]])
idx = [np.array([0,1]), np.array([2,1])]
r = hl.qis(idx, [m, m])

print(r)