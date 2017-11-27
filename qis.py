import numpy as np
import humanleague as hl

# # THIS no longer CRASHES
# m = np.array([[10,20,5,5],[10,5,5,20],[10,10,10,10]])
# idx = [np.array([0,1]), np.array([2,1])]
# r = hl.qis(idx, [m, m])

# print(r)
# print(np.sum(r["result"]))
# print(np.sum(r["expectation"]))


# m0 = np.array([52, 48]) 
# m1 = np.array([87, 13])
# m2 = np.array([67, 33])
# m3 = np.array([55, 45])
# i0 = np.array([0])
# i1 = np.array([1])
# i2 = np.array([2])
# i3 = np.array([3])

# for i in range(0,9):
#   p = hl.qis([i0, i1, i2, i3], [m0, m1, m2, m3], i*100)
#   print(p["conv"])
#   print(p["chiSq"]) 
#   print(p["pValue"])

# print(p)

N = 10000

# sobolSequence not leaking
# qis/ipf leak badly
# qisi leaks more slowly (runs more slowly)

m = np.array([[10,20,5,5],[10,5,5,20],[10,10,10,10]],dtype=float)
s = np.ones([3,4,3,3,4,3],dtype=float)
idx = [np.array([0,1]), np.array([2,1]), np.array([3,4]), np.array([5,4])]
for i in range(0,N):
  r = hl.sobolSequence(10,10000, i)
  #r = hl.ipf(s,idx, [m, m, m, m])
  #print(r)


# print(r["conv"])
#print()