#!/usr/bin/env python3

import numpy as np
import humanleague as hl

m0 = np.array([52.0, 48.0])
m1 = np.array([20, 67.0, 13.0])
m2 = np.array([15, 40.0, 35.0, 10.0])

i2 = [np.array([0]),np.array([1])]
i3 = [np.array([0]),np.array([1]),np.array([2])]

s2 = np.ones([len(m0), len(m1)])
r = hl.ipf(s2, [m0, m1])
print(r)
print(np.sum(r["result"], 0) - m1)
print(np.sum(r["result"], 1) - m0)

s3 = np.ones([len(m0), len(m1), len(m2)])
r = hl.ipf(s3, [m0, m1, m2])
print(r)
print(np.sum(r["result"], (0, 1)) - m2)
print(np.sum(r["result"], (1, 2)) - m0)
print(np.sum(r["result"], (2, 0)) - m1)

#s = np.ones([len(m0), len(m1)])
p = hl.wip_ipf(i2, [m0, m1])
print(p)
print(np.sum(p["result"], 0) - m1)
print(np.sum(p["result"], 1) - m0)

p = hl.wip_ipf(i3, [m0, m1, m2])
#print(r)
print(p)
print(np.sum(p["result"], (0, 1)) - m2)
print(np.sum(p["result"], (1, 2)) - m0)
print(np.sum(p["result"], (2, 0)) - m1)
print(np.array_equal(p["result"],r["result"]))

m2d = np.array([[1.,2.,3.],[2.,2.,2.]])
m1d = np.array([3.,4.,2.,3.])
i2d = [np.array([0,1]), np.array([2])]
p = hl.wip_ipf(i2d, [m2d, m1d])
#print(r)
print(p)

m0=np.array([[25.,26.], [27., 22.]])
m1=np.array([67.,20.,13.])
t=hl.wip_ipf([np.array([0,1]),np.array([2])], [m0,m1])
print(t)
