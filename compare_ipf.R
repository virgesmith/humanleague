
library(mipfp)
#library(humanleague)

N = 1000

m1 = c(5,4,6)
m2 = c(3,2,4,1,5)
m3 = c(8, 7)
idx = c(1,2,3)

tol = 1e-8

errorCount = 0

for (i in 1:N)
{
  seed = array(runif(30)+0.5,c(3,5,2))
  cmp = Ipfp(seed, idx, list(m1, m2, m3), tol = tol)
  res = wip_ipf(seed, idx, list(m1, m2, m3))
  if (max(abs(res$result - cmp$x.hat)) > tol)
    errorCount = errorCount + 1
}

print(paste(errorCount, " errors in ", N, "runs"))

#m = c(1,1,1,1,10,1)
#q = wip_qis(c(1,2),list(m,m))
