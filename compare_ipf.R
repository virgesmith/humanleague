
library(mipfp)
#library(humanleague)

N = 1

# use the 1st 5 primes
m1 = array(rep(7,2*3*5),dim=c(2,3,5))
m2 = array(rep(2,3*5*7),dim=c(3,5,7))

idx = list(c(1,2,3),c(2,3,4))

tol = 1e-8

errorCount = 0

# for (i in 1:N) {
  seed = array(runif(2*3*5*7)+1,c(2,3,5,7))
  cmp = Ipfp(seed, idx, list(m1, m2), tol = tol)
  res = ipf(seed, idx, list(m1, m2))
  res2 = qis(idx, list(m1, m2)) #crashes
  res3 = qisi(seed, idx, list(m1, m2))
  if (max(abs(res$result - cmp$x.hat)) > tol)
    errorCount = errorCount + 1
  stopifnot(all.equal(dim(res2$result),dim(cmp$x.hat)))
  stopifnot(all.equal(dim(res3$result),dim(cmp$x.hat)))
# }

print(paste(errorCount, " errors in ", N, "runs"))

# from testhat
m0=array(c(25, 26, 27, 22),c(2,2))
m1=c(67,20,13)
sizes=c(2,2, length(m1))
s=array(rep(1,prod(sizes)),sizes)
t=ipf(s, list(c(1,2),c(3)),list(m0,m1))
stopifnot(t$conv == TRUE)
stopifnot(t$pop == 100.0)
stopifnot(sum(t$result) == 100)
apply(t$result, c(1,2), sum) - m0
apply(t$result, c(3), sum) - m1
