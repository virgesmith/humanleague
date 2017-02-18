

# 1. histogram p-value for flat 10x10 populations

# problem 1

x <- rep(12,12)

tests=10000

hq = rep(0, tests)
hw = rep(0, tests)

for (i in 1:10000) {

  r = humanleague::synthPop(list(x,x), "iqrs")

  if (r$conv) {
    hq[i] = r$pValue
  }

  r = humanleague::synthPop(list(x,x))
  hw[i] = r$pValue

}
# start = proc.time()
# for (i in 1:tries) {
#   #seed<-matrix(runif(4,0,2), nrow=2)
#   seed<-matrix(rep(2,4), nrow=2)
#   r2 <- mipfp::Ipfp(seed, list(1,2), list(x,x))
#   #h[i]=r$meanSqVariation
#   h2[i] = r2$x.hat[1,1] - r2$x.hat[1,2]
# }
# print(proc.time() - start)


# hist(h, breaks = 101)

# 2. deviation from multivariate normal distribution

#image(r-x2)
