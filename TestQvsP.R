

# 1. histogram p-value for flat 10x10 populations

# problem 1

x <- rep(10,10)

tests=10000

pq = rep(0, tests)
pw = rep(0, tests)
cq = rep(0, tests)
cw = rep(0, tests)

for (i in 1:10000) {

  r = humanleague::synthPop(list(x,x), "iqrs")

  if (r$conv) {
    pq[i] = r$pValue
    cq[i] = r$chiSq
  }

  r = humanleague::synthPop(list(x,x))
  pw[i] = r$pValue
  cw[i] = r$chiSq

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
