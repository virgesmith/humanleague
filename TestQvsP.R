

# 1. histogram differential of 2x2 solutions

# problem 1

x <- c(1,2,3,4,5,6,7,8,9,10)

for (i in 1:1000) {

r = humanleague::synthPop(list(x,x))
tr = chisq.test(r$x.hat)
stopifnot(tr$p.value > 0.05)
s = mipfp::Ipfp(matrix(rep(1, length(x)^2), nrow=length(x)), list(1,2), list(x,x))
#ts = chisq.test(s$x.hat)

print(paste(tr$p.value, ts$p.value))

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
