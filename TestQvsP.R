

# 1. histogram differential of 2x2 solutions

# problem 1
tries <- 100000

x <- c(1000,1000)

h <- rep(NA, tries)

for (i in 1:tries) {
  r <- humanleague::synthPop(list(x,x))$x.hat
  h[i]=r[2,1]-r[1,1]
}

print(paste("mean: ", mean(h)))
print(paste("sd: ", sd(h)))

# hist(h, breaks = 101)

# 2. deviation from multivariate normal distribution

x <- as.integer(dnorm(-50:50,sd=15)*10000)
x2 <- x %*% t(x) / sum(x)
r <- humanleague::synthPop(list(x,x))$x.hat
rms <- sqrt(sum((r-x2)^2))
print(paste("rms: ", rms))

#image(r-x2)
