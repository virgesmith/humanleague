
# Problem 1
m1 <- c(144, 150, 3, 2, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17)
m2 <- c(18, 1, 1, 3, 6, 5, 1, 2, 1, 8, 2, 3, 4, 2, 4, 2, 2, 2, 4, 2, 4, 2, 2, 8, 10, 6, 2, 1, 2, 2, 2, 1, 1, 1, 5, 1, 2, 1, 1, 1, 3, 2, 1, 3, 3, 1, 1, 4, 4, 1, 1, 5, 4, 10, 1, 6, 2, 67, 1, 10, 7, 9, 4, 21, 19, 9, 131, 17, 9, 8, 14, 17, 13, 11, 3, 6, 2, 2, 3, 1, 12, 1, 1, 1, 2, 1, 1, 1, 2, 21, 1, 26, 97, 10, 47, 6, 2, 3, 2, 7, 2, 17, 2, 6, 3, 1, 1, 2, 18, 9, 59, 5, 399, 71, 100, 157, 74, 199, 154, 98, 22, 7, 13, 39, 19, 6, 43, 41, 24, 14, 30, 30, 105, 604, 15, 69, 33, 1, 122, 17, 20, 9, 77, 4, 9, 4, 56, 1, 32, 10, 9, 79, 4, 2, 30, 116, 3, 6, 14, 18, 2, 2, 9, 4, 11, 12, 5, 5, 2, 1, 1, 3, 9, 2, 7, 3, 1, 4, 1, 3, 2, 1, 7, 1, 7, 4, 17, 3, 5, 2, 6, 11, 2, 2, 3, 13, 3, 5, 1, 3, 2, 4, 2, 1, 16, 4, 1, 3, 7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 9, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 330, 28, 281, 12)

# timing (elapsed)
seed<-array(rep(1,length(m1)*length(m2)),dim=c(length(m1),length(m2)))
print(paste("QIPF: ", system.time(qipf <- humanleague::synthPop(list(m1,m2)))[3]))
print(paste("IPFP: ", system.time(ipfp <- mipfp::Ipfp(seed, list(1,2), list(m1,m2)))[3]))
# TOO SLOW print(paste("ML: ", system.time(ipfp <- mipfp::ObtainModelEstimates(seed, list(1,2), list(m1,m2), method="ml"))[3]))
# TOO SLOW print(paste("CHI2: ", system.time(ipfp <- mipfp::ObtainModelEstimates(seed, list(1,2), list(m1,m2), method="chi2"))[3]))
# TOO SLOW print(paste("LSQ: ", system.time(ipfp <- mipfp::ObtainModelEstimates(seed, list(1,2), list(m1,m2), method="lsq"))[3]))


# Problem 2 - bivariate normal degeneracy comparison
x <- as.integer(dnorm(-50:50,sd=15)*10000)
x2 <- x %*% t(x) / sum(x)
r <- humanleague::synthPop(list(x,x))$x.hat
q <-r
rms <- sqrt(sum((r-x2)^2))
print(paste("rms QIPF: ", rms))

seed<-array(rep(1,length(x)*length(x)),dim=c(length(x),length(x)))
r <- mipfp::Ipfp(seed, list(1,2), list(x,x))$x.hat
rms <- sqrt(sum((r-x2)^2))
print(paste("rms IPFP: ", rms))

# Problem 3 timing (elapsed) with smaller marginals
# x <- as.integer(dnorm(-15:15,sd=5)*10000)
# print(paste("QIPF (s): ", system.time(qipf <- humanleague::synthPop(list(x,x)))[3]))
# seed<-array(rep(1,length(x)*length(x)),dim=c(length(x),length(x)))
# print(paste("IPFP (s): ", system.time(ipfp <- mipfp::Ipfp(seed, list(1,2), list(x,x)))[3]))
# print(paste("ML (s): ", system.time(ipfp <- mipfp::ObtainModelEstimates(seed, list(1,2), list(x,x), method="ml"))[3]))
# print(paste("CHI2 (s): ", system.time(ipfp <- mipfp::ObtainModelEstimates(seed, list(1,2), list(x,x), method="chi2"))[3]))
# print(paste("LSQ (s): ", system.time(ipfp <- mipfp::ObtainModelEstimates(seed, list(1,2), list(x,x), method="lsq"))[3]))

