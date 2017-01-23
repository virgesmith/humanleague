
# Comparison of mipfp with Microsim
#library(Microsim)

# desired targets (margins)
target.row <- c(1, 2, 43, 41, 10, 3)
target.col <- c(1, 2, 43, 41, 10, 3)
#target.col <- c(52, 48)
# storing the margins in a list
target.data <- list(target.col, target.row)
# list of dimensions of each marginal constrain
target.list <- list(1, 2)
# calling the mipfp fitting methods
#seed <- array(1, dim=c(length(target.col), length(target.row)))

qipf <- humanleague::synthPop(list(target.col, target.row), 1)
seed <- qipf$x.hat
#stopifnot(qipf$conv)
print("QIPF:")
print(seed)
ipf <- mipfp::Ipfp(seed, target.list, target.data)
print("IPF diff:")
print(ipf$x.hat - seed)
ml <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "ml")
print("ML diff:")
print(ml$x.hat - seed)
chi2 <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "chi2")
print("CHI2 diff:")
print(chi2$x.hat - seed)
lsq <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "lsq")
print("LSQ diff:")
print(lsq$x.hat - seed)
