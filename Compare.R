
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
seed <- array(1, dim=c(length(target.col), length(target.row)))
ipf <- mipfp::Ipfp(seed, target.list, target.data)
print(ipf)
ml <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "ml")
print(ml)
chi2 <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "chi2")
print(chi2)
lsq <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "lsq")
print(lsq)
usim <- Microsim::pop(target.col, target.row, 100)
print(usim)
