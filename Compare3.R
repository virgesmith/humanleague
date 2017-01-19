
# Comparison of mipfp with Microsim
#library(Microsim)

# desired targets (margins)
target.row <- c(20, 20, 35, 30, 20, 20, 15, 15, 15)
target.col <- c(20, 20, 35, 30, 20, 20, 15, 15, 15)
target.slice <- c(20, 20, 35, 30, 20, 20, 15, 15, 15)
# storing the margins in a list
target.data <- list(target.col, target.row, target.slice)
# list of dimensions of each marginal constraint
target.list <- list(1, 2, 3)
# calling the mipfp fitting methods
seed <- array(1, dim=c(length(target.col), length(target.row), length(target.slice)))

t <- proc.time()
ipf <- mipfp::Ipfp(seed, target.list, target.data)
print("ipf")
print(proc.time() - t)
#print(ipf)
t <- proc.time()

ml <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "ml")
print("ml")
print(proc.time() - t)
#print(ml)
t <- proc.time()

chi2 <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "chi2")
print("chi2")
print(proc.time() - t)
#print(chi2)
t <- proc.time()

lsq <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = "lsq")
print("lsq")
print(proc.time() - t)
#print(lsq)
t <- proc.time()

qrpf <- Microsim::pop3(target.col, target.row, target.slice, 10000)
print("qrpf")
print(proc.time() - t)
#print(qrpf)
t <- proc.time()

