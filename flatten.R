# 1D+1D
m=c(1, 19)
n=c(2, 5, 13)

ms=qis(list(1,2),list(m,n))

stopifnot(ms$conv)

a=ms$result
colnames = c("M", "N")
t=flatten(ms$result, colnames)

stopifnot(all.equal(names(t), colnames))
stopifnot(sum(a) == nrow(t))

stopifnot(length(unique(t$M)) == length(m))
stopifnot(length(unique(t$N)) == length(n))

# check row sums match marginals
for (i in 1:length(m)) {
  print(paste(nrow(t[t$M==i,]), m[i]))
}
for (i in 1:length(n)) {
  print(paste(nrow(t[t$N==i,]), n[i]))
}

# 1D+2D
m=c(101, 99, 103, 97, 200)
n=array(c(105, 95, 107, 93, 109, 91), dim=c(3,2))

ms=qis(list(1,c(2,3)),list(m,n))

stopifnot(ms$conv)

a=ms$result
colnames = c("M", "N1", "N2")
t=flatten(ms$result, colnames)

stopifnot(sum(a) == nrow(t))

stopifnot(length(unique(t$M)) == 5)
stopifnot(length(unique(t$N1)) == 3)
stopifnot(length(unique(t$N2)) == 2)

# check row sums match marginals
for (i in 1:5) {
  stopifnot(nrow(t[t$M==i,]) == m[i])
}

for (i in 1:3) {
  for (j in 1:2) {
    stopifnot(nrow(t[t$N1==i & t$N2==j,]) == n[i, j])
  }
}

