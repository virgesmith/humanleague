
# Problem 1
tries<-1

maxAttempts=10000

# 2D
m<-c(1,5,25,5,1)
m = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m),maxAttempts)
  seed<-array(rep(1,25),dim=c(5,5))
  res2<-mipfp::Ipfp(seed, list(1,2), list(m,m))
  #print(attempts)
  attempts = attempts + res$attempts
}
print(attempts/tries)

# 3D
m = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m,m),maxAttempts)
  seed<-array(rep(1,125),dim=c(5,5,5))
  res2<-mipfp::Ipfp(seed, list(1,2), list(m,m))
  #print(res$x.hat - res2$x.hat)
  #if (res$conf)
  attempts = attempts + res$attempts
}
print(attempts/tries)

# 4D
m = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m,m,m),maxAttempts)
  #print(res$attempts)
  #if (res$conf)
  attempts = attempts + res$attempts
}
print(attempts/tries)

# 5D
rem = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m,m,m,m),maxAttempts)
  seed<-array(rep(1,5^5),dim=rep(5,5))
  res2<-mipfp::Ipfp(seed, list(1,2,3,4,5), list(m,m,m,m,m))
  #print(res$attempts)
  #if (res$conf)
  attempts = attempts + res$attempts
}
print(attempts/tries)
