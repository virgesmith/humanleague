
# Problem 1
tries<-100

# 2D
m<-c(1,1,1,5,25,5,1)
m = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m))
  #print(attempts)
  attempts = attempts + res$attempts
}
print(attempts/tries)

# 3D
#m = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m,m))
  #print(res$attempts)
  #if (res$conf)
  attempts = attempts + res$attempts
}
print(attempts/tries)

# 4D
#m = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m,m,m))
  #print(res$attempts)
  #if (res$conf)
  attempts = attempts + res$attempts
}
print(attempts/tries)

# 5D
#m = m * 5
attempts <- 0
for (i in 1:tries) {
  res<-humanleague::synthPop(list(m,m,m,m,m))
  #print(res$attempts)
  #if (res$conf)
  attempts = attempts + res$attempts
}
print(attempts/tries)
