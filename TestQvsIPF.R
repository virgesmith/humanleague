
x <- rep(10,10)

r = humanleague::synthPop(list(x,x))
while (r$pValue > 0.005) {
  r = humanleague::synthPop(list(x,x))
}

print(r$pValue)

r2 = mipfp::Ipfp(r$x.hat, list(1,2), list(x,x))

print(r2$x.hat - r$x.hat)
print(r2$conv)
print(chisq.test(r2$x.hat))
