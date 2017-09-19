#  see http://kbroman.org/pkg_primer/pages/tests.html

context("humanleague")

# Unit test harness
test_that("unit tests", {
  result = humanleague::unitTest()
  expect_gt(result$nTests, 0)
  expect_equal(result$nFails, 0)
  if (result$nFails) {
    print(result$errors)
  }
})

# Regression tests

test_that("marginal sums are invalid", {
  expect_error(humanleague::synthPop(list(c(10,10),c(10,11))))
})

test_that("dimension invalid", {
  expect_error(humanleague::synthPop(list(c(10,10))))
})


test_that("marginal has -ve value", {
  expect_error(humanleague::synthPop(list(c(10,-10),c(10,-10))))
})

test_that("invalid method", {
  expect_error(humanleague::synthPop(list(c(10,10),c(10,10)),"abcd"))
})

m<-c(25,25,25,25,25)
# simple cases of various dimensions
test_that("simple 2D qiws", {
  res<-humanleague::synthPop(list(m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_equal(res$conv, TRUE)
  expect_equal(nrow(res$pop), 125)
  expect_equal(ncol(res$pop), 2)
  expect_gt(res$pValue, 0.005)
  expect_equal(sum(res$error.margins), 0)
  expect_equal(length(res$error.margins), 2)
})


m = m * 125
test_that("simple 5D qiws", {
  res<-humanleague::synthPop(list(m,m,m,m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_equal(res$conv, TRUE)
  expect_equal(nrow(res$pop), 125^2)
  expect_equal(ncol(res$pop), 5)
  expect_gt(res$pValue, 0.005)
  expect_equal(sum(res$error.margins), 0)
  expect_equal(length(res$error.margins), 5)
})


m = m * 125
test_that("simple 8D qiws", {
  res<-humanleague::synthPop(list(m,m,m,m,m,m,m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_equal(res$conv, TRUE)
  expect_gt(res$pValue, 0.005)
  expect_equal(sum(res$error.margins), 0)
  expect_equal(length(res$error.margins), 8)
})


m = c(2^15,2^15)
test_that("simple 12D qiws", {
  res<-humanleague::synthPop(list(m,m,m,m,m,m,m,m,m,m,m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_equal(res$conv, TRUE)
  expect_gt(res$pValue, 0.005)
  expect_equal(sum(res$error.margins), 0)
  expect_equal(length(res$error.margins), 12)
})


# realistic case (iqrs fails)
m1 <- c(144, 150, 3, 2, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17)
m2 <- c(18, 1, 1, 3, 6, 5, 1, 2, 1, 8, 2, 3, 4, 2, 4, 2, 2, 2, 4, 2, 4, 2, 2, 8, 10, 6, 2, 1, 2, 2, 2, 1, 1, 1, 5, 1, 2, 1, 1, 1, 3, 2, 1, 3, 3, 1, 1, 4, 4, 1, 1, 5, 4, 10, 1, 6, 2, 67, 1, 10, 7, 9, 4, 21, 19, 9, 131, 17, 9, 8, 14, 17, 13, 11, 3, 6, 2, 2, 3, 1, 12, 1, 1, 1, 2, 1, 1, 1, 2, 21, 1, 26, 97, 10, 47, 6, 2, 3, 2, 7, 2, 17, 2, 6, 3, 1, 1, 2, 18, 9, 59, 5, 399, 71, 100, 157, 74, 199, 154, 98, 22, 7, 13, 39, 19, 6, 43, 41, 24, 14, 30, 30, 105, 604, 15, 69, 33, 1, 122, 17, 20, 9, 77, 4, 9, 4, 56, 1, 32, 10, 9, 79, 4, 2, 30, 116, 3, 6, 14, 18, 2, 2, 9, 4, 11, 12, 5, 5, 2, 1, 1, 3, 9, 2, 7, 3, 1, 4, 1, 3, 2, 1, 7, 1, 7, 4, 17, 3, 5, 2, 6, 11, 2, 2, 3, 13, 3, 5, 1, 3, 2, 4, 2, 1, 16, 4, 1, 3, 7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 9, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 330, 28, 281, 12)

test_that("msoa qiws", {
  res<-humanleague::synthPop(list(m1,m2))
  expect_equal(rowSums(res$x.hat), m1)
  expect_equal(colSums(res$x.hat), m2)
  expect_equal(nrow(res$pop), sum(m1))
  expect_equal(ncol(res$pop), 2)
  expect_gt(res$pValue, 0.00)
  expect_equal(sum(res$error.margins), 0)
  expect_equal(res$conv, TRUE)
})

##### IPF

test_that("IPF 2d unity seed", {

  m0=c(52,28,20)
  m1=c(87,13)
  sizes=c(length(m0), length(m1))
  s=array(rep(1,prod(sizes)),sizes)
  t=ipf(s,list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), t$pop)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
})

test_that("IPF 2d nonunity seed", {

  m0=c(52,48)
  m1=c(87,13)

  s2=array(rep(1,4),c(2,2))
  s2[1,1] = 0.7
  t=ipf(s2,list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), t$pop)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
})

test_that("IPF 3d unity seed", {

  m0=c(52,48)
  m1=c(10,77,13)
  m2=c(50,5,40,5)

  sizes = c(length(m0), length(m1), length(m2))

  s3=array(rep(1,prod(sizes)),sizes)
  t3=ipf(s3,list(m0,m1,m2))
  expect_equal(t3$conv, TRUE)
  expect_equal(t3$pop, 100)
  expect_equal(sum(t3$result), t3$pop)
  expect_equal(apply(t3$result, c(1), sum), m0)
  expect_equal(apply(t3$result, c(2), sum), m1)
  expect_equal(apply(t3$result, c(3), sum), m2)

})

test_that("IPF 4d unity seed", {

  m0=c(52,48)
  m1=c(10,77,13)
  m2=c(50,5,40,5)
  m3=c(20,20,20,20,20)

  sizes = c(length(m0), length(m1), length(m2), length(m3))

  s=array(rep(1,prod(sizes)),sizes)
  t=ipf(s,list(m0,m1,m2,m3))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100)
  expect_equal(sum(t$result), t$pop)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
  expect_equal(apply(t$result, c(3), sum), m2)
  expect_equal(apply(t$result, c(4), sum), m3)

})

##### QSIPF

test_that("QSIPF 2d unity seed", {

  m0=c(52,28,20)
  m1=c(87,13)
  sizes=c(length(m0), length(m1))
  s=array(rep(1,prod(sizes)),sizes)
  t=qsipf(s,list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100)
  expect_equal(sum(t$result), t$pop)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
})

test_that("QSIPF 2d nonunity seed", {

  m0=c(52,48)
  m1=c(87,13)

  s2=array(rep(1,4),c(2,2))
  s2[1,1] = 0.7
  t=qsipf(s2,list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100)
  expect_equal(sum(t$result), t$pop)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
})

test_that("QSIPF 3d unity seed", {

  m0=c(10,77,13)
  m1=c(52,48)
  m2=c(50,5,40,5)

  sizes = c(length(m0), length(m1), length(m2))

  s3=array(rep(1,prod(sizes)),sizes)
  t3=qsipf(s3,list(m0,m1,m2))
  expect_equal(t3$conv, TRUE)
  expect_equal(t3$pop, 100)
  expect_equal(sum(t3$result), t3$pop)
  expect_equal(apply(t3$result, c(1), sum), m0)
  expect_equal(apply(t3$result, c(2), sum), m1)
  expect_equal(apply(t3$result, c(3), sum), m2)

})

test_that("QSIPF 4d unity seed", {

  m0=c(52,48)
  m1=c(10,77,13)
  m2=c(20,20,20,20,20)
  m3=c(50,5,40,5)

  sizes = c(length(m0), length(m1), length(m2), length(m3))

  s=array(rep(1,prod(sizes)),sizes)
  t=qsipf(s,list(m0,m1,m2,m3))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100)
  expect_equal(sum(t$result), t$pop)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
  expect_equal(apply(t$result, c(3), sum), m2)
  expect_equal(apply(t$result, c(4), sum), m3)

})


##### constrained

# bedrooms cannot exceed rooms
# assumes rooms={1,2...9+} and bedrooms={0,1...5+}
makeConstraint = function(r, b) {
  p = matrix(rep(T,length(r)*length(b)), nrow=length(r))
  for (i in 1:length(r)) {
    for (j in 1:length(b)) {
      if (j > i + 1)
        p[i,j] = F;
    }
  }
  return(p);
}


test_that("constrained1", {
 r = c(0, 3, 17, 124, 167, 79, 46, 22)
 b = c(0, 15, 165, 238, 33, 7)
 res = humanleague::synthPopG(list(r,b),makeConstraint(r,b))
 expect_equal(res$conv, TRUE)
 expect_equal(rowSums(res$x.hat), r)
 expect_equal(colSums(res$x.hat), b)
})

test_that("constrained2", {
  r = c( 1, 1, 8, 3,84, 21, 4, 4, 1)
  b = c( 0, 8, 3, 113, 2, 1)
  res = humanleague::synthPopG(list(r,b),makeConstraint(r,b))
  expect_equal(res$conv, TRUE)
  expect_equal(rowSums(res$x.hat), r)
  expect_equal(colSums(res$x.hat), b)
})

test_that("constrained3", {
  r = c( 1, 3, 7, 19, 96, 4, 5, 1, 1)
  b = c( 0, 7, 21, 109, 0, 0)
  res = humanleague::synthPopG(list(r,b),makeConstraint(r,b))
  expect_equal(res$conv, TRUE)
  expect_equal(rowSums(res$x.hat), r)
  expect_equal(colSums(res$x.hat), b)
})

test_that("constrained4", {
  r = c( 1, 1, 12, 43, 45, 1, 6, 0, 2)
  b = c( 0, 7, 46, 54, 1, 3)
  res = humanleague::synthPopG(list(r,b),makeConstraint(r,b))
  expect_equal(res$conv, TRUE)
  expect_equal(rowSums(res$x.hat), r)
  expect_equal(colSums(res$x.hat), b)
})

# test_that("constrained5", {
#   r = c(0, 3, 17, 124, 167, 79, 46, 22)
#   b = c(0, 15, 165, 238, 33, 7)
#   res = humanleague::synthPop(list(r,b))
#   expect_equal(res$conv, TRUE)
#   cres = humanleague::constrain(res$x.hat,makeConstraint(r,b))
#   expect_equal(cres$conv, TRUE)
#   expect_equal(rowSums(cres$x.hat), r)
#   expect_equal(colSums(cres$x.hat), b)
# })
#
# test_that("constrained6", {
#   r = c( 1, 1, 8, 3,84, 21, 4, 4, 1)
#   b = c( 0, 8, 3, 113, 2, 1)
#   res = humanleague::synthPop(list(r,b))
#   expect_equal(res$conv, TRUE)
#   cres = humanleague::constrain(res$x.hat,makeConstraint(r,b))
#   expect_equal(cres$conv, TRUE)
#   expect_equal(rowSums(cres$x.hat), r)
#   expect_equal(colSums(cres$x.hat), b)
# })
#
# test_that("constrained7", {
#   r = c( 1, 3, 7, 19, 96, 4, 5, 1, 1)
#   b = c( 0, 7, 21, 109, 0, 0)
#   res = humanleague::synthPop(list(r,b))
#   expect_equal(res$conv, TRUE)
#   cres = humanleague::constrain(res$x.hat,makeConstraint(r,b))
#   expect_equal(cres$conv, TRUE)
#   expect_equal(rowSums(cres$x.hat), r)
#   expect_equal(colSums(cres$x.hat), b)
# })
#
# test_that("constrained8", {
#   r = c( 1, 1, 12, 43, 45, 1, 6, 0, 2)
#   b = c( 0, 7, 46, 54, 1, 3)
#   res = humanleague::synthPop(list(r,b))
#   expect_equal(res$conv, TRUE)
#   cres = humanleague::constrain(res$x.hat,makeConstraint(r,b))
#   expect_equal(cres$conv, TRUE)
#   expect_equal(rowSums(cres$x.hat), r)
#   expect_equal(colSums(cres$x.hat), b)
# })

##### Marginal integerisation tests

test_that("population must be positive", {
  expect_error(humanleague::prob2IntFreq(c(0.5,0.5), -100))
})

test_that("probabilities must sum to zero", {
  expect_error(humanleague::prob2IntFreq(c(0.5,0.4999), 100))
})

test_that("simple1", {
  res<-humanleague::prob2IntFreq(c(0.1,0.2,0.3,0.4), 10)
  expect_equal(res$freq, c(1,2,3,4))
  expect_equal(res$var, 0)
})

test_that("simple2", {
  res<-humanleague::prob2IntFreq(c(0.1,0.2,0.3,0.4), 11)
  expect_equal(res$freq, c(1,2,3,5))
  expect_equal(res$var, 0.125)
})

test_that("degenerate", {
  res<-humanleague::prob2IntFreq(c(0.2,0.2,0.2,0.2,0.2), 11)
  expect_equal(res$freq, c(3,2,2,2,2))
  expect_equal(res$var, 0.16)
})

###### Sobol sequence tests

test_that("sobol 1d", {
  res<-humanleague::sobolSequence(1, 10)
  expect_equal(res, matrix(c(0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875, 0.9375)))
})

test_that("sobol 1d skip 1", {
  # NOTE: actual skips will be largest power of two < skips requested
  # i.e. you need to specify 2 to get 1 skip
  res<-humanleague::sobolSequence(1, 10, 2)
  expect_equal(res, matrix(c(0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875, 0.9375, 0.4375)))
})

test_that("sobol 2d", {
  # NOTE: actual skips will be largest power of two < skips requested
  # i.e. you need to specify 2 to get 1 skip
  res<-humanleague::sobolSequence(2, 5)
  expect_equal(res, matrix(c(0.5, 0.75, 0.25, 0.375, 0.875, 0.5, 0.25, 0.75, 0.375, 0.875), nrow=5))
})

test_that("sobol 4d", {
  # NOTE: actual skips will be largest power of two < skips requested
  # i.e. you need to specify 2 to get 1 skip
  res<-humanleague::sobolSequence(4, 3)
  expect_equal(res, matrix(c(0.5, 0.75, 0.25, 0.5, 0.25, 0.75, 0.5, 0.75, 0.25, 0.5, 0.25, 0.75), nrow=3))
})

test_that("sobol 4d skip", {
  # NOTE: actual skips will be largest power of two < skips requested
  # i.e. you need to specify 2 to get 1 skip
  res<-humanleague::sobolSequence(4, 3, 5)
  expect_equal(res, matrix(c(0.1875, 0.6875, 0.9375, 0.3125, 0.8125, 0.0625, 0.3125, 0.8125, 0.5625, 0.6875, 0.1875, 0.9375), nrow=3))
})

test_that("sobol 2d rho=0", {
  res1=humanleague::sobolSequence(2, 5)
  res2=humanleague::correlatedSobol2Sequence(0.0, 5)
  expect_equal(res1, res2)
})

test_that("sobol 2d rho", {
  rhos=seq(-1,1,0.1)
  for (rho in rhos) {
    #print(rho)
    # correlated uniforms
    u=humanleague::correlatedSobol2Sequence(rho, 65536)
    # compute actual correlation of transformed normals
    rho_actual = cor(qnorm(u))[1,2]
    # check within 0.001 of expected value
    expect_lt(abs(rho - rho_actual), 0.001)
  }
})


