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

test_that("dimension indices invalid (missing)", {
  expect_error(humanleague::qis(list(1,3), list(c(10,10),c(10,10))))
})

test_that("dimension indices invalid (only one)", {
  expect_error(humanleague::qis(list(1,1), list(c(10,10),c(10,10))))
})

# 0/-ve dimension values?

test_that("marginal sums are invalid", {
  expect_error(humanleague::qis(list(1,2), list(c(10,10),c(10,11))))
})

test_that("dimension invalid", {
  expect_error(humanleague::qis(list(1),list(c(10,10))))
})

test_that("marginal has -ve value", {
  expect_error(humanleague::qis(list(1,2),list(c(10,-10),c(10,-10))))
})

test_that("seed has -ve value", {
  expect_error(humanleague::ipf(array(rep(-1,4),dim=c(2,2)), list(1,2),list(c(10,10),c(10,10))))
  expect_error(humanleague::qisi(array(rep(-1,4),dim=c(2,2)), list(1,2),list(c(10,10),c(10,10))))
})

test_that("invalid method", {
  expect_error(humanleague::qis(list(c(10,10),c(10,10)),"abcd"))
})

# dimension of seed and marginals inconsistent
test_that("inconsistent dims", {
  seed=array(rep(1,2*5*3),dim=c(2,5,3))
  expect_error(humanleague::ipf(seed, list(1,c(2,3)),list(c(15,15),array(rep(2,15),dim=c(3,5)))))
  expect_error(humanleague::qisi(seed, list(1,c(2,3)),list(c(15,15),array(rep(2,15),dim=c(3,5)))))
})


m<-c(25,25,25,25,25)
# simple cases of various dimensions
test_that("simple 2D qis", {
  res<-humanleague::qis(list(1,2),list(m,m))
  expect_equal(rowSums(res$result), m)
  expect_equal(res$conv, TRUE)
  colnames = c("A","B")
  table = flatten(res$result, colnames)
  expect_true(all.equal(names(table), colnames))
  expect_equal(nrow(table), 125)
  expect_equal(ncol(table), 2)
  expect_gt(res$pValue, 0.99)
})


m = m * 125
test_that("simple 5D qis", {
  res<-humanleague::qis(list(1,2,3,4,5),list(m,m,m,m,m))
  expect_equal(rowSums(res$result), m)
  expect_equal(res$conv, TRUE)
  colnames = c("A","B","C","D","E")
  table = flatten(res$result, colnames)
  expect_true(all.equal(names(table), colnames))
  expect_equal(nrow(table), 125^2)
  expect_equal(ncol(table), 5)
  expect_gt(res$pValue, 0.99)
})


# m = m * 125
# test_that("simple 8D qiws", {
#   res<-humanleague::qis(list(1,2,3,4,5,6,7,8),list(m,m,m,m,m,m,m,m))
#   expect_equal(rowSums(res$result), m)
#   expect_equal(res$conv, TRUE)
#   colnames = c("A","B","C","D","E","F","G","H")
#   table = flatten(res$result, colnames)
#   expect_true(all.equal(names(table), colnames))
#   expect_equal(nrow(table), 125^3)
#   expect_equal(ncol(table), 8)
#   expect_gt(res$pValue, 0.99)
# })
#
#
# m = c(2^15,2^15)
# test_that("simple 12D qiws", {
#   res<-humanleague::qis(list(1,2,3,4,5,6,7,8,9,10,11,12),list(m,m,m,m,m,m,m,m,m,m,m,m))
#   expect_equal(rowSums(res$result), m)
#   expect_equal(res$conv, TRUE)
#   expect_gt(res$pValue, 0.03)
# })

m = array(c(2^14,2^14,2^14,2^14),c(2,2))
test_that("Complex 8 x 2D -> 12D qiws", {
  res<-humanleague::qis(list(c(1,2),c(2,3),c(4,5),c(5,6),c(6,7),c(8,9),c(9,10),c(11,12)),list(m,m,m,m,m,m,m,m))
  expect_equal(rowSums(res$result), rowSums(m))
  expect_equal(res$conv, TRUE)
  expect_gt(res$pValue, 0.02)
})

# TODO add multidim tests to compare QIS expectation to IPF result with unity seed
test_that("QIS expectation", {
  m0 = c(52, 28,20)
  m1 = c(87, 13)
  m2 = c(67, 33)
  m3 = c(55, 45)

  p = qis(list(1,2,3,4), list(m0, m1, m2, m3))
  # TODO dim checking appears not to work...
  s=array(rep(1,24),dim=c(3,2,2,2))
  z=ipf(s, list(1,2,3,4), list(m0, m1, m2, m3))
  expect_equal(p$expectation, z$result, 1e-8)
})

# realistic? case (iqrs fails)
m1 <- c(144, 150, 3, 2, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17)
m2 <- c(18, 1, 1, 3, 6, 5, 1, 2, 1, 8, 2, 3, 4, 2, 4, 2, 2, 2, 4, 2, 4, 2, 2, 8, 10, 6, 2, 1, 2, 2, 2, 1, 1, 1, 5, 1, 2, 1, 1, 1, 3, 2, 1, 3, 3, 1, 1, 4, 4, 1, 1, 5, 4, 10, 1, 6, 2, 67, 1, 10, 7, 9, 4, 21, 19, 9, 131, 17, 9, 8, 14, 17, 13, 11, 3, 6, 2, 2, 3, 1, 12, 1, 1, 1, 2, 1, 1, 1, 2, 21, 1, 26, 97, 10, 47, 6, 2, 3, 2, 7, 2, 17, 2, 6, 3, 1, 1, 2, 18, 9, 59, 5, 399, 71, 100, 157, 74, 199, 154, 98, 22, 7, 13, 39, 19, 6, 43, 41, 24, 14, 30, 30, 105, 604, 15, 69, 33, 1, 122, 17, 20, 9, 77, 4, 9, 4, 56, 1, 32, 10, 9, 79, 4, 2, 30, 116, 3, 6, 14, 18, 2, 2, 9, 4, 11, 12, 5, 5, 2, 1, 1, 3, 9, 2, 7, 3, 1, 4, 1, 3, 2, 1, 7, 1, 7, 4, 17, 3, 5, 2, 6, 11, 2, 2, 3, 13, 3, 5, 1, 3, 2, 4, 2, 1, 16, 4, 1, 3, 7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 9, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 330, 28, 281, 12)

test_that("msoa qiws", {
  res<-humanleague::qis(list(1,2),list(m1,m2))
  expect_equal(rowSums(res$result), m1)
  expect_equal(colSums(res$result), m2)
  table = flatten(res$result, c("A","B"))
  expect_equal(nrow(table), sum(m1))
  expect_equal(ncol(table), 2)
  expect_gt(res$pValue, 0.00)
#  expect_equal(sum(res$error.margins), 0)
  expect_equal(res$conv, TRUE)
})

##### IPF

test_that("IPF 2d unity seed", {

  m0=c(52,28,20)
  m1=c(87,13)
  sizes=c(length(m0), length(m1))
  s=array(rep(1,prod(sizes)),sizes)
  t=ipf(s,c(1,2),list(m0,m1))
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
  t=ipf(s2,c(1,2),list(m0,m1))
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
  t3=ipf(s3,c(1,2,3),list(m0,m1,m2))
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
  t=ipf(s,c(1,2,3,4),list(m0,m1,m2,m3))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100)
  expect_equal(sum(t$result), t$pop)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
  expect_equal(apply(t$result, c(3), sum), m2)
  expect_equal(apply(t$result, c(4), sum), m3)

})


test_that("MIPF 2d", {

  m0=c(52,48)
  m1=c(67,20,13)
  sizes=c(length(m0), length(m1))
  s=array(rep(1,prod(sizes)),sizes)
  t=ipf(s, list(c(1),c(2)),list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), 100)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
})

test_that("MIPF 3d", {
  m0=array(c(25, 26, 27, 22),c(2,2))
  m1=c(67,20,13)
  sizes=c(2,2, length(m1))
  s=array(rep(1,prod(sizes)),sizes)
  t=ipf(s, list(c(1,2),c(3)),list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), 100)
  expect_equal(apply(t$result, c(1,2), sum), m0)
  expect_equal(apply(t$result, c(3), sum), m1)
})

test_that("MIPF 3d (2)", {
  m0=array(c(20, 20, 11, 17, 12, 20),c(3,2))
  m1=c(33,34,20,13)
  sizes=c(3,2,length(m1))
  s=array(rep(1,prod(sizes)),sizes)
  t=ipf(s, list(c(1,2),c(3)),list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), 100)
  expect_equal(apply(t$result, c(1,2), sum), m0)
  expect_equal(apply(t$result, c(3), sum), m1)
})

# QIS

test_that("QIS 2d", {

  m0=c(52,48)
  m1=c(67,20,13)
  t=qis(list(c(1),c(2)),list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), 100)
  expect_equal(apply(t$result, c(1), sum), m0)
  expect_equal(apply(t$result, c(2), sum), m1)
  expect_lt(t$chiSq, 0.6)
  expect_gt(t$pValue, 0.7)
  chi2 = chisq.test(t$result, t$expectation)
  expect_equal(t$chiSq, as.numeric(chi2$statistic)) # as.numeric required as its a labelled value
  expect_equal(chisq.test(t$result, t$expectation)$p.value, t$pValue)
  #expect_lt(t$degeneracy, ?)
})

test_that("QIS 3d", {
  m0=array(c(25, 26, 27, 22),c(2,2))
  m1=c(67,20,13)
  t=qis(list(c(1,2),c(3)),list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), 100)
  # TODO fix dimension indices
  expect_equal(apply(t$result, c(1,2), sum), m0)
  expect_equal(apply(t$result, c(3), sum), m1)
  expect_lt(t$chiSq, 2.2)
  expect_gt(t$pValue, 0.3)
  # Chi-squared approximation may be incorrect
  # chi2 = chisq.test(t$result, t$expectation)
  # expect_equal(t$chiSq, as.numeric(chi2$statistic)) # as.numeric required as its a labelled value
  # expect_equal(chisq.test(t$result, t$expectation)$p.value, t$pValue)
  #expect_lt(t$degeneracy, ?)
})

test_that("QIS 3d (2)", {
  m0=array(c(20, 20, 11, 17, 12, 20),c(3,2))
  m1=c(33,34,20,13)
  t=qis(list(c(1,2),c(3)),list(m0,m1))
  expect_equal(t$conv, TRUE)
  expect_equal(t$pop, 100.0)
  expect_equal(sum(t$result), 100)
  expect_equal(apply(t$result, c(1,2), sum), m0)
  expect_equal(apply(t$result, c(3), sum), m1)
  expect_lt(t$chiSq,7.0)
  expect_gt(t$pValue, 0.3)

  # check expectation matches IPF
  s=t$result * 0 + 1
  x=ipf(s,list(c(1,2),c(3)),list(m0,m1))
  expect_lt(max(t$expectation-x$result), 1e-8)
  expect_gt(min(t$expectation-x$result), -1e-8)

  # Chi-squared approximation may be incorrect
  # chi2 = chisq.test(t$result, t$expectation, simulate.p.value=T)
  # expect_equal(t$chiSq, as.numeric(chi2$statistic)) # as.numeric required as its a labelled value
  # expect_equal(chisq.test(t$result, t$expectation)$p.value, t$pValue)
  #expect_lt(t$degeneracy, ?)
})

test_that("QIS dimension indexing test", {

  # tricky array indexing - 1st dimension of d0 already sampled, remaining dimension
  # indices on slice of d0 need to be remapped

  d0 = c(4,6,4,4)
  d1 = c(4,4,4)

  m0 = array(rep(prod(d1),prod(d0)),dim=d0)
  m1 = array(rep(prod(d0),prod(d1)),dim=d1)

  ms=qis(list(c(1,2,3,4),c(1,5,6)), list(m0,m1))
  expect_true(ms$conv)
})

test_that("QIS degeneracy tests", {

  # 2D
  stateOcc = 8
  statesPerDim = 8
  m=rep(stateOcc, statesPerDim)
  ms=qis(list(1,2),list(m,m),stateOcc^2)
  expect_true(ms$conv)
  expect_lte(max(ms$result - ms$expectation), 1)
  expect_gte(min(ms$result - ms$expectation),-1)

  # 3D (1D+2D)
  stateOcc = 8
  statesPerDim = 8
  m=rep(stateOcc * stateOcc, statesPerDim)
  n=array(rep(stateOcc, statesPerDim * statesPerDim), dim=c(statesPerDim, statesPerDim))
  ms=qis(list(1,c(2,3)),list(m,n), stateOcc^3)
  expect_true(ms$conv)
  # TODO why is this not lower?
  expect_lte(max(ms$result - ms$expectation), 3)
  expect_gte(min(ms$result - ms$expectation), -1)

  # 4D (overlapping 2D+2D+2D)
  stateOcc = 8
  statesPerDim = 8
  n=array(rep(stateOcc^2, statesPerDim * statesPerDim), dim=c(statesPerDim, statesPerDim))
  ms=qis(list(c(1,2),c(2,3),c(3,4)),list(n,n,n), stateOcc^4)
  expect_true(ms$conv)
  # TODO why is this not lower?
  expect_lte(max(ms$result - ms$expectation), 3)
  expect_gte(min(ms$result - ms$expectation), -1)

})


test_that("QISI degeneracy tests", {

  # 2D
  stateOcc = 8
  statesPerDim = 8
  m=rep(stateOcc, statesPerDim)
  s=array(rep(stateOcc, statesPerDim^2),dim=c(statesPerDim, statesPerDim))
  ms=qisi(s, list(1,2),list(m,m),stateOcc^2)
  expect_true(ms$conv)
  expect_lte(max(ms$result - ms$expectation), 0)
  expect_gte(min(ms$result - ms$expectation),-0)

  # 3D (1D+2D)
  stateOcc = 8
  statesPerDim = 8
  m=rep(stateOcc * stateOcc, statesPerDim)
  n=array(rep(stateOcc, statesPerDim * statesPerDim), dim=c(statesPerDim, statesPerDim))
  s=array(rep(stateOcc, statesPerDim^3),dim=c(statesPerDim, statesPerDim, statesPerDim))
  ms=qisi(s, list(1,c(2,3)),list(m,n), stateOcc^3)
  expect_true(ms$conv)
  expect_lte(max(ms$result - ms$expectation), 1)
  expect_gte(min(ms$result - ms$expectation), -1)

  # 4D (overlapping 2D+2D+2D)
  stateOcc = 8
  statesPerDim = 8
  n=array(rep(stateOcc^2, statesPerDim * statesPerDim), dim=c(statesPerDim, statesPerDim))
  s=array(rep(stateOcc, statesPerDim^4),dim=c(statesPerDim, statesPerDim, statesPerDim, statesPerDim))
  ms=qisi(s, list(c(1,2),c(2,3),c(3,4)),list(n,n,n), stateOcc^4)
  expect_true(ms$conv)
  expect_lte(max(ms$result - ms$expectation), 1)
  expect_gte(min(ms$result - ms$expectation), -1)

})


test_that("QIS listify tests", {

  # 1D+1D
  m=c(101, 99, 103, 97, 200)
  n=c(105, 95, 107, 93, 109, 91)

  ms=qis(list(1,2),list(m,n))

  expect_true(ms$conv)

  a=ms$result
  colnames = c("M", "N")
  t=flatten(ms$result, colnames)

  expect_true(all.equal(names(t), colnames))
  expect_equal(sum(a), nrow(t))

  expect_equal(length(unique(t$M)), 5)
  expect_equal(length(unique(t$N)), 6)

  # check row sums match marginals
  for (i in 1:length(m)) {
    expect_equal(nrow(t[t$M==i,]), m[i])
  }
  for (i in 1:length(n)) {
    expect_equal(nrow(t[t$N==i,]), n[i])
  }

  # 1D+2D
  m=c(101, 99, 103, 97, 200)
  n=array(c(105, 95, 107, 93, 109, 91), dim=c(3,2))

  ms=qis(list(1,c(2,3)),list(m,n))

  expect_true(ms$conv)

  a=ms$result
  colnames = c("M", "N1", "N2")
  t=flatten(ms$result, colnames)

  expect_equal(sum(a), nrow(t))

  expect_equal(length(unique(t$M)), 5)
  expect_equal(length(unique(t$N1)), 3)
  expect_equal(length(unique(t$N2)), 2)

  # check row sums match marginals
  for (i in 1:5) {
    expect_equal(nrow(t[t$M==i,]), m[i])
  }

  for (i in 1:3) {
    for (j in 1:2) {
      expect_equal(nrow(t[t$N1==i & t$N2==j,]), n[i,j])
    }
  }

})

##### QIS-IPF

test_that("QIS-IPF 2d unity seed", {

  m0=c(52,28,20)
  m1=c(87,13)
  sizes=c(length(m0), length(m1))
  s=array(rep(1,prod(sizes)),sizes)
  t=qisi(s,list(1,2),list(m0,m1))
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
  t=qisi(s2,list(1,2),list(m0,m1))
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
  t3=qisi(s3,list(1,2,3),list(m0,m1,m2))
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
  t=qisi(s,list(1,2,3,4),list(m0,m1,m2,m3))
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


##### Marginal/population integerisation tests

test_that("integerise non integral population", {
  r=array(c(1.1, 1, 1, 1), dim=c(2,2))

  expect_error(humanleague::integerise(r))
})

test_that("integerise non integral marginal", {
  r=array(c(1.1, 0.9, 1, 1), dim=c(2,2))

  expect_error(humanleague::integerise(r))
})

test_that("integerise no-op", {
  r=array(c(1, 1, 1, 1), dim=c(2,2))
  res = humanleague::integerise(r)
  expect_true(res$conv)
  expect_equal(res$rmse, 0.0)
})

test_that("integerise", {
  m0 = c(111,112,113,114,110)
  m1 = c(136,142,143,139)
  s = array(rep(1, length(m0)*length(m1)),  dim=c(length(m0),length(m1)))

  fpop = humanleague::ipf(s, list(1,2),list(m0,m1))$result
  #print(fpop)

  result = humanleague::integerise(fpop)
  #print(result$result)
  expect_true(result$conv)
  expect_equal(sum(result$result), sum(m0))
  expect_lt(result$rmse, 0.8772)
})

test_that("population must be positive", {
  expect_error(humanleague::prob2IntFreq(c(0.5,0.5), -100))
})

test_that("probabilities must sum to zero", {
  expect_error(humanleague::prob2IntFreq(c(0.5,0.4999), 100))
})

test_that("population zero returns a zero array (and zero var)", {
  res = humanleague::prob2IntFreq(c(0.5,0.4,0.1), 0)
  expect_equal(res$rmse, 0.0)
  expect_equal(length(res$freq), 3)
  expect_equal(unique(res$freq), 0)
})

test_that("simple1", {
  res<-humanleague::prob2IntFreq(c(0.1,0.2,0.3,0.4), 10)
  expect_equal(res$freq, c(1,2,3,4))
  expect_equal(res$rmse, 0)
})

test_that("simple2", {
  res<-humanleague::prob2IntFreq(c(0.1,0.2,0.3,0.4), 11)
  expect_equal(res$freq, c(1,2,3,5))
  expect_equal(res$rmse, sqrt(0.125))
})

test_that("degenerate", {
  res<-humanleague::prob2IntFreq(c(0.2,0.2,0.2,0.2,0.2), 11)
  expect_equal(res$freq, c(3,2,2,2,2))
  expect_equal(res$rmse, 0.4)
})

###### Sobol sequence tests

test_that("sobol 1d", {
  res<-humanleague::sobolSequence(1, 10)
  expect_equal(res, matrix(c(0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875, 0.9375)))
})

test_that("sobol 1d skip 1", {
  # NOTE: actual skips will be largest power of two < skips requested
  res<-humanleague::sobolSequence(1, 10, 2)
  expect_equal(res, matrix(c(0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875, 0.9375, 0.4375)))
})

test_that("sobol 2d", {
  res<-humanleague::sobolSequence(2, 5)
  expect_equal(res, matrix(c(0.5, 0.75, 0.25, 0.375, 0.875, 0.5, 0.25, 0.75, 0.375, 0.875), nrow=5))
})

test_that("sobol 4d", {
  res<-humanleague::sobolSequence(4, 3)
  expect_equal(res, matrix(c(0.5, 0.75, 0.25, 0.5, 0.25, 0.75, 0.5, 0.75, 0.25, 0.5, 0.25, 0.75), nrow=3))
})

test_that("sobol 4d skip", {
  # NOTE: actual skips will be largest power of two < skips requested
  # i.e. 5 will skip 4
  res<-humanleague::sobolSequence(4, 3, 5)
  expect_equal(res, matrix(c(0.875, 0.625, 0.125, 0.875, 0.125, 0.625, 0.125, 0.375, 0.875, 0.625, 0.375, 0.875), nrow=3))
})

# test_that("sobol 2d rho=0", {
#   res1=humanleague::sobolSequence(2, 5)
#   res2=humanleague::correlatedSobol2Sequence(0.0, 5)
#   expect_equal(res1, res2)
# })

# test_that("sobol 2d rho", {
#   rhos=seq(-1,1,0.1)
#   for (rho in rhos) {
#     #print(rho)
#     # correlated uniforms
#     u=humanleague::correlatedSobol2Sequence(rho, 65536)
#     # compute actual correlation of transformed normals
#     rho_actual = cor(qnorm(u))[1,2]
#     # check within 0.001 of expected value
#     expect_lt(abs(rho - rho_actual), 0.001)
#   }
# })


