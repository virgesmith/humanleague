#  see http://kbroman.org/pkg_primer/pages/tests.html

context("humanleague")

test_that("marginal sums are invalid", {
  expect_error(humanleague:::synthPop(list(c(10,10),c(10,11))))
})

test_that("dimension invalid", {
  expect_error(humanleague:::synthPop(list(c(10,10))))
})

test_that("marginal has -ve value", {
  expect_error(humanleague:::synthPop(list(c(10,-10),c(10,-10))))
})

m<-c(25,25,25,25,25)
# simple cases where 1 attempt should
test_that("simple 2D", {
  res<-humanleague:::synthPop(list(m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_lt(res$attempts, 4)
})

m = m * 125
test_that("simple 5D", {
  res<-humanleague:::synthPop(list(m,m,m,m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_lt(res$attempts, 4)
})

m = m * 125
test_that("simple 8D", {
  res<-humanleague:::synthPop(list(m,m,m,m,m,m,m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_lt(res$attempts, 4)
})

# need to reduce length of dims to get the 12D case to run in any reasonable time
m = c(2^15,2^15)
test_that("simple 12D", {
  res<-humanleague:::synthPop(list(m,m,m,m,m,m,m,m,m,m,m,m))
  expect_equal(rowSums(res$x.hat), m)
  expect_lt(res$attempts, 4)
})

