
# Welcome to Not-so-simple-world, pop. 1000
#
# Simple world assumptions:
# - one person per household

# Population generation method: one of ipfp, ml, chi2, lsq, qipf
popMethod = "qipf"

library(dplyr)
library(humanleague)

getPop2 <- function(popMethod, x, y) {
  if (popMethod != "qipf") {
    target.m1 <- x
    target.m2 <- y
    # storing the margins in a list
    target.data <- list(target.m1, target.m2)
    # list of dimensions of each marginal constraint
    target.list <- list(1, 2)
    # calling the mipfp fitting methods

    # initial guess is average pop per state
    seed <- array(sum(x)/(length(x)*length(y)), dim=c(length(target.m1), length(target.m2)))

    if (popMethod == "ipfp") {
      pop <- mipfp::Ipfp(seed, target.list, target.data)
    } else {
      pop <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = popMethod)
    }
  } else {
    pop <- humanleague::synthPop2(x, y, 4000)
  }
  return(pop);
}

getPop3 <- function(t, x, y, z) {
  if (popMethod != "qipf") {
    target.m1 <- x
    target.m2 <- y
    target.m3 <- z
    # storing the margins in a list
    target.data <- list(target.m1, target.m2, target.m3)
    # list of dimensions of each marginal constraint
    target.list <- list(1, 2, 3)
    # calling the mipfp fitting methods

    # initial guess is average pop per state
    seed <- array(sum(x)/(length(x)*length(y)*length(z)), dim=c(length(target.m1), length(target.m2), length(target.m3)))

    if (popMethod == "ipfp") {
      pop <- mipfp::Ipfp(seed, target.list, target.data)
    } else {
      pop <- mipfp::ObtainModelEstimates(seed, target.list, target.data, method = popMethod)
    }
  } else {
    # large maxSamples until fix adjust postion
    pop <- humanleague::synthPop3(x, y, z, 1000)
  }
  return(pop);
}


# Zones
# 1 Rural
# 2 Upmarket residential
# 3 Residential
# 4 Multiethnic residential
# 5 Business/retail
# 6 Residential
# 7 Industrial
# 8 Council housing

# 1 2 3
# 4 5 6
# 7 8 9
pzones <- c(40, 90, 150, 180, 6, 190, 100, 4, 240)

# 50-50 gender split
gender <- c(500, 500)

# Age groups
# 1 0-17
# 2 18-25
# 3 36-45
# 4 46-55
# 5 56-65
# 6 65
age <- c(200, 200, 200, 180, 150, 70 )

populace <- getPop3(popMethod, age, gender, pzones)
colnames(populace) <- c("agegroup", "gender", "zone")

# consistency check
stopifnot(nrow(filter(populace, zone == 0)) == pzones[1])
stopifnot(nrow(filter(populace, agegroup == 0)) == age[1])
stopifnot(nrow(filter(populace, gender == 0)) == gender[1])

nPeople = sum(pzones)
nStates = length(pzones) * length(gender) * length(age)

# Households per zone (250)
hzones <- c(10, 22, 38, 45, 2, 47, 25, 1, 60)
# Household sizes: 1, 2, 3, 4, 5-6, 7+
hsize <- c(25, 45, 55, 60, 45, 20 )

# generate O (houses) microsim per zone
hpop <- getPop2(popMethod, hsize, hzones)
colnames(hpop) = c("size", "zone")

# consistency check
stopifnot(nrow(filter(hpop, zone == 0)) == hzones[1])
stopifnot(nrow(filter(hpop, size == 0)) == hsize[1])


# Workplaces per zone (50)
wzones <- c(1, 2, 3, 4, 20, 2, 15, 3)
# Workplace sizes: 1-10, 11-50, 50-200, 201+
wsize <- c(12, 13, 13, 12)

# generate D (work, shops, social) microsim per zone
wpop <- getPop2(popMethod, wsize, wzones)
colnames(wpop) = c("size", "zone")

# consistency check
stopifnot(nrow(filter(wpop, zone == 0)) == wzones[1])
stopifnot(nrow(filter(wpop, size == 0)) == wsize[1])

# TODO
# how can household populations be consistent with overall population?

# TODO
# generate OD dataset containing depTime, arrTime

