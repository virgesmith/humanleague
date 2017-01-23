
# Welcome to Not-so-simple-world, pop. 1000
#
# Simple world assumptions:
# - one person per household

# Population generation method: one of ipfp, ml, chi2, lsq, qipf
popMethod = "qipf"

library(dplyr)
library(humanleague)


getPop <- function(popMethod, l) {
  # list of dimensions of each marginal constraint
  target.list <- as.list(c(1:length(l)))
  target.data <- l

  # initial guess is average pop per state
  seed <- array(1, dim=sapply(l, length))

  if (popMethod == "qipf") {
    pop <- humanleague::synthPop(l, 10);
  }
  else if (popMethod == "ipfp") {
    pop <- mipfp::Ipfp(seed, target.list, target.data)
  } else {
    pop <- mipfp::ObtainModelEstimates(seed, target.list, l, method = popMethod)
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

pop <- getPop(popMethod, list(age, gender, pzones))
pop2 <- getPop("ipfp", list(age, gender, pzones))
#colnames(populace$x.hat) <- c("agegroup", "gender", "zone")

# consistency check
#stopifnot(nrow(filter(populace$x.hat, zone == 0)) == pzones[1])
#stopifnot(nrow(filter(populace$x.hat, agegroup == 0)) == age[1])
#stopifnot(nrow(filter(populace$x.hat, gender == 0)) == gender[1])

nPeople = sum(pzones)
nStates = length(pzones) * length(gender) * length(age)

# Households per zone (250)
hzones <- c(10, 22, 38, 45, 2, 47, 25, 1, 60)
# Household sizes: 1, 2, 3, 4, 5-6, 7+
hsize <- c(25, 45, 55, 60, 45, 20 )

# generate O (houses) microsim per zone
hpop <- getPop(popMethod, list(hsize, hzones))
hpop2 <- getPop("ipfp", list(hsize, hzones))
#colnames(hpop) = c("size", "zone")

# consistency check
# stopifnot(nrow(filter(hpop, zone == 0)) == hzones[1])
# stopifnot(nrow(filter(hpop, size == 0)) == hsize[1])


# Workplaces per zone (50)
wzones <- c(1, 2, 3, 4, 20, 2, 15, 3)
# Workplace sizes: 1-10, 11-50, 50-200, 201+
wsize <- c(12, 13, 13, 12)

# generate D (work, shops, social) microsim per zone
wpop <- getPop(popMethod, list(wsize, wzones))
#colnames(wpop) = c("size", "zone")

# consistency check
# stopifnot(nrow(filter(wpop, zone == 0)) == wzones[1])
# stopifnot(nrow(filter(wpop, size == 0)) == wsize[1])

# TODO
# how can household populations be consistent with overall population?

# TODO
# generate OD dataset containing depTime, arrTime

#world <- raster::raster(nrows=3, ncols=3, xmn=-150, xmx=150, ymn=-150, ymx =150)
#values(world) <- pzones
#polys <- raster::rasterToPolygons(world)
#stopifnot(polys@data == pzones)

