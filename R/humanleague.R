#' humanleague
#'
#' R package for synthesising populations from aggregate and (optionally) seed data
#'
#' See README.md for detailed information and examples.
#'
#' @section Overview:
#' The package contains algorithms that use a number of different microsynthesis techniques:
#' \itemize{
#'  \item{Iterative Proportional Fitting (IPF), \emph{a la} \pkg{mipfp} package}
#'  \item{\href{http://jasss.soc.surrey.ac.uk/20/4/14.html}{Quasirandom Integer Sampling (QIS)} (no seed population)}
#'- \item{Quasirandom Integer Sampling of IPF (QISI): A combination of the two techniques whereby IPF solutions are used to sample an integer population.}
#'}
#'
#' The latter provides a bridge between deterministic reweighting and combinatorial optimisation, offering advantages of both techniques:
#' \itemize{
#'   \item{generates high-entropy integral populations}
#'   \item{can be used to generate multiple populations for sensitivity analysis}
#'   \item{is less sensitive than IPF to convergence issues when there are a high number of empty cells present in the seed}
#'   \item{relatively fast computation time, though running time is linear in population}
#' }
#'
#' The algorithms:
#' \itemize{
#'   \item{support arbitrary dimensionality* for both the marginals and the seed.}
#'   \item{produce statistical data to ascertain the likelihood/degeneracy of the population (where appropriate).}
#' }
#' [* excluding the legacy functions retained for backward compatibility with version 1.0.1]
#'
#' The package also contains the following utility functions:
#' \itemize{
#'   \item{a Sobol sequence generator}
#' - \item{functionality to convert fractional to nearest-integer marginals (in 1D). This can also be achieved in multiple dimensions by using the QISI algorithm.}
#'   \item{functionality to 'flatten' a population into a table: this converts a multidimensional array containing the population count for each state into a table listing individuals and their characteristics.}
#'}

#'
#' @section Functions:
#' \code{\link{flatten}}
#'
#' \code{\link{ipf}}
#'
#' \code{\link{prob2IntFreq}}
#'
#' \code{\link{qis}}
#'
#' \code{\link{qisi}}
#'
#' \code{\link{sobolSequence}}
#'
#' \code{\link{integerise}}
#'
#' \code{\link{unitTest}}
#'
#' @docType package
#' @name humanleague
"_PACKAGE"

#' @useDynLib humanleague
#' @importFrom Rcpp sourceCpp


