nS <- 200
nW <- 400

T <- 24
Sbar <- 1.0
rho <- 0.99
sigma <- 0.01
lamb <- 0.01

J <- array(0, dim = c(T, nS + 1, nW + 1))
opt <- array(0, dim = dim(J))
p <- array(NA, dim = c(nS + 1, nS + 1))


compute_p <- function() {  # transition probability of S
  for (i in 0:nS) {
    tot <- 0.0
    for (j in 0:nS) {
      Sprev <- i / nS * 2
      Snext <- j / nS * 2
      Snext_AR <- (Sprev - Sbar) * rho + Sbar
      epsilon <- Snext - Snext_AR
      pdf <- 1 / (sigma * sqrt(2 * pi)) * exp(-0.5 * (epsilon / sigma)^2)
      p[i + 1, j + 1] <- pdf
      tot <- tot + pdf
    }

    for (j in 1:(nS + 1)) {  # normalize
      p[i + 1, j] <<- p[i + 1, j] / tot
    }
  }
}


compute_Jt <- function(t) {
  for (i in 0:nS) {
    for (j in 0:nW) {
      J[t, i + 1, j + 1] <<- -1e30
      S <- i / nS * 2
      for (k in 0:j) {
        x <- k / nW * 3
        E <- 0.0  # expectation
        for (l in 1:(nS + 1)) {
          val <- x * (rho * (S - Sbar) + Sbar - lamb * x) + J[t + 1, l, j - k + 1]
          E <- E + p[i + 1, l] * val
        }
        if (E > J[t, i + 1, j + 1]) {
          opt[t, i + 1, j + 1] <<- x
          J[t, i + 1, j + 1] <<- E
        }
      }
    }
  }
}

compute_JT <- function() {
  for (i in 1:(nS + 1)) {
    for (j in 1:(nW + 1)) {
      S <- (i - 1) / nS * 2
      x <- (j - 1) / nW * 3
      val <- x * (rho * (S - Sbar) + Sbar - lamb * x)
      J[T, i, j] <<- val
    }
  }
}

main <- function() {
  opt[] <<- -1
  compute_p()
  compute_JT()

  for (t in (T - 1):1) {
    compute_Jt(t)
  }
}

main()