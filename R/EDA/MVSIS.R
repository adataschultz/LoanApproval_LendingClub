################################################################################
############################ MVSIS Source Code #################################
################################################################################
# Functions to compute the criteria in the simulation.
# 1. M: to compute the minimum model size to ensure the inclusion of all active predictors. 
# 2. mqtl: to compute the 5%, 25%, 50%, 75% and 95% quantiles of the minimum model size out of 1,000 replications.
# 3. Sel.rate: to compute the proportion that every single active predictor is selected 
#    for a given model size, which is defauted c[n/log(n)], in the 1,000 replications.

M <- function(true.v,rank.mtx){
    # Input
    # true.v   :  the true variables index
	# rank.mtx :  the ranked index matrix by screening method for the 1000 replications
	#             each column corresponds the ranked index in one replication.
	# Output
	# M        :  a vector of the minimum model sizes to ensure the inclusion of all active predictors 
    r <- min(dim(rank.mtx)[2], length(rank.mtx))
	  M <- c()
	  for (j in 1:r){
        M[j] <- max(match(true.v, rank.mtx[,j]))
    }
    return(M)
}

mqtl <- function(M){
    # Input    
	# M        :  a vector of the minimum model sizes to ensure the inclusion of all active predictors 
	# Output
	# 5%,25%,50%,75%,95% quantiles of minimum model sizes out of 1000 replications
    quantile(M, probs=c(0.05,0.25,0.5,0.75,0.95))
}

Sel.rate <- function(n,c,true.v,rank.mtx){
    # Input
	# n        :  the sample size
	# c        :  coeficient of cutoffs, for example c=2, cutoff=2[n/log(n)]
	# true.v   :  the true variables index
	# rank.mtx :  the ranked index matrix by screening method for the 1000 replications
	#             each column corresponds the ranked index in one replication.
	# Output
	# rate     :  the proportions that every single active predictor is selected 
    #             for a given model size, which is defauted c[n/log(n)], in the 1,000 replications.
    d <- c * floor(n / log(n))
    rank.mtx.sel <- rank.mtx[1:d,]
    r <- min(dim(rank.mtx)[2], length(rank.mtx))
    p0 <- length(true.v)
	  R <- matrix(0,p0,r)
	  rate <- c()
	  for (i in 1:p0){
	      for (j in 1:r){
            R[i,j] <- (min(abs(rank.mtx.sel[,j] - true.v[i]))==0)
        }
		        rate[i] <- mean(R[i,])
	  }
	  return(rate)
}


creat.sigma1 <- function(rho,p){
		Sigma1<-matrix(0,p,p)
	  for (i in 1:p){
		 		for (j in max(1, (i-50)):min(p, (i+50))){
			 			Sigma1[i,j] < -rho^(abs(i - j))
		 }
	}
    return(Sigma1)
}
#sigma = creat.sigma1(rho,p)

################################################################################
################### Compute MV(X,Y), where Y is discrete ####################### 
################################################################################
Fk <- function(X0,x){
    Fk = c()
    for (i in 1:length(x)){
        Fk[i] = sum(X0 <= x[i]) / length(X0)
    }
    return(Fk)
}

Fkr <- function(X0,Y,yr,x){
    Fkr = c()
	  ind_yr = (Y==yr)
    for (i in 1:length(x)){
        Fkr[i] = sum((X0 <= x[i]) * ind_yr) / sum(ind_yr)
    }
    return(Fkr)
}

MV <- function(Xk,Y){
    Fk0 <- Fk(Xk,Xk)
    Yr <- unique(Y)
	  MVr <- c()
    for (r in 1:length(Yr)){
	      MVr[r] <- (sum(Y==Yr[r]) / length(Y)) * mean((Fkr(Xk,Y,Yr[r],Xk) - Fk0)^2)
    }
	  MV <- sum(MVr)
	  return(MV)
}

################################################################################
###################  THE KOLMOGOROV FILTER-Mai and Zou(2013) ###################
###################  Compute KF(X,Y), where Y is Binary      ###################
################################################################################
KF<-function(X,Y){
		Yc = unique(Y)
	  F1 = c(); F2 = c()
    for (i in 1:length(X)){
	  		F1[i] = sum((X <= X[i]) * (Y == Yc[1])) / sum(Y == Yc[1])
		 		F2[i] = sum((X <= X[i]) * (Y == Yc[2])) / sum(Y == Yc[2])
		}
	  KF = max(abs(F1 - F2))
	  return(KF)
}

################################################################################
################################################################################