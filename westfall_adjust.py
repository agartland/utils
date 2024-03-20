
"""Version in kyotil used in correlates analysis:
https://github.com/youyifong/kyotil/blob/master/R/P.adj.perm.R
"""



###############################################################################
###  Compute p-values on permuted data to adjust for family-wise error rate (FWER) or false discovery rate(FDR)
###  Created by Sue Li,  10-17-2014; modified by Paul T Edlefsen 11-14-2014
################################################################################


### Calculate FWER- and FDR- adjusted p-values (p.FWER,p.FDR,p.unadj, num.null.permutations.by.variable ), returned in sorted order (ascending) by p.
### The return value is a three-column matrix with rownames matching the names of the given value of p (but in a different order).
### p: the vector of p-values estimated from the original data
### p.b:  the matrix of p-values from permuted data set; rows are different permuations.
### num.null.permutations.by.variable: count of non-NA values in the permuted data.
### Precondition: ncol( p.b ) == length( p ).
##################################################
p.adj.usingPValuesOfPermutedData <- function ( p.unadj, p.perms )
{
    stopifnot( ncol( p.perms ) == length( p.unadj ) );

    # If "p" has no names, give it names either from colnames( p.perms ) or as 1:length( p.unadj ).
    if( is.null( names( p.unadj ) ) ) {
        if( is.null( colnames( p.perms ) ) ) {
            names( p.unadj ) <- 1:length( p.unadj );
        } else {
            names( p.unadj ) <- colnames( p.perms );
        }
    }
    
    # We must sort the p.unadj values first.
    mode( p.unadj ) <- "numeric";
    which.are.NA <- which( is.na( p.unadj ) );
    p.unadj.order <- order( p.unadj ); # Note this puts NAs last/"largest".
    # We must maintain the same ordering between p.unadj and the columns of p.perms.
    p.unadj <- p.unadj[ p.unadj.order ];
    p.perms <- p.perms[ , p.unadj.order, drop = FALSE ];
    
    if( any( is.na( p.perms ) ) ) {
        num.null.permutations.by.variable <-
            apply( p.perms, 2, function ( .col ) { sum( !is.na( .col ) ) } );
    } else {
        num.null.permutations.by.variable <- rep( nrow( p.perms ), ncol( p.perms ) );
    }
    
    # FWER (family-wide error rate) adjusted p-values.
    p.FWER <- sapply( 1:length( p.unadj ), function ( j ) { sum( apply( p.perms, 1, base:::min ) <= p.unadj[ j ] ) / num.null.permutations.by.variable[ j ] } );
  
    # calculate p-values adjusted for FDR
    p.FDR <- rep( 0, length( p.unadj ) );

    # First, calculate empirical estimates of E( R0/R | R>0 ), where R0 is number of rejections at a level alpha under the null and R is rejections at level alpha in the observed data.
    p.FDR <- sapply( 1:length( p.unadj ), function ( j ) {
      ## Expected number of rejections at level p.unadj[ j ] under null hypothesis (false rejections R0)
      ER0 <- sum( apply( p.perms <= p.unadj[ j ], 1, sum, na.rm = T ) ) / num.null.permutations.by.variable[ j ];
      ## Number of rejections at level p.unadj[ j ] observed in the actual p-values; since they're sorted, this is just j.
      R <- j;
      ## E( R0/R | R>0 )  (Note that here by definition p.unadj[ j ] is one of the observed values, so R > 0!)
      return( base:::min( ER0 / R, 1 ) );
    } );
  
    ## Actually, FDR( p.unadj[ j ] ) is min_{j:( p.unadj[ j ] >= p.unadj[ i ] )} { FDR..(p.unadj[ j ]) }, so walk down from top.unadj and replace them.
    for( j in ( length( p.unadj ) - 1 ):1 )
    {
      p.FDR[ j ] <- base:::min( p.FDR[ j ], p.FDR[ j+1 ] );
    }
    .rv <- cbind( p.FWER, p.FDR, p.unadj, num.null.permutations.by.variable );
    rownames( .rv ) <- names( p.unadj );
    return( .rv );
} # p.adj.usingPValuesOfPermutedData (..)
  



##################################################
### Calculate the adjusted p-values to control familywise error rate(FWER)and false discovery rate(FDR) 
### using the resampleing based methods. 
###
### Input:
### 	p.unadj: an 1xm vector of unjected p-values calculated from the original data set
### 	p.perms:  an Bxm matrix of p-values calculated from B sets of data sets that are resampled from the orginal data
###         set under null hypothese
###   alpha:  any unjected p-values less than alpha will not be calculated for adjusted p-values and their adjusted
###           p-values are NA. 
### Output:
###	p.FWER: an 1xm vector of adjusted p-values to control FWER
###	p.FDR:  an 1xm vector of adjusted p-values to control FDR
###
### FWER adjusted p-values, P.FWER, are calculated based on the resampling step down procedure (Westfall and Young 1993). 
### FDR adjusted p-values, p.FDR, are calculated based on the estimations of E(R0)/E(R) where E(R0) is the
###     expectation of the number of rejected null hypotheses (R0) that is estimated from the resampled data sets
###     under the null hypotheses; E(R) is the expection of the number of all rejected hypotheses (R) that is estimated
###     by the maximum of R0 and the number of rejections from the observed data set. 
### According to Jensen inequality and R0 is a positive linear function of R, 
###   FDR=E(R0/R) <= E(R0)/E(R). Therefore, our estimation for p.FDR would control below the FDR level. 
### Ref: 
###     Westfall and Young 1993 "Resampling-based multiple testing: Examples and methods for p-value
###         adjustment", John Wiley & Sons. 
###     Westfall and Troendle "Multiple testing wirh minimum aasumptions", Biom J. 2008
###     Storey and Tibshrani "Statistical siggnificance for genomewide studies", PNAS 2003
###
### Created by Sue Li, 4/2015
##################################################
p.adj <- function(p.unadj,p.perms,alpha=0.05)
{
  stopifnot( ncol( p.perms ) == length( p.unadj ) );
  
  # If "p.unadj" has no names, give it names either from colnames( p.perms ) or as 1:length( p.unadj ).
  if( is.null( names( p.unadj ) ) ) {
    if( is.null( colnames( p.perms ) ) ) {
      names( p.unadj ) <- 1:length( p.unadj );
    } else {
      names( p.unadj ) <- colnames( p.perms );
    }
  }
  B = dim(p.perms)[1]
  m = length(p.unadj)

  ### order p from the smallest to the largest
  # We must sort the p.unadj values first.
  mode( p.unadj ) <- "numeric";
  which.are.NA <- which( is.na( p.unadj ) );
  p.unadj.order <- order( p.unadj ); # Note this puts NAs last/"largest".
  # We must maintain the same ordering between p.unadj and the columns of p.perms.
  p.unadj <- p.unadj[ p.unadj.order ];
  p.perms <- p.perms[ , p.unadj.order, drop = FALSE ];
  
  len = sum(round(p.unadj,2)<=alpha)
  # calculate FWER-adjusted p-values 
  p.FWER=rep(NA,length(p.unadj))
  
  for (j in 1:len)
  {
      p.FWER[j] = sum((apply(p.perms[,j:m], 1, min, na.rm = T)<=p.unadj[j]))/B
  }
  ## enforce monotonicity using successive maximization
  p.FWER[1:len] = cummax(p.FWER[1:len])
  
  # calculate FDR-adjusted p-values
  p.FDR=rep(NA,length(p.unadj))
  for (j in 1:len)
  {  
      ## given each p-value
      ## estimate the expectation of # of rejections under null hypotheses  
      R0_by_resample = apply(p.perms<=p.unadj[j], 1, sum, na.rm = T )
      ER0 = sum(R0_by_resample)/B
      ## calculate # of rejections observed in the data
      R.ob = j  #sum(p.unadj<=p.unadj[j])
      ## R is max(R0,R)
      R = sum(pmax(R0_by_resample,R.ob))/B
      ## FDR=E(R0/R|R>0) FDR=0 if R=0
      p.FDR[j]=min(ifelse(R>0,ER0/R,0),1)
  }
  
  o1=order(p.unadj[1:len],decreasing=TRUE)
  ro=order(o1)
  
  p.FDR[1:len]=pmin(1,cummin(p.FDR[1:len][o1]))[ro]
  
  ## the results are in an ascending order of the unadjusted p-values
  .rv <- cbind(p.unadj,p.FWER,p.FDR)
  rownames(.rv) <- names(p.unadj)
  return(.rv)
}
  
"""Test code from Sue Li"""
library("doBy")

p_value <- function(snp,data)
{
    data$group = as.factor(data[[snp]])
    # use case-only (Jame Dai's paper) to test interaction between gene and treatment and calculate the VE by gene 
    out <- glm(treat ~ group,family=binomial,data=data) 
    # test the significance of interaction of treatment and gene group
    nlev <- length(unique(data$group[!is.na(data$group)]))
    p_d <- round(summary(out)$coef[,4],2)
    contr <- diag(nlev)[2:nlev,]
    p <- esticon(out,contr,level=0.95,joint.test=TRUE)
    p <- as.numeric(p[3])
    return(p)
}

load("snp.169match.data")
snp.dat=snp.169match.data
snp <- names(snp.dat)[3:30]

p.da=NULL
for (s in snp)
{
    p.da=c(p.da,p_value(s,snp.dat))
}
p.BH=p.adjust(p,method="BH")
# p.fdr=p.adjust(p,method="fdr")

B=10000
len=length(snp)
p.b=matrix(0,B,len)

dat=snp.dat
for (i in 1:B)
{
    dat$treat = sample(snp.dat$treat)
    for (j in 1:len)
    {
        p.b[i,j]=p_value(snp[j],dat)
    }
}

save(p.b,file="FcRsnps_p_169match.b")





"""From Bing 3-23-2023"""
import numpy as np
from scipy.stats import norm

def westfall_young(p_values, num_permutations=10000):
    """
    Westfall and Young (1993) permutation-based multiplicity adjustment algorithm.
    
    Parameters
    ----------
    p_values : array_like
        Array of p-values to be adjusted.
    num_permutations : int, optional
        Number of permutations to perform. Default is 10000.
        
    Returns
    -------
    adjusted_p_values : ndarray
        Array of adjusted p-values.
    """
    p_values = np.asarray(p_values)
    num_tests = len(p_values)
    
    # Initialize array to store adjusted p-values
    adjusted_p_values = np.zeros(num_tests)
    
    # Compute the maximum statistic for the observed data
    max_statistic = -norm.ppf(p_values).max()
    
    # Initialize counter for number of times the maximum statistic is exceeded
    num_exceeds = np.zeros(num_tests)
    
    # Perform permutations
    for i in range(num_permutations):
        # Generate random normal deviates for each test
        random_deviates = norm.rvs(size=num_tests)
        
        # Compute the maximum statistic for this permutation
        max_statistic_permutation = np.abs(random_deviates).max()
        
        # Update the counter for number of times the maximum statistic is exceeded
        num_exceeds += (max_statistic_permutation >= max_statistic)
        
    # Compute adjusted p-values
    adjusted_p_values = (num_exceeds + 1) / (num_permutations + 1)
    
    return adjusted_p_values