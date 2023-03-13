library(VGAM)
library(extraDistr)
options(warn = - 1)                # Disable warning messages globally

get_alpha_beta = function(junc_counts, cell_cluster_counts){
    
    x=junc_counts
    y=cell_cluster_counts-x

    #x = number of successes
    #y = N-x  
    #N = total counts in the cluster for a given cell
    
    fit=vglm(cbind(x, y) ~ 1, betabinomialff, trace = FALSE)
    alph = Coef(fit)[1]
    beta = Coef(fit)[2]
    res=c(alph, beta)
    return(res)
}