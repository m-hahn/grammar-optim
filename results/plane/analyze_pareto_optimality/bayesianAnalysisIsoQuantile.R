data = read.csv("pareto-data.tsv")


library(dplyr)
library(tidyr)

language = "Estonian"

#for(language in unique(data$Language)) {
    
    data2 = data %>% filter(Language == language)
    
    surp = data2$Surp
    pars = data2$Pars
    
    surpGround = mean(data2$SurpGround)
    parsGround = mean(data2$ParsGround)
    
    surpGround = (surpGround - mean(surp, na.rm=TRUE)) / sd(surp, na.rm=TRUE)
    parsGround = (parsGround - mean(pars, na.rm=TRUE)) / sd(pars, na.rm=TRUE)
    
    surp = (surp - mean(surp, na.rm=TRUE)) / sd(surp, na.rm=TRUE)
    pars = (pars - mean(pars, na.rm=TRUE)) / sd(pars, na.rm=TRUE)
    
    X = cbind(surp, pars)
    X = X[!is.na(X[,1]),]
    X = X[!is.na(X[,2]),]
    
    library(dirichletprocess)
    dp = DirichletProcessMvnormal(X)
    dp = Fit(dp, 2000)
    
    bestLambdas = c()
    bestQuantiles = c()
    
curves = list()


       maxPars = max(parsGround+0.3, max(pars))
       minPars = min(parsGround-0.3, min(pars))
       minSurp = min(surpGround-0.3, min(surp))
       maxSurp = surpGround+0.4

       # 0 -> minPars
       # 21 -> maxPars
       # ax+b
       # a 0 + b = minPars
       # a 21 + minPars = maxPars
       parsOffset = minPars
       parsScale = (maxPars-minPars)/21
       parsGrid_ = (parsScale * (0:(20-1))) + parsOffset #c(parsGround-0.1, parsGround, parsGround+0.1) #((0:(101-1))-90)/20 #c(parsGround)
       surpOffset = minSurp
       surpScale = (maxSurp-minSurp)/21
       surpGrid_ = (surpScale * (0:(20-1))) + surpOffset
       GRID_SIZEp = length(parsGrid_)
       GRID_SIZEs = length(surpGrid_)


    i = 1000
#    for(i in (1000:1020)) {
       cat(i, "\n")
       weight = dp$weightsChain[[i]]
       mu = dp$clusterParametersChain[[i]]$mu
       sig = dp$clusterParametersChain[[i]]$sig
    
       numberOfClusters = length(weight)
       # calculate the density assigned to better efficiency values, for each lambda
       # create a grid of values at which to evaluate


       parsGrid = array(array(parsGrid_, dim=c(GRID_SIZEp, GRID_SIZEs)), dim=c(GRID_SIZEp*GRID_SIZEs))
       surpGrid = array(aperm(array(surpGrid_, dim=c(GRID_SIZEs, GRID_SIZEp)), c(2,1)), dim=c(GRID_SIZEp*GRID_SIZEs))
       
       grid = cbind(parsGrid, surpGrid) # GRID_SIZE x 2
#       gridZ = array(grid, dim=c(GRID_SIZE*GRID_SIZE, 1, 2, numberOfClusters)) - array(mu, dim=c(1, 1, 2, numberOfClusters))
       transform = array(apply(sig, 3, function(x) { return(chol(solve(x))) }), dim=c(2, 2, numberOfClusters))

#       mu2 = array(mu, c(2, numberOfClusters))



       gridZ = array(apply(array(grid, dim=c(GRID_SIZEp*GRID_SIZEs, 1, 2, numberOfClusters)), 1, function(x) { return((x)) }), dim=c(2, numberOfClusters, GRID_SIZEp*GRID_SIZEs, 1))
       gridZ = aperm(gridZ, c(2,3,4,1))
       #expected end: cluster x GRID x 1 x 2

       ground_Z = c(parsGround, surpGround)

       total_gridZ_p = 0
       total_groundZ_p = 0
       for(q in (1:numberOfClusters)) {
#          groundZ_ = transform[,,q] %*% (ground_Z-mu[,,q])
          groundZ_p = pmvnorm(mu[,,q], sig[,,q], ub=ground_Z)
#          groundZ_p = groundZ_p[1] * groundZ_p[2]
          total_groundZ_p = total_groundZ_p + weight[q] * groundZ_p

#          gridZ_ = apply(gridZ[q,,,], 1, function(x) { return(transform[,,q] %*% x) })

          gridZ_p = apply(gridZ[q,,,], 1, function(x) { return(pmvnorm(mu[,,q], sig[,,q], ub=x)) })
#          gridZ_p = pnorm(gridZ_)
#          gridZ_p = gridZ_p[1,] * gridZ_p[2,]
          total_gridZ_p = total_gridZ_p + weight[q] *     gridZ_p
       }
       # TODO it is still very weird that the resulting total_gridZ_p is not monotonic. Problem with the cumulative density!!!!!!!!!!!!!

       total_gridZ_p = array(total_gridZ_p, dim=c(GRID_SIZEp, GRID_SIZEs))
       
       # for each parseability value, find the *largest* grid_p that is below the ground_p
       
       boundarySurprisalPerPars = apply(total_gridZ_p, 1, function(x) { sum(x < total_groundZ_p) })
       
       boundary = data.frame(x=parsGrid_, y=(surpScale*boundarySurprisalPerPars+surpOffset))
#       boundary = boundary[boundary$y != 0,]

#       if(sum(boundary$y == 0) > 0) {
#          boundary[boundary$y == 0,]$y = NA
#       }

       curves[[i-999]] = boundary

#       library(ggplot2)
#       plot = ggplot(boundary, aes(x=x, y=y)) + geom_line() + geom_point(data=data.frame(pars=pars, surp=surp), aes(x=pars, y=surp)) + geom_point(data=data.frame(pars=c(parsGround), surp=c(surpGround)), aes(x=pars, y=surp), color="red")
#       ggsave(plot, file="tmp.pdf")
#       
#}


    
totalSurpGrid = 0*(1:GRID_SIZEp)
for(i in (1:length(curves))) {
  totalSurpGrid = totalSurpGrid + curves[[i]]$y
}

       library(ggplot2)
       plot = ggplot(data.frame(x=parsGrid_, y=totalSurpGrid/length(curves)), aes(x=x, y=y)) + geom_line() + geom_point(data=data.frame(pars=pars, surp=surp), aes(x=pars, y=surp)) + geom_point(data=data.frame(pars=c(parsGround), surp=c(surpGround)), aes(x=pars, y=surp), color="red")
       ggsave(plot, file="tmp.pdf")
#       






    
#    write.csv(data.frame(bestLambdas=bestLambdas, bestQuantiles=bestQuantiles), file=paste("~/CS_SCR/posteriors/pareto-smooth/pareto-", language, sep=""))


#}



#plot(bestLambdas, 1-bestQuantiles)
#
#mean(bestLambdas)
#sd(bestLambdas)
#
#mean(1-bestQuantiles)
#
#ecdf(1-bestQuantiles)(0.5)
#
#
#hist(bestLambdas)
#
##analysis/efficiency-control-brms.R:cat(paste("dependency", "satisfiedFraction", "posteriorMean", "posteriorSD", "posteriorOpposite", sep="\t"), file="../output/results-prevalence-two-09-large.tsv", append=TRUE, sep="\n")
#
#


