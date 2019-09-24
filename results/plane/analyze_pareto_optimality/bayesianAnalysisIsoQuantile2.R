data = read.csv("pareto-data.tsv")


library(dplyr)
library(tidyr)
library(TruncatedNormal)

language = "Basque"

for(language in unique(data$Language)) {
    
    data2 = data %>% filter(Language == language, !is.na(Surp), !is.na(Pars))
    
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
    dp = Fit(dp, 1105)
    
    bestLambdas = c()
    bestQuantiles = c()
    
    curves = list()


       maxPars = max(parsGround, max(pars))+1.0
       minPars = min(parsGround, min(pars))-2.0
       minSurp = min(surpGround, min(surp))-2.0
       maxSurp = surpGround+1.0

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


    for(i in (1:100)) {
       cat(language, i, "\n")
       j = 1000 + 50 * i
       weight = dp$weightsChain[[j]]
       mu = dp$clusterParametersChain[[j]]$mu
       sig = dp$clusterParametersChain[[j]]$sig
       numberOfClusters = length(weight)
       parsGrid = array(array(parsGrid_, dim=c(GRID_SIZEp, GRID_SIZEs)), dim=c(GRID_SIZEp*GRID_SIZEs))
       surpGrid = array(aperm(array(surpGrid_, dim=c(GRID_SIZEs, GRID_SIZEp)), c(2,1)), dim=c(GRID_SIZEp*GRID_SIZEs))
       grid = cbind(parsGrid, surpGrid) # GRID_SIZE x 2
       transform = array(apply(sig, 3, function(x) { return(chol(solve(x))) }), dim=c(2, 2, numberOfClusters))
       gridZ = array(apply(array(grid, dim=c(GRID_SIZEp*GRID_SIZEs, 1, 2, numberOfClusters)), 1, function(x) { return((x)) }), dim=c(2, numberOfClusters, GRID_SIZEp*GRID_SIZEs, 1))
       gridZ = aperm(gridZ, c(2,3,4,1))
       ground_Z = c(parsGround, surpGround)
       total_gridZ_p = 0
       total_groundZ_p = 0
       for(q in (1:numberOfClusters)) {
          groundZ_p = pmvnorm(mu[,,q], sig[,,q], ub=ground_Z)
          total_groundZ_p = total_groundZ_p + weight[q] * groundZ_p
          gridZ_p = apply(gridZ[q,,,], 1, function(x) { return(pmvnorm(mu[,,q], sig[,,q], ub=x)) })
          total_gridZ_p = total_gridZ_p + weight[q] *     gridZ_p
       }
       total_gridZ_p = array(total_gridZ_p, dim=c(GRID_SIZEp, GRID_SIZEs))
       boundarySurprisalPerPars = apply(total_gridZ_p, 1, function(x) { sum(x < total_groundZ_p) })
       boundary = data.frame(x=parsGrid_, y=(surpScale*boundarySurprisalPerPars+surpOffset))
       curves[[i]] = boundary
    }

    library(dplyr)
    library(tidyr)

    totalSurpGridBySample = data.frame(x=c(), y=c(), sample=c())
    totalSurpGrid = 0*(1:GRID_SIZEp)
    for(i in (1:length(curves))) {
      totalSurpGrid = totalSurpGrid + curves[[i]]$y
      totalSurpGridBySample = rbind(totalSurpGridBySample, curves[[i]] %>% mutate(sample=i))
    }
    library(ggplot2)
    isoCurve = data.frame(x=parsGrid_, y=totalSurpGrid/length(curves))
    boundary = nrow(isoCurve %>% filter(y > surpOffset)) + 1
    isoCurve = isoCurve[(1:boundary),]
    plot = ggplot(isoCurve, aes(x=-x, y=-y)) + geom_line() + geom_point(data=data.frame(pars=pars, surp=surp), aes(x=-pars, y=-surp)) + geom_point(data=data.frame(pars=c(parsGround), surp=c(surpGround)), aes(x=-pars, y=-surp), color="red")
    ggsave(plot, file=paste("figures/isoCurve_",language,".pdf", sep=""))

    write.csv(totalSurpGridBySample, file=paste("../../../grammars/pareto-curves/pareto-smooth/iso-pareto-full-", language, sep=""))

}



    


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


