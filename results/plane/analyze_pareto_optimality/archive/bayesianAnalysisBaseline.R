data = read.csv("pareto-data.tsv")

library(dplyr)
library(tidyr)

language = "Telugu"

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

i = 1000
for(i in (1000:2000)) {
   weight = dp$weightsChain[[i]]
   mu = dp$clusterParametersChain[[i]]$mu
   sig = dp$clusterParametersChain[[i]]$sig
  
   # calculate the density assigned to better efficiency values, for each lambda
  
   # shape of mu: [1, 2, numComponents]
   # shape of sig: [2, 2, numComponents]


#directly find the best lambda

   bestQuantile = 2.0  
   bestLambda = NA
   for(lambda_ in (0:19)) {
     lambda = lambda_/20
     phi = c(1, lambda)
    
     numberOfClusters = length(weight)
    
     muEff = colSums(array(phi, dim=c(1, 2, numberOfClusters))  * mu, dims=2)
     sigEff = colSums(sig * aperm(array(phi, dim=c(2, 2, numberOfClusters)), perm=c(2,1,3)) * array(phi, dim=c(2, 2, numberOfClusters)), dim=2)
     
     effGround = sum(phi * c(parsGround, surpGround))
     
     quantiles = pnorm(effGround, mean=muEff, sd=sigEff)
     
     totalQuantile = sum(quantiles * weight)
#     cat(lambda, totalQuantile, "\n")
     if(totalQuantile <= bestQuantile) {
	     bestQuantile = totalQuantile
             bestLambda = lambda
     }
     if(totalQuantile > 1) {
	     break
   }
   }
   cat(bestLambda, bestQuantile, "\n")
   bestLambdas = c(bestLambdas, bestLambda)
   bestQuantiles = c(bestQuantiles, bestQuantile)
}	


plot(bestLambdas, 1-bestQuantiles)

mean(bestLambdas)
sd(bestLambdas)

mean(1-bestQuantiles)

ecdf(1-bestQuantiles)(0.5)


hist(bestLambdas)

#analysis/efficiency-control-brms.R:cat(paste("dependency", "satisfiedFraction", "posteriorMean", "posteriorSD", "posteriorOpposite", sep="\t"), file="../output/results-prevalence-two-09-large.tsv", append=TRUE, sep="\n")




