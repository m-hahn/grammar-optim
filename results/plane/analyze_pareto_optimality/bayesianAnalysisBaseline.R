data = read.csv("pareto-data.tsv")


library(dplyr)
library(tidyr)

data2 = data %>% filter(Language == "English")

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

i = 1000
#for(i in (1000:2000)) {
 weight = dp$weightsChain[i]
 mu = dp$clusterParametersChain[[i]]$mu
 sig = dp$clusterParametersChain[[i]]$sig

 # calculate the density assigned to better efficiency values, for each lambda


#}	




