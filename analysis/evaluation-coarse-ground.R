
data = read.csv("CS_SCR/deps/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)



library(dplyr)
library(tidyr)
library(ggplot2)

dryer_greenberg_fine = data %>% mutate(DH_Weight = DH_Mean_NoPunct)



languages = read.csv("languages.tsv", sep="\t")
dryer_greenberg_fine  = merge(dryer_greenberg_fine, languages, by=c("Language"), all.x=TRUE)

#dryer_greenberg_fine %>% group_by(Dependency) %>% summarise(DH_Weight = quantile(DH_Weight - 2*DH_Sigma_NoPunct,0.999)) %>% print(n=50
#dryer_greenberg_fine %>% group_by(Dependency) %>% summarise(DH_Weight = quantile(DH_Weight + 2*DH_Sigma_NoPunct,0.001)) %>% print(n=50


# Interest in general biases
#dryer_greenberg_fine %>% group_by(Dependency) %>% summarise(DH_Weight = quantile(DH_Weight,0.1)) %>% print(n=50)
#dryer_greenberg_fine %>% group_by(Dependency) %>% summarise(DH_Weight = quantile(DH_Weight,0.9)) %>% print(n=50)
#+
#advmod
#clf
#det
#expl
#nsubj
#nummod
#
#- things that come after the head in basically all languages
#appos
#conj
#fixed
#flat
#list
# hist((dryer_greenberg_fine %>% filter(Dependency == "appos"))$DH_Weight) manual inspecton

# dryer_greenberg_fine %>% group_by(Dependency) %>% summarise(Distance_Mean_NoPunct = quantile(Distance_Mean_NoPunct,0.9)) %>% print(n=50
# dryer_greenberg_fine %>% group_by(Dependency) %>% summarise(Distance_Mean_NoPunct = quantile(Distance_Mean_NoPunct,0.1)) %>% print(n=50)

library("rstan")
library("brms")

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

dependency = "nmod"

getCorrPair = function(dependency) {
   corr_pair = dryer_greenberg_fine %>% filter((Dependency == dependency) | (Dependency == "obj"))
   corr_pair = unique(corr_pair %>% select(Family, Language, FileName, Dependency, DH_Weight, ModelName)) %>% spread(Dependency, DH_Weight)
   corr_pair$correlator = corr_pair[[dependency]]
   corr_pair = corr_pair %>% mutate(correlator_s = pmax(0,sign(correlator)), obj_s=pmax(0,sign(obj)))
   return(corr_pair)
}

corr_pair = getCorrPair("nmod")
model2 = brm(correlator_s ~ obj_s + (1+obj_s|Family) , family="bernoulli", data=corr_pair, prior=set_prior("normal(0,10)"))

dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")



sink("results-ground.tsv")
cat("")
sink()

cat(paste("dependency", "satisfiedFraction", "posteriorMean", "posteriorSD", "num_langs", "posteriorOpposite", sep="\t"), file="results-ground.tsv", append=TRUE, sep="\n")


for(dependency in dependencies) {
   corr_pair = getCorrPair(dependency)
   model2 = update(model2, newdata=corr_pair, iter=4000)
   summary(model2)
   
   samples = posterior_samples(model2, "b_obj_s")[,]
   posteriorOpposite = ecdf(samples)(0.0)
   posteriorMean = mean(samples)
   posteriorSD = sd(samples)
   satisfiedFraction = mean((corr_pair$correlator_s == corr_pair$obj_s), na.rm=TRUE)
   cat(paste(dependency, satisfiedFraction, posteriorMean, posteriorSD, nrow(corr_pair), posteriorOpposite, sep="\t"), file="results-ground.tsv", append=TRUE, sep="\n")
}


