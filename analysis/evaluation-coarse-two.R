
data = read.csv("CS_SCR/deps/manual_output_funchead_two_coarse/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)



library(dplyr)
library(tidyr)
library(ggplot2)

dryer_greenberg_fine = data



languages = read.csv("languages.tsv", sep="\t")
dryer_greenberg_fine  = merge(dryer_greenberg_fine, languages, by=c("Language"), all.x=TRUE)


library("brms")

#options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

dependency = "nmod"

getCorrPair = function(dependency) {
   corr_pair = dryer_greenberg_fine %>% filter((CoarseDependency == dependency) | (CoarseDependency == "obj"))
   corr_pair = unique(corr_pair %>% select(Family, Language, FileName, CoarseDependency, DH_Weight, ModelName)) %>% spread(CoarseDependency, DH_Weight)
   corr_pair$correlator = corr_pair[[dependency]]
   corr_pair = corr_pair %>% mutate(correlator_s = pmax(0,sign(correlator)), obj_s=pmax(0,sign(obj)))
   return(corr_pair)
}

corr_pair = getCorrPair("nmod")
model2 = brm(correlator_s ~ obj_s + (1+obj_s|Family) + (1+obj_s|Language), family="bernoulli", data=corr_pair)

dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")



sink("results-two.tsv")
cat("")
sink()

cat(paste("dependency", "satisfiedFraction", "posteriorMean", "posteriorSD", "posteriorOpposite", sep="\t"), file="results-two.tsv", append=TRUE, sep="\n")


for(dependency in dependencies) {
   corr_pair = getCorrPair(dependency)
   model2 = update(model2, newdata=corr_pair, iter=4000)
   summary(model2)
   
   samples = posterior_samples(model2, "b_obj_s")[,]
   posteriorOpposite = ecdf(samples)(0.0)
   posteriorMean = mean(samples)
   posteriorSD = sd(samples)
   satisfiedFraction = mean((corr_pair$correlator_s == corr_pair$obj_s), na.rm=TRUE)
   cat(paste(dependency, satisfiedFraction, posteriorMean, posteriorSD, posteriorOpposite, sep="\t"), file="results-two.tsv", append=TRUE, sep="\n")
}


