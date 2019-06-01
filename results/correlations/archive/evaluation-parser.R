
# _final/
data = read.csv("../../grammars/manual_output_funchead_two_coarse_parser_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)

library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)

data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))




dryer_greenberg_fine = data



languages = read.csv("../languages/languages-iso_codes.tsv")
dryer_greenberg_fine  = merge(dryer_greenberg_fine, languages, by=c("Language"), all.x=TRUE)


library("brms")

#options(mc.cores = parallel::detectCores())
#rstan_options(auto_write = TRUE)

dependency = "nmod"

getCorrPair = function(dependency) {
   corr_pair = dryer_greenberg_fine %>% filter((CoarseDependency == dependency) | (CoarseDependency == "obj"))
   corr_pair = unique(corr_pair %>% select(Family, Language, FileName, CoarseDependency, DH_Weight )) %>% spread(CoarseDependency, DH_Weight)
   corr_pair$correlator = corr_pair[[dependency]]
   corr_pair = corr_pair %>% mutate(correlator_s = pmax(0,sign(correlator)), obj_s=pmax(0,sign(obj)))
#   corr_pair = corr_pair %>% mutate(obj_s = obj_s - mean(obj_s))
   corr_pair = corr_pair %>% mutate(correlator_s = ifelse(correlator == 0, NA, correlator_s)) # special case of actual zero, indicating fully random ordering: has to be excluded from analysis. Such cases occur when a dependency does not occur in the training partition, but does occur in the validation partition.
   corr_pair$agree = (corr_pair$correlator_s == corr_pair$obj_s)
   return(corr_pair)
}

corr_pair = getCorrPair("lifted_cop")
#model2 = brm(correlator_s ~ obj_s + (1+obj_s|Family) + (1+obj_s|Language), family="bernoulli", data=corr_pair, iter=5000)

# Simpler analysis for posterior prevalence of correlation
model3 = brm(agree ~ (1|Family) + (1|Language), family="bernoulli", data=corr_pair)
   samples = posterior_samples(model3, "b_Intercept")[,]
   posteriorOpposite = ecdf(samples)(0.0)


# The full set of UD relations
dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")

# Relations formalizing Dryer's correlations
#dependencies = c("acl", "advmod", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "nsubj", "obl", "xcomp")


outpath = "output/results-prevalence-parser-best.tsv"
sink(outpath)
cat("")
sink()

cat(paste("dependency", "satisfiedFraction", "posteriorMean", "posteriorSD", "posteriorOpposite", sep="\t"), file=outpath, append=TRUE, sep="\n")


for(dependency in dependencies) {
   corr_pair = getCorrPair(dependency)
   model3 = update(model3, newdata=corr_pair, iter=5000) # potentially add , control=list(adapt_delta=0.9)
   summary(model3)
   
   samples = posterior_samples(model3, "b_Intercept")[,]
   posteriorOpposite = ecdf(samples)(0.0)
   posteriorMean = mean(samples)
   posteriorSD = sd(samples)
   satisfiedFraction = mean((corr_pair$correlator_s == corr_pair$obj_s), na.rm=TRUE)
   cat(paste(dependency, satisfiedFraction, posteriorMean, posteriorSD, posteriorOpposite, sep="\t"), file=outpath, append=TRUE, sep="\n")
}


