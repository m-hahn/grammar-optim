
# _final/
data = read.csv("../../grammars/manual_output_funchead_langmod_coarse_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)

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

mean(getCorrPair("lifted_cop")$agree)




dataS = read.csv("../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = rbind(dataS, dataS2) %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced")

dataE = dataS %>% group_by(Type, Language, Model) %>% summarise(Surp = mean(Surp))

dataE = dataE[order(dataE$Language, dataE$Model),]

eff = dataE$Surp
models = c()
for(i in (1:202)) {
   if(eff[2*i] < eff[2*i+1]) {
	   models = c(models, dataE$Model[2*i])
   } else {
	   models = c(models, dataE$Model[2*i+1])
   }
}


dryer_greenberg_fine = dryer_greenberg_fine%>% filter(FileName %in% models)



