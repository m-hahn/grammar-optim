
# _final/
data = read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)

library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)

data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))




dryer_greenberg_fine = data



languages = read.csv("../languages/languages-iso_codes.tsv")
dryer_greenberg_fine  = merge(dryer_greenberg_fine, languages, by=c("Language"), all.x=TRUE)



#options(mc.cores = parallel::detectCores())
#rstan_options(auto_write = TRUE)


dataS = read.csv("../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = rbind(dataS, dataS2) %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_balanced")

dataP = read.csv("../../grammars/plane/plane-parse.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP2 = read.csv("../../grammars/plane/plane-parse-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = rbind(dataP, dataP2) %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_balanced")


dataE = merge(dataS, dataP, by=c("Type", "Language", "Model")) %>% mutate(Eff = Pars+0.9*Surp) %>% group_by(Type, Language, Model) %>% summarise(Eff = mean(Eff))

dataE = dataE[order(dataE$Language, dataE$Model),]

eff = dataE$Eff
models = c()
for(i in (1:202)) {
   if(eff[2*i] < eff[2*i+1]) {
	   models = c(models, dataE$Model[2*i])
   } else {
	   models = c(models, dataE$Model[2*i+1])
   }
}


dryer_greenberg_fine = dryer_greenberg_fine%>% filter(FileName %in% models)




dependency = "nmod"

getCorrPair = function(dependency) {
   corr_pair = dryer_greenberg_fine %>% filter((CoarseDependency == dependency) | (CoarseDependency == "obj"))
   corr_pair = unique(corr_pair %>% select(Family, Language, FileName, CoarseDependency, DH_Weight )) %>% spread(CoarseDependency, DH_Weight)
   corr_pair$correlator = corr_pair[[dependency]]
   corr_pair[[dependency]] = NULL
   corr_pair = corr_pair %>% mutate(correlator_s = pmax(0,sign(correlator)), obj_s=pmax(0,sign(obj)))
#   corr_pair = corr_pair %>% mutate(obj_s = obj_s - mean(obj_s))
   corr_pair = corr_pair %>% mutate(correlator_s = ifelse(correlator == 0, NA, correlator_s)) # special case of actual zero, indicating fully random ordering: has to be excluded from analysis. Such cases occur when a dependency does not occur in the training partition, but does occur in the validation partition.
   if(dependency == "aux") {
       corr_pair = corr_pair %>% mutate(correlator_s=1-correlator_s)
   }

   return(corr_pair)
}

corr_pair = getCorrPair("lifted_cop")

dependencies = c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod",  "obl", "xcomp")
# advmod, nsubj

corr_pairs = data.frame()
for(dependency in dependencies) {
   corr_pair = getCorrPair(dependency)
   corr_pair$Dependency = dependency
   corr_pairs = rbind(corr_pairs, corr_pair)
}

corr_pairs$weight = 1
#mean_obj_s = mean(corr_pairs$obj_s)
#corr_pairs[corr_pairs$obj_s == 1,]$weight = 0.5/mean_obj_s # * corr_pairs[corr_pairs$obj_s == 1,]$obj_s
#corr_pairs = corr_pairs[!is.na(corr_pairs$correlator_s),]
##corr_pairs[corr_pairs$correlator_s == 1,]$correlator_s = 0.5/mean_obj_s * corr_pairs[corr_pairs$correlator_s == 1,]$correlator_s
#
#plot = ggplot(corr_pairs, aes(x=obj_s, y=correlator_s, group=Dependency)) + geom_count(aes(size=..n..)) + facet_wrap(~Dependency) + xlim(-0.5, 1.5) + ylim(-0.5, 1.5)

corr_pairs = corr_pairs %>% mutate(agree = (obj_s == correlator_s))
corr_pairs_sum = corr_pairs %>% group_by(Dependency, obj_s, correlator_s) %>% summarise(agree=mean(agree), count = sum(weight))
corr_pairs_sum = corr_pairs_sum %>% mutate(agree_color = ifelse(agree, "red", "blue"))
plot = ggplot(corr_pairs_sum, aes(x=obj_s, y=correlator_s, group=Dependency)) + geom_point(aes(size=count, color=agree_color)) + facet_wrap(~Dependency) + theme_bw() + xlim(-0.5, 1.5) + ylim(-0.5, 1.5)  + theme(legend.position="none")





