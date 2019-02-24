
# _final/
data = read.csv("../../grammars/manual_output_funchead_langmod_coarse_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)

library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)


best = read.csv("../strongest_models/best-langmod-best-balanced.csv")

data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))

data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))



dryer_greenberg_fine = data



languages = read.csv("../languages/languages-iso_codes.tsv")
dryer_greenberg_fine  = merge(dryer_greenberg_fine, languages, by=c("Language"), all.x=TRUE)



#options(mc.cores = parallel::detectCores())
#rstan_options(auto_write = TRUE)

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

   corr_pair$agree = (corr_pair$correlator_s == corr_pair$obj_s)
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

for(dependency in dependencies) {
   plot = ggplot(corr_pairs_sum %>% filter(Dependency == dependency), aes(x=obj_s, y=correlator_s)) + geom_point(aes(size=count, color=agree_color))  + theme_bw() + xlim(-0.5, 1.5) + ylim(-0.5, 1.5)  + theme(legend.position="none") + xlab(NULL) + ylab(NULL)+
  theme(axis.title=element_blank(),
        axis.text=element_blank(),
        axis.ticks.x=element_blank(),
        axis.ticks=element_blank()) +  theme(panel.border = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank())
  ggsave(plot, file=paste("figures/correlations/correlation-langmod-", dependency, ".pdf", sep=""), width=1, height=1)


}






