
data = read.csv("../../grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)



library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)

data = data %>% mutate(DH_Weight = DH_Mean_NoPunct)


library(forcats)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek" = "Ancient"))

languages = read.csv("../languages/languages-iso_codes.tsv", sep=",")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)




dependency = "nmod"

getCorrPair = function(dependency) {
   corr_pair = data %>% filter((Dependency == dependency) | (Dependency == "obj"))
   corr_pair = unique(corr_pair %>% select(Family, Language, FileName, Dependency, DH_Weight, ModelName)) %>% spread(Dependency, DH_Weight)
   corr_pair$correlator = corr_pair[[dependency]]
   corr_pair[[dependency]] = NULL
   corr_pair = corr_pair %>% mutate(correlator_s = pmax(0,sign(correlator)), obj_s=pmax(0,sign(obj)))
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
mean_obj_s = mean(corr_pairs$obj_s)
corr_pairs[corr_pairs$obj_s == 1,]$weight = 0.5/mean_obj_s # * corr_pairs[corr_pairs$obj_s == 1,]$obj_s
corr_pairs = corr_pairs[!is.na(corr_pairs$correlator_s),]
#corr_pairs[corr_pairs$correlator_s == 1,]$correlator_s = 0.5/mean_obj_s * corr_pairs[corr_pairs$correlator_s == 1,]$correlator_s

plot = ggplot(corr_pairs, aes(x=obj_s, y=correlator_s, group=Dependency)) + geom_count(aes(size=..n..)) + facet_wrap(~Dependency) + xlim(-0.5, 1.5) + ylim(-0.5, 1.5)

corr_pairs = corr_pairs %>% mutate(agree = (obj_s == correlator_s))
corr_pairs_sum = corr_pairs %>% group_by(Dependency, obj_s, correlator_s) %>% summarise(agree=mean(agree), count = sum(weight))
corr_pairs_sum = corr_pairs_sum %>% mutate(agree_color = ifelse(agree, "red", "blue"))
plot = ggplot(corr_pairs_sum, aes(x=obj_s, y=correlator_s, group=Dependency)) + geom_point(aes(size=count, color=agree_color)) + facet_wrap(~Dependency) + theme_bw() + xlim(-0.5, 1.5) + ylim(-0.5, 1.5)  + theme(legend.position="none")





