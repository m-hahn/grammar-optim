
data = read.csv("../../grammars/manual_output_funchead_langmod_coarse_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)

best = read.csv("../strongest_models/best-langmod-best-balanced.csv")


library(dplyr)
library(tidyr)
library(ggplot2)

library(forcats)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek"= "Ancient"))





data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))

data = data



languages = read.csv("../languages/languages-iso_codes.tsv", sep=",")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)




dependency = "nmod"

getCorrPair = function(dependency) {
   corr_pair = data %>% filter((CoarseDependency == dependency) | (CoarseDependency == "obj"))
   corr_pair = unique(corr_pair %>% select(Family, Language, FileName, CoarseDependency, DH_Weight)) %>% spread(CoarseDependency, DH_Weight)
   corr_pair$correlator = corr_pair[[dependency]]
   corr_pair = corr_pair %>% mutate(correlator_s = pmax(0,sign(correlator)), obj_s=pmax(0,sign(obj)))
   return(corr_pair)
}

corr_pair = getCorrPair("nmod")

dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")

for(dependency in dependencies) {
   corr_pair = getCorrPair(dependency)
   satisfiedFraction = mean((corr_pair$correlator_s == corr_pair$obj_s), na.rm=TRUE)
   cat(paste(dependency, satisfiedFraction, sep="\t"), sep="\n") #, file="results-two.tsv", append=TRUE, sep="\n")
}



ofInterest =  c("acl", "advmod", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "nsubj", "obl", "xcomp")
#ofInterest =  c("acl",  "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")

# , "amod", "nummod"


data = data %>% select(CoarseDependency, Family, Language, FileName, DH_Weight)

D = data %>% filter(CoarseDependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
data = merge(data, D, by=c("FileName"))

data = data %>% filter(CoarseDependency %in% ofInterest)

data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))

reverseOrder = c("aux")
data[data$CoarseDependency %in% reverseOrder,]$Agree = 1-data[data$CoarseDependency %in% reverseOrder,]$Agree


D = data %>% group_by(Language, Family, CoarseDependency) %>% summarise(Agree = mean(Agree))

library(lme4)
#summary(glmer(Dir ~ DirObj + (1+DirObj|Family), data=data %>% filter(CoarseDependency == "acl"), family="binomial"))
#summary(glmer(Dir ~ DirObj + (1+DirObj|Family), data=data %>% filter(CoarseDependency == "advmod"), family="binomial"))
#summary(glmer(Dir ~ DirObj + (1+DirObj|Family), data=data %>% filter(CoarseDependency == "aux"), family="binomial"))
#summary(glmer(Dir ~ DirObj + (1+DirObj|Family), data=data %>% filter(CoarseDependency == "lifted_case"), family="binomial"))
#summary(glmer(Dir ~ DirObj + (1+DirObj|Family), data=data %>% filter(CoarseDependency == "lifted_cop"), family="binomial"))
#summary(glmer(Dir ~ DirObj + (1+DirObj|Family), data=data %>% filter(CoarseDependency == "lifted_mark"), family="binomial"))
#summary(glmer(Dir ~ DirObj + (1+DirObj|Family), data=data %>% filter(CoarseDependency == "nmod"), family="binomial"))




library(scales)

plot = ggplot(D, aes(x = CoarseDependency, y = Language)) + 
  geom_tile(aes(fill=Agree)) + 
  scale_fill_gradient(low="white", high="red", limits=c(0.5, 1), oob=squish) +
  theme_bw() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none")

ggsave(file="figures/coverage-langmod-best.pdf", plot=plot)

#
#
#
#ordered_languages = (read.csv("ordered_languages.csv"))$x
#ordered_deps = (read.csv("ordered_dependencies.csv"))$x
#
#plot = ggplot(D, aes(x = factor(CoarseDependency, levels=ordered_deps), y = factor(Language, levels=ordered_languages))) + 
#  geom_tile(aes(fill=Agree)) + 
#  scale_fill_gradient(low="white", high="red", limits=c(0.5, 1), oob=squish) +
#  labs(x="Correlations", y="Languages", title="Matrix") +
#  theme_bw() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
#                     axis.text.y=element_text(size=9),
#                     plot.title=element_text(size=11))
#
#ggsave(file="coverage-langmod-ordered-best.png", plot=plot)
#
#
