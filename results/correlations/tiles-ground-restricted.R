
data = read.csv("../../grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)



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
   corr_pair = corr_pair %>% mutate(correlator_s = pmax(0,sign(correlator)), obj_s=pmax(0,sign(obj)))
   return(corr_pair)
}

corr_pair = getCorrPair("nmod")

dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")

for(dependency in dependencies) {
   corr_pair = getCorrPair(dependency)
   satisfiedFraction = mean((corr_pair$correlator_s == corr_pair$obj_s), na.rm=TRUE)
   cat(paste(dependency, satisfiedFraction, sep="\t"), sep="\n") #, file="results-ground.tsv", append=TRUE, sep="\n")
}



ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")

# , "amod", "nummod"


data = data %>% select(Dependency, Family, Language, FileName, DH_Weight)

D = data %>% filter(Dependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
data = merge(data, D, by=c("FileName"))

data = data %>% filter(Dependency %in% ofInterest)

data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))

reverseOrder = c("aux")
data[data$Dependency %in% reverseOrder,]$Agree = 1-data[data$Dependency %in% reverseOrder,]$Agree


D = data %>% group_by(Language, Family, Dependency) %>% summarise(Agree = mean(Agree))


library(scales)

D$AgreeB = as.factor(as.character(round(pmax(0, pmin(1, D$Agree)))))
plot = ggplot(D, aes(x = Dependency, y = Language)) + 
#  geom_tile(aes(colour="white")) +
  geom_point(aes(colour = AgreeB, size =1)) +
  scale_fill_manual(values=c("red", "blue")) +
  theme_bw() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none")

ggsave(file="figures/coverage-ground-circles.pdf", plot=plot)


#E = D %>% group_by(Language) %>% mutate(AgreeSum = sum(Agree))
##E = E[order(E$AgreeSum),]
#ordered_languages = unique(E$Language)
##write.csv(ordered_languages, file="ordered_languages.csv")
#
#E = E %>% group_by(Dependency) %>% mutate(AgreeSumDep = sum(Agree))
#E = E[order(-E$AgreeSumDep),]
#ordered_deps = unique(E$Dependency)
#write.csv(ordered_deps, file="ordered_dependencies.csv")
#
#
#
#plot = ggplot(E, aes(x = factor(Dependency, levels=ordered_deps), y = factor(Language, levels=ordered_languages))) + 
#  geom_tile(aes(fill=Agree)) + 
#  scale_fill_gradient(low="white", high="red", limits=c(0.5, 1), oob=squish) +
#  labs(x="Correlations", y="Languages", title="Matrix") +
#  theme_bw() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
#                     axis.text.y=element_text(size=9),
#                     plot.title=element_text(size=11))
#
#
#ggsave(file="coverage-ground-ordered.png", plot=plot)
#
#
#
