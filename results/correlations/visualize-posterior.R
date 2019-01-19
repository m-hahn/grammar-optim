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



ofInterest =  c("acl", "advmod", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "nsubj", "obl", "xcomp")
#ofInterest =  c("acl",  "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")

# , "amod", "nummod"


data = data %>% select(Dependency, Family, Language, FileName, DH_Weight)

D = data %>% filter(Dependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
data = merge(data, D, by=c("FileName"))

data = data %>% filter(Dependency %in% ofInterest)

data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))

reverseOrder = c("aux")
data[data$Dependency %in% reverseOrder,]$Agree = 1-data[data$Dependency %in% reverseOrder,]$Agree


D = data %>% group_by(Language, Family, Dependency) %>% summarise(Agree = mean(Agree)) %>% group_by(Dependency) %>% summarise(Agree = mean(Agree)) %>% mutate(Type = "UD Corpora")


dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")

dependencies =  c("acl", "advmod", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "nsubj", "obl", "xcomp")


#balanced = read.csv("results-ground-agree.tsv", sep="\t")
#balanced = balanced %>% 

library(dplyr)
library(tidyr)
library(ggplot2)


data2 = data.frame(X=c(), x=c(), Type = c(), Dependency = c())

for(dependency in dependencies) {
   two = read.csv(paste("/home/user/CS_SCR/samples/agree_", "two", "_", dependency, ".csv", sep=""))
   parse = read.csv(paste("/home/user/CS_SCR/samples/agree_", "parse", "_", dependency, ".csv", sep=""))
   langmod = read.csv(paste("/home/user/CS_SCR/samples/agree_", "langmod", "_", dependency, ".csv", sep=""))
 
   data = rbind(two %>% mutate(Type = "two"), parse %>% mutate(Type = "parse"), langmod %>% mutate(Type = "langmod"))
   if(dependency == "aux") {
   	data = data %>% mutate(x=1-x)
   }
   data2 = rbind(data %>% mutate(Dependency=dependency), data2)
   plot = ggplot(data=data) + geom_density(aes(x=x, y=..scaled.., fill=Type, group=Type), alpha=.5) + xlim(0,1)
   ggsave(plot, file=paste("figures/posterior_", dependency, ".pdf", sep=""), width=9, height=2)
}

  plot = ggplot(data=data2) + geom_density(aes(x=x, y=..scaled.., fill=Type, group=Type), alpha=.5) + xlim(0,1) + geom_bar(data=D, stat="identity", width = 0.01, aes(x=Agree, y=1, fill=Type, group=Type)) + facet_wrap( ~ Dependency, ncol=1) 
  ggsave(plot, file=paste("figures/posterior_with_ud" , ".pdf", sep=""))




