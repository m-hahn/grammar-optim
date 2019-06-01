# Visualizes posteriors per relation, for optimized and real grammars


data = read.csv("../../../grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)



library(dplyr)
library(tidyr)
library(ggplot2)

data = data %>% mutate(DH_Weight = DH_Mean_NoPunct)


library(forcats)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek" = "Ancient"))

languages = read.csv("../../languages/languages-iso_codes.tsv", sep=",")
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


data = data %>% select(Dependency, Family, Language, FileName, DH_Weight)

D = data %>% filter(Dependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
data = merge(data, D, by=c("FileName"))

data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))

#reverseOrder = c("aux")
#data[data$Dependency %in% reverseOrder,]$Agree = 1-data[data$Dependency %in% reverseOrder,]$Agree


D = data %>% group_by(Language, Family, Dependency) %>% summarise(Agree = mean(Agree)) %>% group_by(Dependency) %>% summarise(Agree = mean(Agree)) %>% mutate(Type = "UD Corpora")


E = data %>% group_by(Language, Family, Dependency) %>% summarise(Agree = mean(Agree)) %>% group_by(Dependency, Family) %>% summarise(Agree = mean(Agree)) %>% group_by(Dependency) %>% summarise(Agree = mean(Agree)) %>% mutate(Type = "UD Corpora Balanced")



#################################


dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")



#balanced = read.csv("results-ground-agree.tsv", sep="\t")
#balanced = balanced %>% 

library(dplyr)
library(tidyr)
library(ggplot2)




library(dplyr)
library(tidyr)
library(ggplot2)



type = "Efficiency"
dependency = "acl"

for(type in c("Efficiency", "Predictability", "Parseability", "DependencyLength")) {
  if(type == "Efficiency") {
  	color = "blue"
        typeName = "efficiency"
  } else if(type == "Predictability") {
  	color = "red"
        typeName = "langmod"
  } else if (type == "Parseability") {
  	color = "green"
        typeName = "parseability"
  } else if (type == "DependencyLength") {
        color = "orange"
        typeName = "depl"
  }
  for(dependency in dependencies) {
     data =  read.csv(paste("~/CS_SCR/posteriors/posterior-", dependency, "-", typeName, "-large.csv", sep=""))
     data = data %>% mutate(Prevalence = 1/(1+exp(-b_Intercept)))
     if(dependency == "aux") {
         data = data %>% mutate(Prevalence=1-Prevalence)
     }
     plot = ggplot(data=data)
     plot = plot  + geom_density(aes(x=Prevalence, y=..scaled..), alpha=.5, fill=color)
     plot = plot + xlim(0,1)
     plot = plot + theme_classic()
     plot = plot + theme_void()
     plot = plot  + theme(legend.position="none")
     plot = plot + geom_segment(aes(x=0.5, xend=0.5, y=0, yend=1), linetype=2)
     ggsave(paste("../figures/posteriors/posterior_perRelation_", type, "_", dependency, ".pdf", sep=""), plot=plot, height=1, width=2)
  }
}







type = "Real"
for(dependency in dependencies) {
data = D %>% filter(Dependency==dependency)
     if(dependency == "aux") {
         data = data %>% mutate(Agree=1-Agree)
     }

   plot = ggplot(data=data)
   plot = plot + geom_bar(stat="identity", width = 0.1, aes(x=Agree, y=1))
   plot = plot + xlim(0,1)
   plot = plot + theme_classic()
   plot = plot + theme_void()
   plot = plot  + theme(legend.position="none")
   plot = plot + geom_segment(aes(x=0.5, xend=0.5, y=0, yend=1), linetype=2)
   plot = plot + theme(axis.line.x = element_line(colour = "black"))
   ggsave(paste("../figures/posteriors/posterior_perRelation_", type, "_", dependency, ".pdf", sep=""), plot=plot, height=1, width=2)
}





