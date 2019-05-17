# For the first \lambda=0.9 experiment



dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")

dependencies =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")

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


cat("\nReading posterior samples\n")
#parse = read.csv("/home/user/CS_SCR/posteriors/posterior-10-parseability.csv")
two = read.csv("~/CS_SCR/posteriors/posterior-10-first10-efficiency.csv")
#langmod = read.csv("/home/user/CS_SCR/posteriors/posterior-10-langmod.csv")



data = rbind(two %>% mutate(Type="Efficiency"))

data2 = data %>% gather(Dependency, Intercept, b_acl_Intercept                             ,b_aux_Intercept                                       ,  b_liftedcase_Intercept,b_liftedcop_Intercept  ,                                 b_liftedmark_Intercept  ,                                b_nmod_Intercept         ,                                                             b_obl_Intercept            ,                             b_xcomp_Intercept           )

data2 = data2 %>% mutate(Prevalence = 1/(1+exp(-Intercept)))

data2 = data2 %>% mutate(Dependency = substring(Dependency,3, nchar(Dependency)-10))
data2 = data2 %>% mutate(Dependency = case_when(Dependency == "liftedcase" ~ "lifted_case", Dependency == "liftedcop" ~ "lifted_cop", Dependency == "liftedmark" ~ "lifted_mark", TRUE ~ Dependency))

data2 = data2 %>% mutate(Prevalence = case_when(Dependency == "aux" ~ 1-Prevalence, TRUE ~ Prevalence))

  plot = ggplot(data=data2) + geom_density(aes(x=Prevalence, y=..scaled.., fill=Type, group=Type), alpha=.5) + xlim(0,1) +  facet_wrap( ~ Dependency, ncol=1) 
  ggsave(plot, file=paste("figures/posterior_joint_first10" , ".pdf", sep=""))





####################



  plot = ggplot(data=data2) + geom_density(aes(x=Prevalence, y=..scaled.., fill=Type, group=Type), alpha=.5) + xlim(0,1) + geom_bar(data=D, stat="identity", width = 0.01, aes(x=Agree, y=1, fill=Type, group=Type)) + facet_wrap( ~ Dependency, ncol=1) 
  ggsave(plot, file=paste("figures/posterior_joint_with_ud_first10" , ".pdf", sep=""))

#  plot = ggplot(data=data2) + geom_density(aes(x=Prevalence, y=..scaled.., fill=Type, group=Type), alpha=.5) + xlim(0,1) + geom_bar(data=D, stat="identity", width = 0.01, aes(x=Agree, y=1, fill=Type, group=Type)) + facet_wrap( ~ Dependency, ncol=1) + geom_bar(data=E, stat="identity", width = 0.01, aes(x=Agree, y=1, fill=Type, group=Type))
#  ggsave(plot, file=paste("posterior/posterior_joint_with_ud_balanced" , ".pdf", sep=""))


#for(type in c("
  type = "Efficiency"
  dependency = "acl"

  dependencies =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")
for(type in c("Efficiency")) {
		color = "blue"
  for(dependency in dependencies) {
     plot = ggplot(data=data2 %>% filter(Type==type, Dependency==dependency))
     plot = plot  + geom_density(aes(x=Prevalence, y=..scaled.., fill=Type, group=Type), alpha=.5, fill=color)
     plot = plot + xlim(0,1)
     plot = plot + theme_classic()
     plot = plot + theme_void()
     plot = plot  + theme(legend.position="none")
     plot = plot + geom_segment(aes(x=0.5, xend=0.5, y=0, yend=1), linetype=2)
     ggsave(paste("figures/posteriors/posterior_first10_", type, "_", dependency, ".pdf", sep=""), plot=plot, height=1, width=2)
  }
}

