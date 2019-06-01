# For the first \lambda=0.9 experiment



dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")

dependencies =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")

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



ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")



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


library(dplyr)
library(tidyr)
library(ggplot2)




library(dplyr)
library(tidyr)
library(ggplot2)


cat("\nReading posterior samples\n")
u = read.csv("/home/user/CS_SCR/posteriors/posterior-10-efficiency.csv")



u$satisfied = 8 - ((u$b_acl_Intercept < 0) + (u$b_aux_Intercept > 0 ) + (u$b_liftedcase_Intercept < 0 ) + (u$b_liftedcop_Intercept < 0 ) + (u$b_liftedmark_Intercept < 0 ) + (u$b_nmod_Intercept < 0 ) + (u$b_obl_Intercept < 0 ) + (u$b_xcomp_Intercept < 0 ))


library(ggplot2)

u$SamplesNum = NROW(u)
data2 = u %>% group_by(satisfied) %>% summarise(SamplesNum = mean(SamplesNum), posterior = NROW(satisfied)) %>% mutate(posteriorProb = posterior/SamplesNum)
u = NULL




plot = ggplot(data = data2, aes(x=satisfied, y=posteriorProb)) + geom_bar(stat="identity") + theme_bw() 
plot = plot + xlim(0,8.5)
plot = plot + xlab("Number of Predicted Correlations") 
plot = plot + ylab("Posterior")
plot = plot + geom_text(aes(label=posteriorProb), vjust=0, size=3.3)
plot = plot + ylim(0, 1.1) 
plot = plot + theme(text = element_text(size=14),
        axis.text.x = element_text(angle=90, hjust=1)) 
plot = plot + theme(legend.position="none")

ggsave(plot=plot, filename="../figures/posterior-satisfied-universals-together-large-prior-efficiency09.pdf", width=3, height=3)







