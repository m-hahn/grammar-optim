# per language


library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
# Read predictability estimates
dataS =  read.csv("../../../grammars/plane/controls/plane-fixed-nondeterministic.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_nondet", Type)) %>% mutate(Model = paste(Model, "nondet", sep="_"))
dataS1 = read.csv("../../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(Model = paste(Model, "det", sep="_"))
dataS2 = read.csv("../../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(Model = paste(Model, "det", sep="_"))
dataS3 = read.csv("../../../grammars/plane/plane-fixed-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(FullSurp=NULL) %>% mutate(Model = paste(Model, "det", sep="_"))
dataS4 = read.csv("../../../grammars/plane/plane-fixed-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(FullSurp=NULL) %>% mutate(Model = paste(Model, "det", sep="_"))
dataS5 = read.csv("../../../grammars/plane/plane-fixed-random3.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(FullSurp=NULL) %>% mutate(Model = paste(Model, "det", sep="_"))
dataS6 = read.csv("../../../grammars/plane/plane-fixed-random4.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(FullSurp=NULL) %>% mutate(Model = paste(Model, "det", sep="_"))
dataS7 = read.csv("../../../grammars/plane/plane-fixed-random5.tsv", sep="\t") %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(FullSurp=NULL) %>% mutate(Model = paste(Model, "det", sep="_"))
dataS = rbind(dataS, dataS1, dataS2, dataS3, dataS4, dataS5, dataS6, dataS7)
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surprisal = mean(Surp, na.rm=TRUE))
#dataS = as.data.frame(dataS) %>% filter(Language %in% c("Czech", "English", "Japanese"))

# Read parseability estimates
dataP =  read.csv("../../../grammars/plane/controls/plane-parse-nondeterministic.tsv", sep="\t")  %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_nondet", Type)) %>% mutate(FullSurprisal=NULL) %>% mutate(Model = paste(Model, "nondet", sep="_"))
dataP2 = read.csv("../../../grammars/plane/plane-parse-redone.tsv", sep="\t")  %>% mutate(Model = as.character(Model), Type=as.character(Type)) %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM_det", Type)) %>% mutate(FullSurprisal=NULL) %>% mutate(Model = paste(Model, "det", sep="_"))
dataP = rbind(dataP, dataP2) 
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS, na.rm=TRUE), Pars = mean(Pars, na.rm=TRUE))
#dataP = as.data.frame(dataP) %>% filter(Language %in% c("Czech", "English", "Japanese"))

# Ensure everything is treated as a string, not a factor, to ensure the frames can be merged
dataS = as.data.frame(dataS) %>% mutate(Type = as.character(Type))
dataP = as.data.frame(dataP) %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)



dataBaseline = data %>% filter(Type == "manual_output_funchead_RANDOM_nondet")
dataGround = data %>% filter(Type == "REAL_REAL") %>% select(Language, Surprisal, Pars) %>% rename(SurprisalGround = Surprisal) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurprisalGround) %>% group_by(Language)
data = merge(dataBaseline, dataGround, by=c("Language"))


write.csv(data, file="analyze_pareto_optimality/pareto-data.tsv")




