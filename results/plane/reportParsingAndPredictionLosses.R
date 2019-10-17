# per language


library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
depl = read.csv("../../grammars/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(tidyr)
library(dplyr)
depl = depl %>% filter(grepl("FuncHead", ModelName)) %>% filter(grepl("Coarse", ModelName))
dataS = read.csv("../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS3 = read.csv("../../grammars/plane/plane-fixed-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS4 = read.csv("../../grammars/plane/plane-fixed-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS5 = read.csv("../../grammars/plane/plane-fixed-random3.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS6 = read.csv("../../grammars/plane/plane-fixed-random4.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS7 = read.csv("../../grammars/plane/plane-fixed-random5.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS = rbind(dataS, dataS2, dataS3, dataS4, dataS5, dataS6, dataS7)
#dataP = read.csv("../../grammars/plane/plane-parse-unified.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = read.csv("../../grammars/plane/plane-parse-redone.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surprisal = mean(Surp, na.rm=TRUE))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(Pars = mean(Pars, na.rm=TRUE), LAS = mean(LAS, na.rm=TRUE), UAS = mean(UAS, na.rm=TRUE), ParsU = mean(ParsU, na.rm=TRUE))
dataS = as.data.frame(dataS)
dataP = as.data.frame(dataP)
library(lme4)
dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)


data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM2", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM3", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM4", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM5", "manual_output_funchead_RANDOM", as.character(Type)))





dataComp = data
dataCompBaseline = dataComp %>% filter(grepl("RANDOM", Type))
dataCompReal = dataComp %>% filter(Type == "REAL_REAL") %>% select(Language, Surprisal, Pars) %>% rename(SurprisalGround = Surprisal) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurprisalGround) %>% group_by(Language)
dataCompGround = dataComp %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% select(Language, Surprisal, Pars) %>% rename(SurprisalGround = Surprisal) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurprisalGround) %>% group_by(Language)
dataComp = merge(dataCompBaseline, dataCompGround, by=c("Language"))

dataComp$Eff = dataComp$Pars + 0.9*dataComp$Surprisal

u = dataComp %>% group_by(Language) %>% summarise(BetterSurprisal = sum(Surprisal > SurprisalGround, na.rm=TRUE), WorseSurprisal = sum(Surprisal <= SurprisalGround, na.rm=TRUE), TotalSurprisal = BetterSurprisal+WorseSurprisal, BetterFracSurprisal = BetterSurprisal/TotalSurprisal, BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars, BetterEff = sum(Eff > EffGround, na.rm=TRUE), WorseEff = sum(Eff <= EffGround, na.rm=TRUE), TotalEff = BetterEff+WorseEff, BetterFracEff = BetterEff/TotalEff )





data = merge(data, depl %>% select(Language, Model,AverageLengthPerWord) %>% mutate(Model = as.character(Model)), by=c("Language", "Model"), all.x=TRUE)
data = data %>% mutate(Two = 0.9*Surprisal+Pars)

data2 = rbind(data)
data2 = data2 %>% group_by(Language, Type) %>% summarise(Surprisal=mean(Surprisal, na.rm=TRUE), Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% mutate(MeanSurprisal = mean(Surprisal, na.rm=TRUE), SDSurprisal = sd(Surprisal, na.rm=TRUE)) %>% mutate(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE))

dataPBest = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% group_by(Language) %>% summarise(Pars = min(Pars))
data2Best = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% group_by(Language) %>% summarise(Two = min(Two, na.rm=TRUE))
dataSBest = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% group_by(Language) %>% summarise(Surprisal = min(Surprisal, na.rm=TRUE))

dataP = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced")
data2 = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large")
dataS = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced")

dataP = merge(dataP, dataPBest, by=c("Language", "Pars"))
data2 = merge(data2, data2Best, by=c("Language", "Two"))
dataS = merge(dataS, dataSBest, by=c("Language", "Surprisal"))


dataRandom = data %>% filter(grepl("RANDOM", Type))
dataReal = data %>% filter(grepl("REAL_REAL", Type)) #%>% group_by(Language) %>% summarise(Surprisal = mean(Surprisal), Pars = mean(Pars))
dataGround = data %>% filter(grepl("ground", Type)) #%>% group_by(Language) %>% summarise(Surprisal = mean(Surprisal), Pars = mean(Pars))


D = dataGround %>% select(Language, Type, Model)






data = rbind(dataP, data2)
data = rbind(data, dataS)
data = rbind(data, dataRandom)
data = rbind(data, dataGround)
data = rbind(data, dataReal)






#corpusSize = read.csv("../corpus-size/corpus-sizes.tsv", sep="\t")
#languagesOrdered = corpusSize$language[order(-corpusSize$sents_train)]
#
#data$Language = factor(data$Language, levels=languagesOrdered)
#subData$Language = factor(subData$Language, levels=languagesOrdered)

   language = "English"

library(stringr)
sink("results-observed-and-mle.tex")
for(language in unique(data$Language)) {
   u = data %>% filter(Language == language)
   
   ground = (u %>% filter(Type == "manual_output_funchead_ground_coarse_final"))
   observed = (u %>% filter(Type == "REAL_REAL"))
   efficient = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))
   
   language2 = str_replace(language, "_", " ")
   language2 = str_replace(language2, "_", " ")

   cat(paste(language2, round(observed$UAS[1], 3), round(observed$LAS[1], 3), round(observed$ParsU[1], 3), round(observed$Pars[1], 3), round(observed$Surprisal[1],3), round(ground$UAS[1], 3), round(ground$LAS[1], 3), round(observed$ParsU[1], 3), round(observed$Pars[1], 3),  round(ground$Surprisal[1],3), "\\\\ \n", sep="  &  "))
}
sink()


