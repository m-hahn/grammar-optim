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


dataComp = data
dataCompBaseline = dataComp %>% filter(grepl("RANDOM", Type))
dataCompGround = dataComp %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% select(Language, Surprisal, Pars) %>% rename(SurprisalGround = Surprisal) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurprisalGround) %>% group_by(Language)
dataComp = merge(dataCompBaseline, dataCompGround, by=c("Language"))

dataComp$Eff = dataComp$Pars + 0.9*dataComp$Surprisal

u = dataComp %>% group_by(Language) %>% summarise(BetterSurprisal = sum(Surprisal > SurprisalGround, na.rm=TRUE), WorseSurprisal = sum(Surprisal <= SurprisalGround, na.rm=TRUE), TotalSurprisal = BetterSurprisal+WorseSurprisal, BetterFracSurprisal = BetterSurprisal/TotalSurprisal, BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars, BetterEff = sum(Eff > EffGround, na.rm=TRUE), WorseEff = sum(Eff <= EffGround, na.rm=TRUE), TotalEff = BetterEff+WorseEff, BetterFracEff = BetterEff/TotalEff )










data = data %>% mutate(Two = 0.9*Surprisal+Pars)

data2 = rbind(data)
data2 = data2 %>% group_by(Language, Type) %>% summarise(Surprisal=mean(Surprisal, na.rm=TRUE), Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% mutate(MeanSurprisal = mean(Surprisal, na.rm=TRUE), SDSurprisal = sd(Surprisal, na.rm=TRUE)) %>% mutate(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE))
plot = ggplot(data2, aes(x=(Pars-MeanPars)/SDPars, y=(Surprisal-MeanSurprisal)/SDSurprisal, color=Type, group=Type)) +geom_point()

dataPBest = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% group_by(Language) %>% summarise(Pars = min(Pars, na.rm=TRUE))
data2Best = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% group_by(Language) %>% summarise(Two = min(Two, na.rm=TRUE))
dataSBest = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% group_by(Language) %>% summarise(Surprisal = min(Surprisal, na.rm=TRUE))

dataP = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced")
data2 = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large")
dataS = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced")

dataP = merge(dataP, dataPBest, by=c("Language", "Pars"))
data2 = merge(data2, data2Best, by=c("Language", "Two"))
dataS = merge(dataS, dataSBest, by=c("Language", "Surprisal"))


dataRandom = data %>% filter(grepl("RANDOM", Type))
dataGround = data %>% filter(grepl("ground", Type)) 
dataReal = data %>% filter(grepl("REAL_REAL", Type)) 


D = dataGround %>% select(Language, Type, Model)



data = rbind(dataP, data2)
data = rbind(data, dataS)
data = rbind(data, dataRandom)
data = rbind(data, dataGround)
data = rbind(data, dataReal)



dataMean = data %>% group_by(Language, Type) %>% summarise(Surprisal=mean(Surprisal, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanSurprisal = mean(Surprisal, na.rm=TRUE), SDSurprisal = sd(Surprisal, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataMean = data %>% group_by(Language, Type) %>% summarise(Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))

dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanSurprisalRand = mean(Surprisal, na.rm=TRUE), SDSurprisalRand = sd(Surprisal, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanParsRand = mean(Pars, na.rm=TRUE), SDParsRand = sd(Pars, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))



D = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% select(Language, Type, Model, Pars, Surprisal)
D = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% select(Language, Type, Model, Surprisal)
D = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% select(Language, Type, Model, Pars)





data = data %>% mutate(Pars_z = (Pars-MeanPars)/SDPars, Surprisal_z=(Surprisal-MeanSurprisal)/SDSurprisal)

subData = data %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced"))


#######################################

subData = subData[order(subData$Type),]

Step = c()
Surprisal_z = c()
Pars_z = c()
Language = c()
for(language in unique(subData$Language)) {
   u = subData %>% filter(Language == language)

   # Pareto hull

   # Find the maximum predictability end, with the associated parseability
   pred = min(u$Surprisal_z)
   pars = (u %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced"))$Pars_z[1]

   Surprisal_z = c(Surprisal_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 1)

   # Find the maximum efficiency end
   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Surprisal_z[1]
   pars = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Pars_z[1]

   Surprisal_z = c(Surprisal_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 2)

   # Find the maximum parseability end, with the associated predictability
   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced"))$Surprisal_z[1]
   pars = min(u$Pars_z)

   Surprisal_z = c(Surprisal_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 3)

}

subData = data.frame(Language=Language, Surprisal_z=Surprisal_z, Pars_z=Pars_z, Step=Step)
subData$Type = "Pareto"

#######################################


corpusSize = read.csv("../../corpus-size/corpus-sizes.tsv", sep="\t")
languagesOrdered = corpusSize$language[order(-corpusSize$sents_train)]

data$Language = factor(data$Language, levels=languagesOrdered)
subData$Language = factor(subData$Language, levels=languagesOrdered)



data$Group = "det"
data[data$Type==  "REAL_REAL",]$Group = "nondet"
data[data$Type == "manual_output_funchead_RANDOM_nondet",]$Group = "nondet"
data[data$Type == "manual_output_funchead_RANDOM_det",]$Group = "det"

subData$Group = "Pareto"


data = merge(data, corpusSize %>% rename(Language=language), by=c("Language"))
subData = merge(subData, corpusSize %>% rename(Language=language), by=c("Language"))

data$Language_ = paste(data$Language, "\nn = ", data$sents_train, sep="")
subData$Language_ = paste(subData$Language, "\nn = ", subData$sents_train, sep="")
corpusSize$Language_ = paste(corpusSize$language, "\nn = ", corpusSize$sents_train, sep="")


languagesOrdered = corpusSize$Language[order(-corpusSize$sents_train)]

data$Language_ = factor(data$Language_, levels=languagesOrdered)
subData$Language_ = factor(subData$Language_, levels=languagesOrdered)



plot = ggplot(data %>% filter(grepl("RANDOM", Type)) %>% filter(abs(Surprisal_z) < 3), aes(x=-Pars_z, y=-Surprisal_z, color=Group, group=Type))
plot = plot + geom_point(data = data  %>% filter(grepl("RANDOM_det", Type)) %>% filter(abs(Surprisal_z) < 3), size=0.5)
plot = plot + geom_point(data = data  %>% filter(grepl("RANDOM_nondet", Type)) %>% filter(abs(Surprisal_z) < 3), size=1.0)
plot = plot + geom_path(data=subData, aes(x=-Pars_z, y=-Surprisal_z, group=1), size=1.5)
plot = plot + geom_point(data=data %>% filter(Type == "manual_output_funchead_ground_coarse_final"), shape=4, size=1.5, stroke=2)
plot = plot + geom_point(data=data %>% filter(Type == "REAL_REAL"), shape=2, size=1.5, stroke=2)
plot = plot + facet_wrap(~Language_, scales="free")
plot = plot + theme_bw()
plot = plot + scale_x_continuous(name="Parseability") + scale_y_continuous(name="Predictability")
plot = plot + theme(legend.title = element_blank())  
plot = plot + guides(color=guide_legend(nrow=2,ncol=4,byrow=TRUE)) 
plot = plot + theme(legend.title = element_blank(), legend.position="bottom")
plot = plot + theme(axis.title.x = element_text(size=17))
plot = plot + theme(axis.title.y = element_text(size=17))
plot = plot + theme(legend.text = element_text(size=12))
plot = plot + theme(legend.margin=margin(t = 0, unit='cm'))
plot = plot + theme(legend.position = "none")
ggsave(plot, file="pareto-plane-perLanguage-nondeterministic-mle-allLanguages.pdf", width=12, height=12)





