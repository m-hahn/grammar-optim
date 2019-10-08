

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
dataP = read.csv("../../grammars/plane/plane-parse-redone.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surprisal = mean(Surp, na.rm=TRUE))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(Pars = mean(Pars, na.rm=TRUE))
dataS = as.data.frame(dataS)
dataP = as.data.frame(dataP)
library(lme4)
summary(lmer(Surprisal ~ Type + (1|Language), data=dataS %>% filter(grepl("langm", Type))))
summary(lmer(Surprisal ~ Type + (1|Language), data=dataS %>% filter(grepl("RL", Type) | grepl("ground", Type))))
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
dataGround = data %>% filter(grepl("ground", Type)) #%>% group_by(Language) %>% summarise(Surprisal = mean(Surprisal), Pars = mean(Pars))


D = dataGround %>% select(Language, Type, Model)



data = rbind(dataP, data2)
data = rbind(data, dataS)
data = rbind(data, dataRandom)
data = rbind(data, dataGround)



dataMean = data %>% group_by(Language, Type) %>% summarise(Surprisal=mean(Surprisal, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanSurprisal = mean(Surprisal, na.rm=TRUE), SDSurprisal = sd(Surprisal, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataMean = data %>% group_by(Language, Type) %>% summarise(Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))

dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanSurprisalRand = mean(Surprisal, na.rm=TRUE), SDSurprisalRand = sd(Surprisal, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanParsRand = mean(Pars, na.rm=TRUE), SDParsRand = sd(Pars, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))



D = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% select(Language, Type, Model, Pars, Surprisal)
#write.csv(D, file="best-two-lambda09-best-balanced.csv")

D = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% select(Language, Type, Model, Surprisal)
#write.csv(D, file="../strongest_models/best-langmod-best-balanced.csv")

D = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% select(Language, Type, Model, Pars)
#write.csv(D, file="best-parse-best-balanced.csv")





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

   pred = min(u$Surprisal_z)
   pars = (u %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced"))$Pars_z[1]

   Surprisal_z = c(Surprisal_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 1)

   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Surprisal_z[1]
   pars = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Pars_z[1]

   Surprisal_z = c(Surprisal_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 2)

   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced"))$Surprisal_z[1]
   pars = min(u$Pars_z)

   Surprisal_z = c(Surprisal_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 3)

}

subData = data.frame(Language=Language, Surprisal_z=Surprisal_z, Pars_z=Pars_z, Step=Step)
subData$Type = "Pareto"



summarizedData = data %>% group_by(Type) %>% summarise(Surprisal = mean((Surprisal-MeanSurprisal)/SDSurprisal, na.rm=TRUE), Pars = mean((Pars-MeanPars)/SDPars, na.rm=TRUE))
summarizedDataRand = data %>% group_by(Type) %>% summarise(Surprisal = mean((Surprisal-MeanSurprisalRand)/SDSurprisalRand, na.rm=TRUE), Pars = mean((Pars-MeanParsRand)/SDParsRand, na.rm=TRUE))

iso = read.csv("../languages/languages-iso_codes.tsv")

data = merge(data, iso, by=c("Language"))



library(forcats)
data$Type = as.factor(data$Type)
data = data %>% mutate(TypeN = fct_recode(Type, "Parseability" = "manual_output_funchead_two_coarse_parser_best_balanced", "Baseline Grammars" = "manual_output_funchead_RANDOM", "Efficiency" = "manual_output_funchead_two_coarse_lambda09_best_large", "Dependency Length" = "manual_output_funchead_coarse_depl", "Predictability" = "manual_output_funchead_langmod_coarse_best_balanced", "Real Grammars" = "manual_output_funchead_ground_coarse_final"))
summarizedData = summarizedData %>% mutate(TypeN = fct_recode(Type, "Parseability" = "manual_output_funchead_two_coarse_parser_best_balanced", "Baseline Grammars" = "manual_output_funchead_RANDOM", "Efficiency" = "manual_output_funchead_two_coarse_lambda09_best_large", "Dependency Length" = "manual_output_funchead_coarse_depl", "Predictability" = "manual_output_funchead_langmod_coarse_best_balanced", "Real Grammars" = "manual_output_funchead_ground_coarse_final"))

dataPlot = rbind(data %>% mutate(Pars =(Pars-MeanPars)/SDPars, Surprisal = (Surprisal-MeanSurprisal)/SDSurprisal) %>% select(Pars, Surprisal, TypeN, iso_code) %>% mutate(ParsMean=NA, SurprisalMean=NA), summarizedData %>% select(Pars, Surprisal, TypeN) %>% mutate(iso_code=NA) %>% rename(ParsMean=Pars,SurprisalMean=Surprisal)%>%mutate(Surprisal=NA,Pars=NA))
dataPlot = dataPlot %>% mutate(Pars_z = NA, Surprisal_z = NA, Pars_z_end = NA, Surprisal_z_end = NA)


dataPlot = dataPlot %>% mutate(TypeN = factor(TypeN, levels=c("Parseability", "Predictability", "Efficiency", "Dependency Length", "Real Grammars", "Baseline Grammars"), ordered=TRUE))
forColoring  = factor(dataPlot$TypeN, levels=c("Real Grammars", "Baseline Grammars", "Parseability", "Predictability", "Efficiency", "Dependency Length"), ordered=TRUE)
#Create a custom color scale
library(RColorBrewer)
myColors <- brewer.pal(6,"Set1")
names(myColors) <- levels(forColoring)




library(ggrepel)




subData2 = subData %>% mutate(TypeN=Type) %>% group_by(Step) %>% summarise(Surprisal = mean(Surprisal_z), Pars = mean(Pars_z), Surprisal_SD = sd(Surprisal_z), Pars_SD = sd(Pars_z))




plot = ggplot(dataPlot)  
plot = plot + theme_bw() 
plot = plot + geom_density_2d(data=dataPlot %>% filter(grepl("Baseline", TypeN)), aes(x=-Pars, y=-Surprisal, color=TypeN, group=TypeN), size=0.3, bins=5)
plot = plot + geom_path(data=subData2, aes(x=-Pars, y=-Surprisal, group=1), color="gray", size=5)

plot = plot + geom_path(data=subData2, aes(x=-Pars+Pars_SD, y=-Surprisal+Surprisal_SD, group=1), color="gray", size=2, linetype="dashed")
plot = plot + geom_path(data=subData2, aes(x=-Pars-Pars_SD, y=-Surprisal-Surprisal_SD, group=1), color="gray", size=2, linetype="dashed")

plot = plot + scale_x_continuous(name="Parseability") + scale_y_continuous(name="Predictability")
plot = plot + geom_text(data=dataPlot %>% filter(grepl("Real", TypeN)), aes(x=-Pars, y=-Surprisal, color=TypeN, group=TypeN, label=iso_code),hjust=0.8, vjust=0.8, size=4.5)
plot = plot + theme(legend.title = element_blank())  
plot = plot + guides(color=guide_legend(nrow=2,ncol=4,byrow=TRUE)) 
plot = plot + theme(legend.title = element_blank(), legend.position="bottom")
plot = plot + scale_colour_manual(name = "TypeN",values = myColors)
plot = plot + theme(axis.title.x = element_text(size=17))
plot = plot + theme(axis.title.y = element_text(size=17))
plot = plot + theme(legend.text = element_text(size=12))
plot = plot + theme(legend.margin=margin(t = 0, unit='cm'))
ggsave(plot, file="pareto-plane-iso-best-balanced-legend-viz-10-fontsize_pareto_SD.pdf", width=6.6, height=6.6)





