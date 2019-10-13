# per language


library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
#depl = read.csv("../../../grammars/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)

# Read predictability estimates
dataS =  read.csv("../../../grammars/plane/controls/plane-fixed-withoutPOS.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surp = mean(Surp, na.rm=TRUE))
dataS = as.data.frame(dataS)

# Read parseability estimates
dataP =  read.csv("../../../grammars/plane/controls/plane-parse-withoutPOS.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS, na.rm=TRUE), Pars = mean(Pars, na.rm=TRUE))
dataP = as.data.frame(dataP)

# Ensure everything is treated as a string, not a factor, to ensure the frames can be merged
dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)

# Collapse all five groups of random grammars (10 random grammars per group per language)
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM2", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM3", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM4", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM5", "manual_output_funchead_RANDOM", as.character(Type)))





dataComp = data
dataCompBaseline = dataComp %>% filter(grepl("RANDOM", Type))
dataCompGround = dataComp %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% select(Language, Surp, Pars) %>% rename(SurpGround = Surp) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurpGround) %>% group_by(Language)
dataComp = merge(dataCompBaseline, dataCompGround, by=c("Language"))

dataComp$Eff = dataComp$Pars + 0.9*dataComp$Surp

u = dataComp %>% group_by(Language) %>% summarise(BetterSurp = sum(Surp > SurpGround, na.rm=TRUE), WorseSurp = sum(Surp <= SurpGround, na.rm=TRUE), TotalSurp = BetterSurp+WorseSurp, BetterFracSurp = BetterSurp/TotalSurp, BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars, BetterEff = sum(Eff > EffGround, na.rm=TRUE), WorseEff = sum(Eff <= EffGround, na.rm=TRUE), TotalEff = BetterEff+WorseEff, BetterFracEff = BetterEff/TotalEff )










#data = merge(data, depl %>% select(Language, Model,AverageLengthPerWord) %>% mutate(Model = as.character(Model)), by=c("Language", "Model"), all.x=TRUE)
data = data %>% mutate(Two = 0.9*Surp+Pars)

data2 = rbind(data)
data2 = data2 %>% group_by(Language, Type) %>% summarise(Surp=mean(Surp, na.rm=TRUE), Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% mutate(MeanSurp = mean(Surp, na.rm=TRUE), SDSurp = sd(Surp, na.rm=TRUE)) %>% mutate(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE))
plot = ggplot(data2, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point()
#plot = ggplot(data, aes(x=(Pars), y=(Surp), color=Type, group=Type)) +geom_point()


#dataMean = data %>% group_by(Language) %>% summarise(MeanSurp = mean(Surp), SDSurp = sd(Surp)+0.0001)
#data_ = merge(data, dataMean, by=c("Language"))
#dataMean = data %>% group_by(Language) %>% summarise(MeanUAS = mean(UAS), SDUAS = sd(UAS)+0.0001)
#data_ = merge(data_, dataMean, by=c("Language"))
#
#plot = ggplot(data_, aes(x=(UAS), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point()

dataPBest = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% group_by(Language) %>% summarise(Pars = min(Pars, na.rm=TRUE))
data2Best = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% group_by(Language) %>% summarise(Two = min(Two, na.rm=TRUE))
dataSBest = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% group_by(Language) %>% summarise(Surp = min(Surp, na.rm=TRUE))

dataP = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced")
data2 = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large")
dataS = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced")

dataP = merge(dataP, dataPBest, by=c("Language", "Pars"))
data2 = merge(data2, data2Best, by=c("Language", "Two"))
dataS = merge(dataS, dataSBest, by=c("Language", "Surp"))


dataRandom = data %>% filter(grepl("RANDOM", Type))
#dataDepL = data %>% filter(grepl("depl", Type)) #%>% filter(Model %in% deplBest) ###%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))
dataGround = data %>% filter(grepl("ground", Type)) #%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))
#dataReal = data %>% filter(grepl("REAL", Type)) #%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))


D = dataGround %>% select(Language, Type, Model)
#write.csv(D, file="../strongest_models/models-mle.csv")




#dataDepLBest = dataDepL %>% group_by(Language) %>% summarise(AverageLengthPerWord = min(AverageLengthPerWord))
#dataDepL = merge(dataDepL, dataDepLBest, by=c("Language", "AverageLengthPerWord"))

#D = dataDepL %>% select(Language, Type, Model, AverageLengthPerWord)
#write.csv(D, file="../strongest_models/best-depl.csv")




data = rbind(dataP, data2)
data = rbind(data, dataS)
data = rbind(data, dataRandom)
#data = rbind(data, dataDepL)
data = rbind(data, dataGround)
#data = rbind(data, dataReal)


#data2 = data2 %>% %>% group_by(Language) %>% mutate(MeanSurp = mean(Surp, na.rm=TRUE), SDSurp = sd(Surp, na.rm=TRUE)) %>% mutate(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE))

dataMean = data %>% group_by(Language, Type) %>% summarise(Surp=mean(Surp, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanSurp = mean(Surp, na.rm=TRUE), SDSurp = sd(Surp, na.rm=TRUE)+0.0001)
#write.csv(dataMean, file="surp-z.csv")
data = merge(data, dataMean, by=c("Language"))
#dataDepL = merge(dataDepL, dataMean, by=c("Language"))
dataMean = data %>% group_by(Language, Type) %>% summarise(Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE)+0.0001)
#write.csv(dataMean, file="pars-z.csv")
data = merge(data, dataMean, by=c("Language"))
#dataDepL = merge(dataDepL, dataMean, by=c("Language"))

dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanSurpRand = mean(Surp, na.rm=TRUE), SDSurpRand = sd(Surp, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
#dataDepL = merge(dataDepL, dataMean, by=c("Language"))
dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanParsRand = mean(Pars, na.rm=TRUE), SDParsRand = sd(Pars, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
#dataDepL = merge(dataDepL, dataMean, by=c("Language"))



D = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% select(Language, Type, Model, Pars, Surp)
#write.csv(D, file="best-two-lambda09-best-balanced.csv")


D = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% select(Language, Type, Model, Surp)
#write.csv(D, file="../strongest_models/best-langmod-best-balanced.csv")


D = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% select(Language, Type, Model, Pars)
#write.csv(D, file="best-parse-best-balanced.csv")






plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point()
plot = ggplot(data, aes(x=(Pars-MeanPars), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point()

plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point() + facet_wrap(~Language, scales="free")

plot = ggplot(data, aes(x=(Pars-MeanPars), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point() + facet_wrap(~Language)


data = data %>% mutate(Pars_z = (Pars-MeanPars)/SDPars, Surp_z=(Surp-MeanSurp)/SDSurp)

subData = data %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced"))

subData = subData[order(subData$Type),]

Step = c()
Surp_z = c()
Pars_z = c()
Language = c()
for(language in unique(subData$Language)) {
   u = subData %>% filter(Language == language)

   # Pareto hull

   # Find the maximum predictability end, with the associated parseability
   pred = min(u$Surp_z)
   pars = (u %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced"))$Pars_z[1]

   Surp_z = c(Surp_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 1)

   # Find the maximum efficiency end
   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Surp_z[1]
   pars = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Pars_z[1]

   Surp_z = c(Surp_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 2)

   # Find the maximum parseability end, with the associated predictability
   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced"))$Surp_z[1]
   pars = min(u$Pars_z)

   Surp_z = c(Surp_z, pred)
   Pars_z = c(Pars_z, pars)
   Language = c(Language, language)
   Step = c(Step, 3)

}

subData = data.frame(Language=Language, Surp_z=Surp_z, Pars_z=Pars_z)
subData$Type = "Pareto"



corpusSize = read.csv("../../corpus-size/corpus-sizes.tsv", sep="\t")
languagesOrdered = corpusSize$language[order(-corpusSize$sents_train)]

data$Language = factor(data$Language, levels=languagesOrdered)
subData$Language = factor(subData$Language, levels=languagesOrdered)



plot = ggplot(data %>% filter(Type %in% c("manual_output_funchead_RANDOM")) %>% filter(Surp_z < 3), aes(x=-Pars_z, y=-Surp_z, color=Type, group=Type))
plot = plot + geom_point()
plot = plot + geom_path(data=subData, aes(x=-Pars_z, y=-Surp_z, group=1), size=1.5)
plot = plot + geom_point(data=data %>% filter(Type == "manual_output_funchead_ground_coarse_final"), shape=4, size=1.5, stroke=2)
plot = plot + facet_wrap(~Language, scales="free")
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
ggsave(plot, file="pareto-plane-perLanguage-lexicalized-mle.pdf", width=12, height=12)





