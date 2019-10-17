

library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(dplyr)
dataS = read.csv("../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS3 = read.csv("../../grammars/plane/plane-fixed-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS4 = read.csv("../../grammars/plane/plane-fixed-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS5 = read.csv("../../grammars/plane/plane-fixed-random3.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS6 = read.csv("../../grammars/plane/plane-fixed-random4.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS7 = read.csv("../../grammars/plane/plane-fixed-random5.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS = rbind(dataS, dataS2, dataS3, dataS4, dataS5, dataS6, dataS7)
dataP = read.csv("../../grammars/plane/plane-parse-unified.tsv", sep="\t") %>% mutate(Model = as.character(Model))
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
#write.csv(D, file="../strongest_models/models-mle.csv")



data = rbind(dataP, data2)
data = rbind(data, dataS)
data = rbind(data, dataRandom)
data = rbind(data, dataGround)



dataMean = data %>% group_by(Language, Type) %>% summarise(Surprisal=mean(Surprisal, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanSurprisal = mean(Surprisal, na.rm=TRUE), SDSurprisal = sd(Surprisal, na.rm=TRUE)+0.0001)
write.csv(dataMean, file="surp-z.csv")
data = merge(data, dataMean, by=c("Language"))
dataMean = data %>% group_by(Language, Type) %>% summarise(Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE)+0.0001)
write.csv(dataMean, file="pars-z.csv")
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
Surprisal = c()
Pars = c()
Language = c()
for(language in unique(subData$Language)) {
   u = subData %>% filter(Language == language)

   # Pareto hull

   pred = min(u$Surprisal)
   pars = (u %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced"))$Pars[1]

   Surprisal = c(Surprisal, pred)
   Pars = c(Pars, pars)
   Language = c(Language, language)
   Step = c(Step, 1)

   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Surprisal[1]
   pars = (u %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large"))$Pars[1]

   Surprisal = c(Surprisal, pred)
   Pars = c(Pars, pars)
   Language = c(Language, language)
   Step = c(Step, 2)

   pred = (u %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced"))$Surprisal[1]
   pars = min(u$Pars)

   Surprisal = c(Surprisal, pred)
   Pars = c(Pars, pars)
   Language = c(Language, language)
   Step = c(Step, 3)

}

subData = data.frame(Language=Language, Surprisal=Surprisal, Pars=Pars, Step=Step)
subData$Type = "Pareto"




corpusProperties = read.csv("../../models/revision/corpus-properties/corpusProperties.tsv", sep="\t")


subData = merge(corpusProperties, subData, by=c("Language"))

subData1=subData %>% filter(Step==1)
subData2=subData %>% filter(Step==2)
subData3=subData %>% filter(Step==3)

surp__ = subData %>% mutate(Pars=NULL) %>% spread(Step, Surprisal)
pars__ = subData %>% mutate(Surprisal=NULL) %>% spread(Step, Pars)



cor.test(surp__[["2"]] - surp__[["3"]], surp__$MedianSentenceLength)
cor.test(surp__[["2"]] - surp__[["3"]], surp__$MedianTreeDepth)
cor.test(surp__[["1"]], log(surp__$SentenceCount))


# Predictors:
#"SentenceCount"
#  "MedianArity"          "MedianTreeDepth"      
# "MedianSentenceLength" 

cor.test(subData1$UnigramEntropy, subData1$Surprisal)



cor(subData3$MedianSentenceLength, subData3$Surprisal)
cor(subData1$MedianSentenceLength, subData1$Surprisal)
cor(subData2$MedianSentenceLength, subData2$Surprisal)

cor.test(subData3$MedianSentenceLength, subData3$Pars)



# Properties of the efficiency-optimized ends
modelS1 = (lm(-Surprisal ~ MedianSentenceLength  + MedianTreeDepth+MeanArity + UnigramEntropy+ log(SentenceCount)  , data=subData2))
modelP1 = (lm(-Pars ~ MedianSentenceLength + MedianTreeDepth +MeanArity + UnigramEntropy+ log(SentenceCount) , data=subData2))

# 
surp__$A2Minus3 = (surp__[["2"]] - surp__[["3"]])
surp__$A2Minus1 = surp__[["2"]] - surp__[["1"]]

modelS3 = (lm(A2Minus3 ~ MedianSentenceLength  + MedianTreeDepth+MeanArity + UnigramEntropy+ log(SentenceCount) , data=surp__)) # parseability-optimized end
modelS2 = (lm(A2Minus1 ~ MedianSentenceLength  + MedianTreeDepth+MeanArity + UnigramEntropy+ log(SentenceCount) , data=surp__)) # predictability-optimized end


pars__$A2Minus3 = (pars__[["2"]] - pars__[["3"]])
pars__$A2Minus1 = pars__[["2"]] - pars__[["1"]]

modelP3 = (lm(A2Minus3 ~ MedianSentenceLength  + MedianTreeDepth+MeanArity + UnigramEntropy+ log(SentenceCount) , data=pars__))
modelP2 = (lm(A2Minus1 ~ MedianSentenceLength  + MedianTreeDepth+MeanArity + UnigramEntropy+ log(SentenceCount) , data=pars__))


#

sig = function(x) {
  if(x > 0.05) {
	  return("")
  } else if(x > 0.01) {
	  return("$^*$")
  } else if(x > 0.001) {
	  return("$^{**}$")
  } else {
	  return("$^{***}$")
  }
}


coefs1 = summary(modelS1)$coefficients
coefs2 = summary(modelS2)$coefficients
coefs3 = summary(modelS3)$coefficients


createTable = function(coefs1, coefs2, coefs3) {
   for(i in (1:6)) {
	   if(i > 1) {
	   cat("(", i-1, ") ", sep="")
	   }
      cat(rownames(coefs1)[i], " & ")
      for(coefs in list(coefs1, coefs2, coefs3)) {
      cat(round(coefs[i,1],2), " & ", sep="")
      cat(round(coefs[i,2],2), " & ", sep="")
      cat(round(coefs[i,3],2), sig(coefs[i,4]), " & ", sep="")
      }
      if(i < 6) {
      cat("\\\\\n")
      }
   }
cat("\\\\ \\hline \\hline\n")
}

sink("corpus-frontier-regressions.tex")

createTable(coefs1, coefs2, coefs3)

coefs1 = summary(modelP1)$coefficients
coefs2 = summary(modelP2)$coefficients
coefs3 = summary(modelP3)$coefficients

createTable(coefs1, coefs2, coefs3)
sink()

