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
dataS4 = read.csv("~/CS_SCR/deps/plane-fixed-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS = rbind(dataS, dataS2, dataS3, dataS4)
dataP = read.csv("../../grammars/plane/plane-parse.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP2 = read.csv("../../grammars/plane/plane-parse-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP3 = read.csv("../../grammars/plane/plane-parse-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP4 = read.csv("~/CS_SCR/deps/plane-parse-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = rbind(dataP, dataP2, dataP3, dataP4)
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surp = mean(Surp, na.rm=TRUE))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS, na.rm=TRUE), Pars = mean(Pars, na.rm=TRUE))
dataS = as.data.frame(dataS)
dataP = as.data.frame(dataP)
library(lme4)
summary(lmer(Surp ~ Type + (1|Language), data=dataS %>% filter(grepl("langm", Type))))
summary(lmer(Surp ~ Type + (1|Language), data=dataS %>% filter(grepl("RL", Type) | grepl("ground", Type))))
dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)


data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM2", "manual_output_funchead_RANDOM", as.character(Type)))


dataBaseline = data %>% filter(Type == "manual_output_funchead_RANDOM")
dataGround = data %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% select(Language, Surp, Pars) %>% rename(SurpGround = Surp) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurpGround) %>% group_by(Language)
data = merge(dataBaseline, dataGround, by=c("Language"))

data$Eff = data$Pars + 0.9*data$Surp

u = data %>% group_by(Language) %>% summarise(BetterSurp = sum(Surp > SurpGround, na.rm=TRUE), WorseSurp = sum(Surp <= SurpGround, na.rm=TRUE), TotalSurp = BetterSurp+WorseSurp, BetterFracSurp = BetterSurp/TotalSurp, BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars, BetterEff = sum(Eff > EffGround, na.rm=TRUE), WorseEff = sum(Eff <= EffGround, na.rm=TRUE), TotalEff = BetterEff+WorseEff, BetterFracEff = BetterEff/TotalEff )



pValuesSurp_bin = c()
pValuesSurp_t = c()
for(i in (1:51)) {
	pValuesSurp_bin = c(pValuesSurp_bin, binom.test(u$BetterSurp[[i]], u$TotalSurp[[i]])$p.value)

	language = u$Language[[i]]
v = data %>% filter(Language == language)
	pValuesSurp_t = c(pValuesSurp_t, t.test(v$Surp, mu=mean(v$SurpGround), alternative="greater")$p.value)

}
u$pValuesSurp_bin = pValuesSurp_bin
u$pValuesSurp_t = pValuesSurp_t


pValuesEff_bin = c()
pValuesEff_t = c()
for(i in (1:51)) {
	if(u$TotalEff[[i]] == 0) {
	pValuesEff_bin = c(pValuesEff_bin, 0.5)

	} else {
	pValuesEff_bin = c(pValuesEff_bin, binom.test(u$BetterEff[[i]], u$TotalEff[[i]])$p.value)
	}

	language = u$Language[[i]]
v = data %>% filter(Language == language)
if(is.na(mean(v$EffGround))) {
	pValuesEff_t = c(pValuesEff_t, 0.5)

} else {
	pValuesEff_t = c(pValuesEff_t, t.test(v$Eff, mu=mean(v$EffGround), alternative="greater")$p.value)
}

}
u$pValuesEff_bin = pValuesEff_bin
u$pValuesEff_t = pValuesEff_t




#u = u %>% summarise(BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars)
pValuesPars_bin = c()
pValuesPars_t = c()
for(i in (1:51)) {
	if(u$TotalPars[[i]] == 0) {
	pValuesPars_bin = c(pValuesPars_bin, 0.5)

	} else {
	pValuesPars_bin = c(pValuesPars_bin, binom.test(u$BetterPars[[i]], u$TotalPars[[i]])$p.value)
	}

	language = u$Language[[i]]
v = data %>% filter(Language == language)
if(is.na(mean(v$ParsGround))) {
	pValuesPars_t = c(pValuesPars_t, 0.5)

} else {
	pValuesPars_t = c(pValuesPars_t, t.test(v$Pars, mu=mean(v$ParsGround), alternative="greater")$p.value)
}

}
u$pValuesPars_bin = pValuesPars_bin
u$pValuesPars_t = pValuesPars_t



mean(u$pValuesPars_t < 0.05)
mean(u$pValuesSurp_t < 0.05)
mean(u$pValuesPars_t < 0.025 | u$pValuesSurp_t < 0.025)
mean(u$pValuesEff_t < 0.05)


mean(u$pValuesPars_bin < 0.05)
mean(u$pValuesSurp_bin < 0.05)
mean(u$pValuesPars_bin < 0.025 | u$pValuesSurp_bin < 0.025)
mean(u$pValuesEff_bin < 0.05)


