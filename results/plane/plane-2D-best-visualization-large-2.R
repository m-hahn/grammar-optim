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




dataComp = data
dataCompBaseline = dataComp %>% filter(grepl("RANDOM", Type))
dataCompGround = dataComp %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% select(Language, Surp, Pars) %>% rename(SurpGround = Surp) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurpGround) %>% group_by(Language)
dataComp = merge(dataCompBaseline, dataCompGround, by=c("Language"))

dataComp$Eff = dataComp$Pars + 0.9*dataComp$Surp

u = dataComp %>% group_by(Language) %>% summarise(BetterSurp = sum(Surp > SurpGround, na.rm=TRUE), WorseSurp = sum(Surp <= SurpGround, na.rm=TRUE), TotalSurp = BetterSurp+WorseSurp, BetterFracSurp = BetterSurp/TotalSurp, BetterPars = sum(Pars > ParsGround, na.rm=TRUE), WorsePars = sum(Pars <= ParsGround, na.rm=TRUE), TotalPars = BetterPars+WorsePars, BetterFracPars = BetterPars/TotalPars, BetterEff = sum(Eff > EffGround, na.rm=TRUE), WorseEff = sum(Eff <= EffGround, na.rm=TRUE), TotalEff = BetterEff+WorseEff, BetterFracEff = BetterEff/TotalEff )



pValuesSurp_bin = c()
pValuesSurp_t = c()
for(i in (1:51)) {
	pValuesSurp_bin = c(pValuesSurp_bin, binom.test(u$BetterSurp[[i]], u$TotalSurp[[i]])$p.value)

	language = u$Language[[i]]
v = dataComp %>% filter(Language == language)
	pValuesSurp_t = c(pValuesSurp_t, t.test(v$Surp, mu=mean(v$SurpGround), alternative="greater")$p.value)

}
#u$pValuesSurp_bin = pValuesSurp_bin
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
v = dataComp %>% filter(Language == language)
if(is.na(mean(v$EffGround))) {
	pValuesEff_t = c(pValuesEff_t, 0.5)

} else {
	pValuesEff_t = c(pValuesEff_t, t.test(v$Eff, mu=mean(v$EffGround), alternative="greater")$p.value)
}

}
#u$pValuesEff_bin = pValuesEff_bin
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
v = dataComp %>% filter(Language == language)
if(is.na(mean(v$ParsGround))) {
	pValuesPars_t = c(pValuesPars_t, 0.5)

} else {
	pValuesPars_t = c(pValuesPars_t, t.test(v$Pars, mu=mean(v$ParsGround), alternative="greater")$p.value)
}

}
#u$pValuesPars_bin = pValuesPars_bin
u$pValuesPars_t = pValuesPars_t



mean(u$pValuesPars_t < 0.05)
mean(u$pValuesSurp_t < 0.05)
mean(u$pValuesPars_t < 0.025 | u$pValuesSurp_t < 0.025)
mean(u$pValuesEff_t < 0.05)










data = merge(data, depl %>% select(Language, Model,AverageLengthPerWord) %>% mutate(Model = as.character(Model)), by=c("Language", "Model"), all.x=TRUE)
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

dataPBest = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% group_by(Language) %>% summarise(Pars = min(Pars))
data2Best = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% group_by(Language) %>% summarise(Two = min(Two, na.rm=TRUE))
dataSBest = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% group_by(Language) %>% summarise(Surp = min(Surp, na.rm=TRUE))

dataP = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced")
data2 = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large")
dataS = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced")

dataP = merge(dataP, dataPBest, by=c("Language", "Pars"))
data2 = merge(data2, data2Best, by=c("Language", "Two"))
dataS = merge(dataS, dataSBest, by=c("Language", "Surp"))


dataRandom = data %>% filter(grepl("RANDOM", Type))
dataDepL = data %>% filter(grepl("depl", Type)) #%>% filter(Model %in% deplBest) ###%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))
dataGround = data %>% filter(grepl("ground", Type)) #%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))
#dataReal = data %>% filter(grepl("REAL", Type)) #%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))


D = dataGround %>% select(Language, Type, Model)
#write.csv(D, file="../strongest_models/models-mle.csv")




dataDepLBest = dataDepL %>% group_by(Language) %>% summarise(AverageLengthPerWord = min(AverageLengthPerWord))
dataDepL = merge(dataDepL, dataDepLBest, by=c("Language", "AverageLengthPerWord"))

D = dataDepL %>% select(Language, Type, Model, AverageLengthPerWord)
#write.csv(D, file="../strongest_models/best-depl.csv")




data = rbind(dataP, data2)
data = rbind(data, dataS)
data = rbind(data, dataRandom)
data = rbind(data, dataDepL)
data = rbind(data, dataGround)
#data = rbind(data, dataReal)


#data2 = data2 %>% %>% group_by(Language) %>% mutate(MeanSurp = mean(Surp, na.rm=TRUE), SDSurp = sd(Surp, na.rm=TRUE)) %>% mutate(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE))

dataMean = data %>% group_by(Language, Type) %>% summarise(Surp=mean(Surp, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanSurp = mean(Surp, na.rm=TRUE), SDSurp = sd(Surp, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataDepL = merge(dataDepL, dataMean, by=c("Language"))
dataMean = data %>% group_by(Language, Type) %>% summarise(Pars=mean(Pars, na.rm=TRUE)) %>% group_by(Language) %>% summarise(MeanPars = mean(Pars, na.rm=TRUE), SDPars = sd(Pars, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataDepL = merge(dataDepL, dataMean, by=c("Language"))

dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanSurpRand = mean(Surp, na.rm=TRUE), SDSurpRand = sd(Surp, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataDepL = merge(dataDepL, dataMean, by=c("Language"))
dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanParsRand = mean(Pars, na.rm=TRUE), SDParsRand = sd(Pars, na.rm=TRUE)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataDepL = merge(dataDepL, dataMean, by=c("Language"))



D = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% select(Language, Type, Model, Pars, Surp)
#write.csv(D, file="best-two-lambda09-best-balanced.csv")


D = data %>% filter(Type == "manual_output_funchead_langmod_coarse_best_balanced") %>% select(Language, Type, Model, Surp)
#write.csv(D, file="../strongest_models/best-langmod-best-balanced.csv")


D = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_best_balanced") %>% select(Language, Type, Model, Pars)
#write.csv(D, file="best-parse-best-balanced.csv")



plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point()
plot = ggplot(data, aes(x=(Pars-MeanPars), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point()

plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point() + facet_wrap(~Language)

plot = ggplot(data, aes(x=(Pars-MeanPars), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point() + facet_wrap(~Language)


subData = data %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced"))

subData = subData[order(subData$Type),]

plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point() + geom_path(data=subData, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, group=1)) + facet_wrap(~Language)


summarizedData = data %>% group_by(Type) %>% summarise(Surp = mean((Surp-MeanSurp)/SDSurp, na.rm=TRUE), Pars = mean((Pars-MeanPars)/SDPars, na.rm=TRUE))
plot = ggplot(summarizedData, aes(x=Pars, y=Surp, color=Type, group=Type)) + geom_point()


summarizedDataRand = data %>% group_by(Type) %>% summarise(Surp = mean((Surp-MeanSurpRand)/SDSurpRand, na.rm=TRUE), Pars = mean((Pars-MeanParsRand)/SDParsRand, na.rm=TRUE))
plot = ggplot(summarizedDataRand, aes(x=Pars, y=Surp, color=Type, group=Type)) + geom_point()



plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point() + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1))


plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point() + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  +geom_text(data=data %>% filter(grepl("ground", Type)), aes(label=Language, alpha=1.0),hjust=0, vjust=0) + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0))+ geom_point(data=data %>% filter(grepl("REAL", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  +geom_text(data=data %>% filter(grepl("REAL", Type)), aes(label=Language, alpha=1.0),hjust=0, vjust=0)


plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point() + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0))



plot = ggplot(data, aes(x=(Pars-MeanParsRand)/SDParsRand, y=(Surp-MeanSurpRand)/SDSurpRand, color=Type, group=Type, alpha=0.9)) +geom_point() + geom_point(data=summarizedDataRand, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + geom_path(data=summarizedDataRand %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanParsRand)/SDParsRand, y=(Surp-MeanSurpRand)/SDSurpRand, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0))


dataGroundArrow = data %>% filter(grepl("ground", Type))
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z = (Pars-MeanPars)/SDPars, Surp_z = (Surp-MeanSurp)/SDSurp)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_end = (MeanParsRand-MeanPars)/SDPars, Surp_z_end = (MeanSurpRand-MeanSurp)/SDSurp)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_dir = Pars_z_end - Pars_z, Surp_z_dir = Surp_z_end - Surp_z)
dataGroundArrow = dataGroundArrow %>% mutate(z_length = sqrt(Pars_z_dir**2 + Surp_z_dir**2))
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_dir = Pars_z_dir/z_length, Surp_z_dir = Surp_z_dir/z_length)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_end = Pars_z + 0.2 * Pars_z_dir, Surp_z_end = Surp_z + 0.2 * Surp_z_dir)


plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point()  + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) + geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")

plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes()) + geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1)) + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")


#ggsave(plot, file="pareto-plane-best-balanced.pdf")



plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +
	geom_point()  + theme_bw() + theme(legend.position="none")  + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") +
#       	geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  +
       	geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) #+ 
#	geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type, alpha=1.0)) + 
#	geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + 
#	geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") 



plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +
	geom_point()  + theme_bw() + theme(legend.position="none")  + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") +
       	geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0)) + 
       	geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) + 
	geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type, alpha=1.0)) #+ 
#	geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + 
#	geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") 


plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point()  + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) + geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + geom_point(data=summarizedData%>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")








plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point()  + theme_bw() + geom_point(data=dataDepL, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")

#ggsave(plot, file="pareto-plane-depl-best-balanced.pdf")



iso = read.csv("../languages/languages-iso_codes.tsv")

data = merge(data, iso, by=c("Language"))

# geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  +
plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point()  + theme_bw() +  geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) + geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")  +geom_text(data=data %>% filter(grepl("ground", Type)), aes(label=iso_code, alpha=1.0),hjust=0.8, vjust=0.8)

plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type))   + theme_bw() +  geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes()) + geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_best_balanced", "manual_output_funchead_two_coarse_lambda09_best_large", "manual_output_funchead_langmod_coarse_best_balanced")), aes(x=Pars, y=Surp, size=5, group=1)) + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8)) + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")  +geom_text(data=data %>% filter(grepl("ground", Type)), aes(label=iso_code),hjust=0.8, vjust=0.8)


#ggsave(plot, file="pareto-plane-iso-best-balanced.pdf")



library(forcats)
data$Type = as.factor(data$Type)
data = data %>% mutate(TypeN = fct_recode(Type, "Parseability" = "manual_output_funchead_two_coarse_parser_best_balanced", "Baseline Languages" = "manual_output_funchead_RANDOM", "Efficiency" = "manual_output_funchead_two_coarse_lambda09_best_large", "Dependency Length" = "manual_output_funchead_coarse_depl", "Predictability" = "manual_output_funchead_langmod_coarse_best_balanced", "Real Languages" = "manual_output_funchead_ground_coarse_final"))
summarizedData = summarizedData %>% mutate(TypeN = fct_recode(Type, "Parseability" = "manual_output_funchead_two_coarse_parser_best_balanced", "Baseline Languages" = "manual_output_funchead_RANDOM", "Efficiency" = "manual_output_funchead_two_coarse_lambda09_best_large", "Dependency Length" = "manual_output_funchead_coarse_depl", "Predictability" = "manual_output_funchead_langmod_coarse_best_balanced", "Real Languages" = "manual_output_funchead_ground_coarse_final"))
dataGroundArrow = dataGroundArrow  %>% mutate(TypeN = fct_recode(Type, "Parseability" = "manual_output_funchead_two_coarse_parser_best_balanced", "Baseline Languages" = "manual_output_funchead_RANDOM", "Efficiency" = "manual_output_funchead_two_coarse_lambda09_best_large", "Dependency Length" = "manual_output_funchead_coarse_depl", "Predictability" = "manual_output_funchead_langmod_coarse_best_balanced", "Real Languages" = "manual_output_funchead_ground_coarse_final"))

#summarizedData = summarizedData %>% mutate(TypeN = factor(TypeN, levels=c("Parseability", "Predictability", "Efficiency", "Real Languages", "Baseline Languages", "Dependency Length"), ordered=TRUE))
#dataGroundArrow = dataGroundArrow %>% mutate(TypeN = factor(TypeN, levels=c("Parseability", "Predictability", "Efficiency", "Real Languages", "Baseline", "Dependency Length"), ordered=TRUE))


dataPlot = rbind(data %>% mutate(Pars =(Pars-MeanPars)/SDPars, Surp = (Surp-MeanSurp)/SDSurp) %>% select(Pars, Surp, TypeN, iso_code) %>% mutate(ParsMean=NA, SurpMean=NA), summarizedData %>% select(Pars, Surp, TypeN) %>% mutate(iso_code=NA) %>% rename(ParsMean=Pars,SurpMean=Surp)%>%mutate(Surp=NA,Pars=NA))
dataPlot = dataPlot %>% mutate(Pars_z = NA, Surp_z = NA, Pars_z_end = NA, Surp_z_end = NA)
dataPlot = rbind(dataPlot, dataGroundArrow %>% select(Pars, Surp, TypeN, Pars_z, Surp_z, Pars_z_end, Surp_z_end) %>% mutate(iso_code=NA) %>% mutate(ParsMean=NA, SurpMean=NA, Surp=NA, Pars=NA))


dataPlot = dataPlot %>% mutate(TypeN = factor(TypeN, levels=c("Parseability", "Predictability", "Efficiency", "Dependency Length", "Real Languages", "Baseline Languages"), ordered=TRUE))
forColoring  = factor(dataPlot$TypeN, levels=c("Real Languages", "Baseline Languages", "Parseability", "Predictability", "Efficiency", "Dependency Length"), ordered=TRUE)
#Create a custom color scale
library(RColorBrewer)
myColors <- brewer.pal(6,"Set1")
names(myColors) <- levels(forColoring)

plot = ggplot(dataPlot)  
plot = plot + theme_bw() 
#plot = plot + geom_segment(aes(x=-Pars_z, y=-Surp_z, xend=-Pars_z_end, yend = -Surp_z_end, color=TypeN, group=TypeN), size=0.6)
plot = plot + geom_density_2d(data=dataPlot %>% filter(grepl("Baseline", TypeN)), aes(x=-Pars, y=-Surp, color=TypeN, group=TypeN), size=0.3)
plot = plot + geom_path(data=dataPlot %>% filter(TypeN %in% c("Parseability", "Efficiency", "Predictability")), aes(x=-ParsMean, y=-SurpMean, group=1), color="gray", size=5)
#plot = plot + geom_point(data=dataPlot, aes(x=-ParsMean, y=-SurpMean, color=TypeN, group=TypeN), size=6)
plot = plot + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Predictability")
plot = plot + geom_text(data=dataPlot %>% filter(grepl("Real", TypeN)), aes(x=-Pars, y=-Surp, color=TypeN, group=TypeN, label=iso_code),hjust=0.8, vjust=0.8)
plot = plot + theme(legend.title = element_blank())  
plot = plot + guides(color=guide_legend(nrow=2,ncol=4,byrow=TRUE)) 
plot = plot + theme(legend.title = element_blank(), legend.position="bottom")
plot = plot + scale_colour_manual(name = "TypeN",values = myColors)
plot = plot + theme(axis.title.x = element_text(size=17))
plot = plot + theme(axis.title.y = element_text(size=17))
plot = plot + theme(legend.text = element_text(size=12))

#ggsave(plot, file="pareto-plane-iso-best-balanced-legend-viz.pdf")

#TODO
# get the full Pareto frontier
# quantify direction of arrows
# visualize ISO codes without crowding
# uncertainty from corpus size


library(ggrepel)

plot = ggplot(dataPlot)  
plot = plot + theme_bw() 
#plot = plot + geom_segment(aes(x=-Pars_z, y=-Surp_z, xend=-Pars_z_end, yend = -Surp_z_end, color=TypeN, group=TypeN), size=0.6)
plot = plot + geom_density_2d(data=dataPlot %>% filter(grepl("Baseline", TypeN)), aes(x=-Pars, y=-Surp, color=TypeN, group=TypeN), size=0.3)
plot = plot + geom_path(data=dataPlot %>% filter(TypeN %in% c("Parseability", "Efficiency", "Predictability")), aes(x=-ParsMean, y=-SurpMean, group=1), color="gray", size=5)
#plot = plot + geom_point(data=dataPlot, aes(x=-ParsMean, y=-SurpMean, color=TypeN, group=TypeN), size=6)
plot = plot + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Predictability")
plot = plot + geom_text_repel(data=dataPlot %>% filter(grepl("Real", TypeN)), aes(x=-Pars, y=-Surp, color=TypeN, group=TypeN, label=iso_code),hjust=0.8, vjust=0.8, force=0.01, point.padding = NA, segment.size=0)
plot = plot + theme(legend.title = element_blank())  
plot = plot + guides(color=guide_legend(nrow=2,ncol=4,byrow=TRUE)) 
plot = plot + theme(legend.title = element_blank(), legend.position="bottom")
plot = plot + scale_colour_manual(name = "TypeN",values = myColors)
plot = plot + theme(axis.title.x = element_text(size=17))
plot = plot + theme(axis.title.y = element_text(size=17))
plot = plot + theme(legend.text = element_text(size=12))

#ggsave(plot, file="pareto-plane-iso-best-balanced-legend-viz-2.pdf")



plot = ggplot(dataPlot)  
plot = plot + theme_bw() 
#plot = plot + geom_segment(aes(x=-Pars_z, y=-Surp_z, xend=-Pars_z_end, yend = -Surp_z_end, color=TypeN, group=TypeN), size=0.6)
plot = plot + geom_density_2d(data=dataPlot %>% filter(grepl("Baseline", TypeN)), aes(x=-Pars, y=-Surp, color=TypeN, group=TypeN), size=0.3, bins=5)
plot = plot + geom_path(data=dataPlot %>% filter(TypeN %in% c("Parseability", "Efficiency", "Predictability")), aes(x=-ParsMean, y=-SurpMean, group=1), color="gray", size=5)
#plot = plot + geom_point(data=dataPlot, aes(x=-ParsMean, y=-SurpMean, color=TypeN, group=TypeN), size=6)
plot = plot + scale_x_continuous(name="Parsability") + scale_y_continuous(name="Predictability")
plot = plot + geom_text(data=dataPlot %>% filter(grepl("Real", TypeN)), aes(x=-Pars, y=-Surp, color=TypeN, group=TypeN, label=iso_code),hjust=0.8, vjust=0.8)
plot = plot + theme(legend.title = element_blank())  
plot = plot + guides(color=guide_legend(nrow=2,ncol=4,byrow=TRUE)) 
plot = plot + theme(legend.title = element_blank(), legend.position="bottom")
plot = plot + scale_colour_manual(name = "TypeN",values = myColors)
plot = plot + theme(axis.title.x = element_text(size=17))
plot = plot + theme(axis.title.y = element_text(size=17))
plot = plot + theme(legend.text = element_text(size=12))
#plot = plot + annotate("text", x=-1.5, y=-1.3, label = "Baselines", hjust=0, size=8, color="blue")

ggsave(plot, file="pareto-plane-iso-best-balanced-legend-viz-10.pdf")




