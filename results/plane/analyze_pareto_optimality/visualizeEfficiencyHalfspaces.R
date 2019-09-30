# per language


library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
depl = read.csv("../../../grammars/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(tidyr)
library(dplyr)
depl = depl %>% filter(grepl("FuncHead", ModelName)) %>% filter(grepl("Coarse", ModelName))
dataS  = read.csv("../../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS3 = read.csv("../../../grammars/plane/plane-fixed-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS4 = read.csv("../../../grammars/plane/plane-fixed-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS5 = read.csv("../../../grammars/plane/plane-fixed-random3.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS6 = read.csv("../../../grammars/plane/plane-fixed-random4.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS7 = read.csv("../../../grammars/plane/plane-fixed-random5.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS = rbind(dataS, dataS2, dataS3, dataS4, dataS5, dataS6, dataS7)
dataP = read.csv("../../../grammars/plane/plane-parse-redone.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surprisal = mean(Surp, na.rm=TRUE))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(Pars = mean(ParsU, na.rm=TRUE))
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




dataGroundArrow = data %>% filter(grepl("ground", Type))
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z = (Pars-MeanPars)/SDPars, Surprisal_z = (Surprisal-MeanSurprisal)/SDSurprisal)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_end = (MeanParsRand-MeanPars)/SDPars, Surprisal_z_end = (MeanSurprisalRand-MeanSurprisal)/SDSurprisal)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_dir = Pars_z_end - Pars_z, Surprisal_z_dir = Surprisal_z_end - Surprisal_z)
dataGroundArrow = dataGroundArrow %>% mutate(z_length = sqrt(Pars_z_dir**2 + Surprisal_z_dir**2))
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_dir = Pars_z_dir/z_length, Surprisal_z_dir = Surprisal_z_dir/z_length)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_end = Pars_z + 0.5 * Pars_z_dir, Surprisal_z_end = Surprisal_z + 0.5 * Surprisal_z_dir)







data_pareto = read.csv("~/CS_SCR/posteriors/pareto-smooth/pareto-total.csv")



data_pareto_ = data_pareto %>% group_by(Language) %>% summarise(bestLambdas_mean = 0.9, bestLambdas_sd = 0.01)
data_pareto_ = merge(data %>% filter(Type == "manual_output_funchead_ground_coarse_final"), data_pareto_, by=c("Language"))
data_pareto_line1 = data_pareto_ %>% mutate(Pars_z = ((Pars + 10)-MeanPars)/SDPars, Surprisal_z = ((Surprisal + 10*bestLambdas_mean)-MeanSurprisal)/SDSurprisal)
data_pareto_line2 = data_pareto_ %>% mutate(Pars_z = ((Pars - 10)-MeanPars)/SDPars, Surprisal_z = ((Surprisal - 10*bestLambdas_mean)-MeanSurprisal)/SDSurprisal)
data_pareto_line = rbind(data_pareto_line1, data_pareto_line2)


plot = ggplot(data %>% filter(Type %in% c("manual_output_funchead_RANDOM", "manual_output_funchead_ground_coarse_final")) %>% filter(Surprisal_z < 3), aes(x=-Pars_z, y=-Surprisal_z, color=Type, group=Type))
plot = plot + geom_density_2d(data=data %>% filter(Type == "manual_output_funchead_RANDOM") %>% filter(Surprisal_z < 3), aes(x=-Pars_z, y=-Surprisal_z, color=Type, group=Type), size=0.3, bins=5)
plot = plot + geom_path(data=subData, aes(x=-Pars_z, y=-Surprisal_z, group=1), size=1.5)
plot = plot + geom_point(data=data %>% filter(Type == "manual_output_funchead_ground_coarse_final"), size=3)
#plot = plot + geom_line(data=data_pareto_line, aes(x=-Pars_z, y=-Surprisal_z, group=Language))
plot = plot + geom_abline(data=data_pareto_, aes(intercept=  -Surprisal_z - 1/((bestLambdas_mean+1e-10)/SDSurprisal) * Pars_z , slope=-1/((bestLambdas_mean+1e-10)/SDSurprisal), group=Language))
plot = plot + geom_segment(data=dataGroundArrow, aes(x=-Pars_z, y=-Surprisal_z, xend=-Pars_z_end, yend = -Surprisal_z_end, color=Type, group=Type), size=2)
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
ggsave(plot, file="pareto-plane-perLanguage-arrows-smoothed-halfspace.pdf", width=12, height=12)


data_baselines = read.csv("pareto-data.tsv")

data_baselines = merge(data_pareto_%>% select(Language, bestLambdas_mean), data_baselines, by=c("Language"))
data_baselines = data_baselines %>% mutate(Eff = Pars + bestLambdas_mean * Surprisal, EffGround = ParsGround + bestLambdas_mean * SurprisalGround)

################################### TODO somehow this doesn't work yet

plot = ggplot(data_baselines)
plot = plot + geom_density(aes(x=Eff, y=..scaled..))
plot = plot + geom_bar(data=data_baselines %>% group_by(Language) %>% summarise(EffGround = mean(EffGround)), stat="identity", aes(x=EffGround, y=1), size=0.5)
plot = plot + facet_wrap(~Language, scales="free")

####################################

