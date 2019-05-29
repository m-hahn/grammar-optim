



library(tidyr)
library(dplyr)
library(ggplot2)




dataS = read.csv("plane-fixed.tsv", sep="\t")
dataP = read.csv("plane-parse.tsv", sep="\t")

dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surp = mean(Surp))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS), Pars = mean(Pars))

dataS = as.data.frame(dataS)
dataP = as.data.frame(dataP)

#library(lme4)
#summary(lmer(Surp ~ Type + (1|Language), data=dataS %>% filter(grepl("langm", Type))))



dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))

dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))

dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))


data = merge(dataS, dataP, by=c("Language", "Model", "Type"))

#data = data %>% filter(Surp < 20)
#data = data %>% filter(Pars < 4)

data = data %>% mutate(Two = Surp+Pars)

data2 = rbind(data)
data2 = data2 %>% group_by(Language) %>% mutate(MeanSurp = mean(Surp), SDSurp = sd(Surp)) %>% mutate(MeanPars = mean(Pars), SDPars = sd(Pars))
plot = ggplot(data2, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point()
#plot = ggplot(data, aes(x=(Pars), y=(Surp), color=Type, group=Type)) +geom_point()


#dataMean = data %>% group_by(Language) %>% summarise(MeanSurp = mean(Surp), SDSurp = sd(Surp)+0.0001)
#data_ = merge(data, dataMean, by=c("Language"))
#dataMean = data %>% group_by(Language) %>% summarise(MeanUAS = mean(UAS), SDUAS = sd(UAS)+0.0001)
#data_ = merge(data_, dataMean, by=c("Language"))
#
#plot = ggplot(data_, aes(x=(UAS), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point()

dataPBest = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_final") %>% group_by(Language) %>% summarise(Pars = min(Pars))
data2Best = data %>% filter(Type == "manual_output_funchead_two_coarse_final") %>% group_by(Language) %>% summarise(Two = min(Two))
dataSBest = data %>% filter(Type == "manual_output_funchead_langmod_coarse_final") %>% group_by(Language) %>% summarise(Surp = min(Surp))

dataP = data %>% filter(Type == "manual_output_funchead_two_coarse_parser_final")
data2 = data %>% filter(Type == "manual_output_funchead_two_coarse_final")
dataS = data %>% filter(Type == "manual_output_funchead_langmod_coarse_final")

dataP = merge(dataP, dataPBest, by=c("Language", "Pars"))
data2 = merge(data2, data2Best, by=c("Language", "Two"))
dataS = merge(dataS, dataSBest, by=c("Language", "Surp"))


dataRandom = data %>% filter(grepl("RANDOM", Type))
dataDepL = data %>% filter(grepl("depl", Type)) #%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))
dataGround = data %>% filter(grepl("ground", Type)) #%>% group_by(Language) %>% summarise(Surp = mean(Surp), Pars = mean(Pars))


data = rbind(dataP, data2)
data = rbind(data, dataS)
data = rbind(data, dataRandom)
#data = rbind(data, dataDepL)
data = rbind(data, dataGround)

dataMean = data %>% group_by(Language) %>% summarise(MeanSurp = mean(Surp), SDSurp = sd(Surp)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataMean = data %>% group_by(Language) %>% summarise(MeanPars = mean(Pars), SDPars = sd(Pars)+0.0001)
data = merge(data, dataMean, by=c("Language"))

dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanSurpRand = mean(Surp), SDSurpRand = sd(Surp)+0.0001)
data = merge(data, dataMean, by=c("Language"))
dataMean = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(MeanParsRand = mean(Pars), SDParsRand = sd(Pars)+0.0001)
data = merge(data, dataMean, by=c("Language"))




#plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point()
#plot = ggplot(data, aes(x=(Pars-MeanPars), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point()
#
#plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point() + facet_wrap(~Language)
#
#plot = ggplot(data, aes(x=(Pars-MeanPars), y=(Surp-MeanSurp), color=Type, group=Type)) +geom_point() + facet_wrap(~Language)


subData = data %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_final", "manual_output_funchead_two_coarse_final", "manual_output_funchead_langmod_coarse_final"))

subData = subData[order(subData$Type),]

#plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point() + geom_path(data=subData, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, group=1)) + facet_wrap(~Language)


summarizedData = data %>% group_by(Type) %>% summarise(Surp = mean((Surp-MeanSurp)/SDSurp), Pars = mean((Pars-MeanPars)/SDPars))
#plot = ggplot(summarizedData, aes(x=Pars, y=Surp, color=Type, group=Type)) + geom_point()


summarizedDataRand = data %>% group_by(Type) %>% summarise(Surp = mean((Surp-MeanSurpRand)/SDSurpRand), Pars = mean((Pars-MeanParsRand)/SDParsRand))
#plot = ggplot(summarizedDataRand, aes(x=Pars, y=Surp, color=Type, group=Type)) + geom_point()



#plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type)) +geom_point() + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_final", "manual_output_funchead_two_coarse_final", "manual_output_funchead_langmod_coarse_final")), aes(x=Pars, y=Surp, size=5, group=1))


#plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point() + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_final", "manual_output_funchead_two_coarse_final", "manual_output_funchead_langmod_coarse_final")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0))



#plot = ggplot(data, aes(x=(Pars-MeanParsRand)/SDParsRand, y=(Surp-MeanSurpRand)/SDSurpRand, color=Type, group=Type, alpha=0.9)) +geom_point() + geom_point(data=summarizedDataRand, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + geom_path(data=summarizedDataRand %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_final", "manual_output_funchead_two_coarse_final", "manual_output_funchead_langmod_coarse_final")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanParsRand)/SDParsRand, y=(Surp-MeanSurpRand)/SDSurpRand, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0))


dataGroundArrow = data %>% filter(grepl("ground", Type))
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z = (Pars-MeanPars)/SDPars, Surp_z = (Surp-MeanSurp)/SDSurp)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_end = (MeanParsRand-MeanPars)/SDPars, Surp_z_end = (MeanSurpRand-MeanSurp)/SDSurp)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_dir = Pars_z_end - Pars_z, Surp_z_dir = Surp_z_end - Surp_z)
dataGroundArrow = dataGroundArrow %>% mutate(z_length = sqrt(Pars_z_dir**2 + Surp_z_dir**2))
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_dir = Pars_z_dir/z_length, Surp_z_dir = Surp_z_dir/z_length)
dataGroundArrow = dataGroundArrow %>% mutate(Pars_z_end = Pars_z + 0.2 * Pars_z_dir, Surp_z_end = Surp_z + 0.2 * Surp_z_dir)


plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point()  + theme_bw() + geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  + geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) + geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_final", "manual_output_funchead_two_coarse_final", "manual_output_funchead_langmod_coarse_final")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parseability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")



ggsave(plot, file="pareto-plane.pdf")

iso = read.csv("languages-iso_codes.tsv")

data = merge(data, iso, by=c("Language"))

# geom_point(data=data %>% filter(grepl("ground", Type)), aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=1.0))  +
plot = ggplot(data, aes(x=(Pars-MeanPars)/SDPars, y=(Surp-MeanSurp)/SDSurp, color=Type, group=Type, alpha=0.9)) +geom_point()  + theme_bw() +  geom_density_2d(data=data %>% filter(grepl("RANDOM", Type)), aes(alpha=1.0)) + geom_segment(data=dataGroundArrow, aes(x=Pars_z, y=Surp_z, xend=Pars_z_end, yend = Surp_z_end, color=Type, group=Type, alpha=1.0)) + geom_path(data=summarizedData %>% filter(Type %in% c("manual_output_funchead_two_coarse_parser_final", "manual_output_funchead_two_coarse_final", "manual_output_funchead_langmod_coarse_final")), aes(x=Pars, y=Surp, size=5, group=1, alpha=1.0)) + geom_point(data=summarizedData, aes(x=Pars, y=Surp, size=8, alpha=1.0)) + scale_x_continuous(name="Parseability") + scale_y_continuous(name="Surprisal") + theme(legend.position="none")  +geom_text(data=data %>% filter(grepl("ground", Type)), aes(label=iso_code, alpha=1.0),hjust=0.8, vjust=0.8)

ggsave(plot, file="pareto-plane-iso.pdf")





