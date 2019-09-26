# Second Paragraph of Study 2: Correlation between efficiency and the number of satisfied correlations, for real grammars.



library(tidyr)
library(dplyr)
library(lme4)
library(ggplot2)
library(forcats)



ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")


annotateCorrelations = function(data) {
    data = data %>% select(CoarseDependency, Language, FileName, DH_Weight)
    D = data %>% filter(CoarseDependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
    data = merge(data, D, by=c("FileName"))
    data = data %>% filter(CoarseDependency %in% ofInterest)
    data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))
    reverseOrder = c("aux")
    data[data$CoarseDependency %in% reverseOrder,]$Agree = 1-data[data$CoarseDependency %in% reverseOrder,]$Agree
    D = data %>% group_by(Language, FileName, CoarseDependency) %>% summarise(Agree = mean(Agree))
    return(D)
}


E = data.frame(Language = c(), FileName = c(), Dependency = c(), Agree = c())


data = read.csv("../../grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
data = data %>% mutate(DH_Weight = DH_Mean_NoPunct, CoarseDependency = Dependency)
corrs = annotateCorrelations(data)
E = corrs


data = read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
corrs = annotateCorrelations(data)
E = rbind(E, corrs)


data = read.csv("../../grammars/manual_output_funchead_two_coarse_parser_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
corrs = annotateCorrelations(data)
E = rbind(E, corrs)

data = read.csv("../../grammars/manual_output_funchead_langmod_coarse_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
corrs = annotateCorrelations(data)
E = rbind(E, corrs)


data = read.csv("../../grammars/manual_output_funchead_RANDOM/auto-summary-lstm.tsv", sep="\t")
corrs = annotateCorrelations(data)
E = rbind(E, corrs)

data = read.csv("../../grammars/manual_output_funchead_RANDOM2/auto-summary-lstm.tsv", sep="\t")
corrs = annotateCorrelations(data)
E = rbind(E, corrs)




E = E %>% mutate(Model = as.character(FileName))

depl = read.csv("../../grammars/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)
depl = depl %>% filter(grepl("FuncHead", ModelName)) %>% filter(grepl("Coarse", ModelName))
dataS = read.csv("../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS3 = read.csv("../../grammars/plane/plane-fixed-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS4 = read.csv("../../grammars/plane/plane-fixed-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model)) %>% mutate(FullSurp = NULL)
dataS = rbind(dataS, dataS2, dataS3, dataS4)
dataP = read.csv("../../grammars/plane/controls/plane-parse2.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surp = mean(Surp, na.rm=TRUE))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS, na.rm=TRUE), Pars = mean(Pars, na.rm=TRUE))
dataS = as.data.frame(dataS)
dataP = as.data.frame(dataP)
dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)


data = data %>% mutate(Type = ifelse(grepl("RANDOM", Type), "manual_output_funchead_RANDOM", as.character(Type)))

trafo_surp = read.csv("../plane/surp-z.csv")
trafo_pars = read.csv("../plane/pars-z.csv")

data = merge(data, trafo_surp, by=c("Language"))
data = merge(data, trafo_pars, by=c("Language"))

data = data %>% mutate(Surp_z = (Surp-MeanSurp)/SDSurp, Pars_z = (Pars-MeanPars)/SDPars, Eff_z =  0.9 * (Surp-MeanSurp)/SDSurp + (Pars-MeanPars)/SDPars)


F = E %>% group_by(Language, Model) %>% summarise(Agree = sum(Agree, na.rm=TRUE))
F = merge(F, data, by=c("Language", "Model"))

u = F %>% filter(grepl("ground", Type))
cor.test(u$Agree, 0.9*u$Surp_z+u$Pars_z)

# Robustness check:
cor.test(u$Agree, 0.9*u$Surp_z+u$Pars_z, method="spearman")
cor.test(u$Agree, 0.9*u$Surp_z+u$Pars_z, method="kendal")

library("ggpubr")
plot = ggplot(u, aes(x=Agree, y=-(0.9*u$Surp_z+u$Pars_z)))
plot = plot + geom_point()
plot = plot + geom_smooth(method="lm")
plot = plot + xlab("Satisfied Correlations")
plot = plot + ylab("Efficiency")
plot = plot + theme_bw()
plot = plot + theme(axis.title=element_text(size=20))
plot = plot + stat_cor(method = "pearson") #, label.x = 2, label.y = 0.8)
#plot = plot + stat_cor(method = "spearman", label.x = 2, label.y = 0.5)
ggsave(plot, file="correlations-by-grammar/ground-corrs-efficiency.pdf")

plot = ggplot(u, aes(x=Agree, y=-u$Surp_z))
plot = plot + geom_point()
plot = plot + geom_smooth(method="lm")
plot = plot + xlab("Satisfied Correlations")
plot = plot + ylab("Predictability")
plot = plot + theme_bw()
plot = plot + theme(axis.title=element_text(size=20))
plot = plot + stat_cor(method = "pearson") #, label.x = 2, label.y = 0.8)
ggsave(plot, file="correlations-by-grammar/ground-corrs-predictability.pdf")

plot = ggplot(u, aes(x=Agree, y=-u$Pars_z))
plot = plot + geom_point()
plot = plot + geom_smooth(method="lm")
plot = plot + xlab("Satisfied Correlations")
plot = plot + ylab("Parseability")
plot = plot + theme_bw()
plot = plot + theme(axis.title=element_text(size=20))
plot = plot + stat_cor(method = "pearson") #, label.x = 2, label.y = 0.8)
ggsave(plot, file="correlations-by-grammar/ground-corrs-parseability.pdf")



