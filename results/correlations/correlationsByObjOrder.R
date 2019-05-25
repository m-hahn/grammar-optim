
library(tidyr)
library(dplyr)
library(lme4)
library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)

data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek" = "Ancient"))



dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")


ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")


annotateCorrelations = function(data) {
    data = data %>% select(CoarseDependency, Language, FileName, DH_Weight)
    D = data %>% filter(CoarseDependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
    data = merge(data, D, by=c("FileName"))
    data = data %>% filter(CoarseDependency %in% ofInterest)
    data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))
    reverseOrder = c("aux")
    data[data$CoarseDependency %in% reverseOrder,]$Dir = 1-data[data$CoarseDependency %in% reverseOrder,]$Dir
    D = data %>% select(Language, FileName, CoarseDependency, Dir, DirObj)
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
dataP = read.csv("../../grammars/plane/plane-parse.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP2 = read.csv("../../grammars/plane/plane-parse-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP3 = read.csv("../../grammars/plane/plane-parse-best-large.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP4 = read.csv("../../grammars/plane/plane-parse-random2.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = rbind(dataP, dataP2, dataP3, dataP4)
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surp = mean(Surp, na.rm=TRUE))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS, na.rm=TRUE), Pars = mean(Pars, na.rm=TRUE))
dataS = as.data.frame(dataS)
dataP = as.data.frame(dataP)
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



trafo_surp = read.csv("../plane/surp-z.csv")
trafo_pars = read.csv("../plane/pars-z.csv")

data = merge(data, trafo_surp, by=c("Language"))
data = merge(data, trafo_pars, by=c("Language"))

data = data %>% mutate(Surp_z = (Surp-MeanSurp)/SDSurp, Pars_z = (Pars-MeanPars)/SDPars)



data = merge(E, data, by=c("Language", "Model"))



uEff = data %>% filter(grepl("large", Type)) %>% group_by(Type, Language, Model) %>% summarise(DirObj = mean(DirObj), Dir=sum(Dir)-4)
uEff = merge(uEff, read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/most-successful-seeds.tsv"), by=c("Language", "Model", "Type")) %>% select(Type, Language, Model, DirObj, Dir)
plot = ggplot(uEff, aes(y=DirObj, x=Dir, group=Model)) + geom_point() + geom_jitter()


uReal = data %>% filter(grepl("ground", Type)) %>% group_by(Type, Language, Model) %>% summarise(DirObj = mean(DirObj), Dir=sum(Dir)-4)
plot = ggplot(uReal, aes(y=DirObj, x=Dir, group=Model)) + geom_point() + geom_jitter()


uRand = data %>% filter(grepl("RANDOM", Type)) %>% group_by(Type, Language, Model) %>% summarise(DirObj = mean(DirObj), Dir=sum(Dir)-4)
plot = ggplot(uRand, aes(y=DirObj, x=Dir, group=Model)) + geom_point() + geom_jitter()

u = rbind(uEff, as.data.frame(uReal), as.data.frame(uRand))


u = u %>% mutate(Type = case_when(Type == "manual_output_funchead_ground_coarse_final" ~ "Real", Type == "manual_output_funchead_two_coarse_lambda09_best_large" ~ "Optimized", Type == "manual_output_funchead_RANDOM" ~ "Baselines"))
u = u %>% mutate(DirObj = case_when(DirObj == 0 ~ "Verb-Object", DirObj == 1 ~ "Object-Verb"))




plot = ggplot(u, aes(x=Dir+4, y=..count../max(..count..), fill=Type, color=Type)) + geom_bar() + facet_wrap(~DirObj + Type, nrow=2, scales="free_y") + xlim(-0.5,8.5) + theme_classic() + ylab("Grammars")  + theme(legend.position = "none") + xlab("Object Patterners Preceding Verb Patterners") +
theme(
  strip.background = element_blank(),
  strip.text.x = element_blank()
) +  theme(axis.text.y = element_blank())

ggsave(plot, file="figures/correlations-histograms.pdf", height=7, width=7)


#u = u %>% group_by(DirObj, Type, Dir) %>% summarise(count = NROW(Dir))
#v = u %>% group_by(DirObj, Type) %>% summarise(maxCount = max(count))
#u = merge(u, v, by=c("DirObj", "Type"))
#plot = ggplot(u, aes(x=Dir+4, y=count/maxCount, color=DirObj)) + geom_density(size=2) + facet_wrap(~ Type, nrow=1, scales="free_y") + xlim(-0.5,8.5) + theme_classic() + ylab("Grammars")  + theme(legend.position = "none") + xlab("Object Patterners Following Verb Patterners") +

plot = ggplot(u, aes(x=Dir+4, y=..scaled.., color=DirObj)) + geom_density(size=1.5, linetype="dashed") + facet_wrap(~ Type, nrow=1, scales="free_y") + xlim(-0.5,8.5) + theme_classic() + ylab("Grammars")   + xlab("Object Patterners Preceding Verb Patterners")
plot = plot +  theme(axis.text.y = element_blank())
plot = plot + theme(legend.position="bottom") +  theme(legend.title = element_blank())
plot = plot + theme(        legend.margin=margin(0,0,0,0), legend.box.margin=margin(-5,-10,2,-10))

ggsave(plot, file="figures/correlations-curve.pdf", height=3, width=7)


plot = ggplot(u, aes(x=Dir+4, y=..scaled.., color=DirObj)) 
plot = plot + geom_density(size=1.5, linetype="dashed")
plot = plot + facet_wrap(~ Type, nrow=1, scales="free_y") + xlim(-0.5,8.5) + theme_classic() + ylab("Grammars")   + xlab("Object Patterners Preceding Verb Patterners")
plot = plot + theme(axis.text.y = element_blank())
plot = plot + theme(legend.position="bottom") +  theme(legend.title = element_blank())
plot = plot + theme(        legend.margin=margin(0,0,0,0), legend.box.margin=margin(-5,-10,2,-10))
plot = plot + geom_segment(data=data.frame(x=c(1), xend=c(7), y=c(0), Type=c("Baselines")), aes(x=x, xend=xend, y=0, yend=0), color="white", size=1.8)
plot = plot + geom_segment(data=data.frame(x=c(0), xend=c(8), y=c(0), Type=c("Optimized")), aes(x=x, xend=xend, y=0, yend=0), color="white", size=1.8)
plot = plot + geom_segment(data=data.frame(x=c(0), xend=c(3.5), y=c(0), Type=c("Real")), aes(x=x, xend=xend, y=0, yend=0), color="white", size=1.8)

ggsave(plot, file="figures/correlations-curve-whiteaxis.pdf", height=2.2, width=5.5)




# scale_x_continuous(breaks=c(0,2,4,6,8))



#
#F = E %>% group_by(Language, Model) %>% summarise(Agree = sum(Agree, na.rm=TRUE))
#F = merge(F, data, by=c("Language", "Model"))
#
#G = E %>% select(Language, Model, CoarseDependency, Agree)
#G = merge(G, data, by=c("Language", "Model"))
#
#
##D$AgreeB = as.factor(as.character(round(pmax(0, pmin(1, D$Agree)))))
#
#
#
#
#u = F %>% filter(grepl("ground", Type))
#cor.test(u$Agree, 0.9*u$Surp_z+u$Pars_z)
#
## Robustness check:
#cor.test(u$Agree, 0.9*u$Surp_z+u$Pars_z, method="spearman")
#
#
#
#
#plot = ggplot(u, aes(x=-Surp_z, y=-Pars_z, color=Agree)) + geom_point()
#
#
