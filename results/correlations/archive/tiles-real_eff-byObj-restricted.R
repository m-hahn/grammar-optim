
data = read.csv("../../grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(dplyr)
library(tidyr)
library(ggplot2)
data = data %>% mutate(DH_Weight = DH_Mean_NoPunct)
library(forcats)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek" = "Ancient"))
languages = read.csv("../languages/languages-iso_codes.tsv", sep=",")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")
# , "amod", "nummod"
data = data %>% select(Dependency, Family, Language, FileName, DH_Weight)
D = data %>% filter(Dependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
data = merge(data, D, by=c("FileName"))
data = data %>% filter(Dependency %in% ofInterest)
data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))
reverseOrder = c("aux")
data[data$Dependency %in% reverseOrder,]$Agree = 1-data[data$Dependency %in% reverseOrder,]$Agree
D = data %>% group_by(Language, Family, Dependency) %>% summarise(Dir = mean(Dir), DirObj = mean(DirObj))
D_Ground = rbind(D)

D$Direction = ifelse(D$DirObj == 1, "OV", "VO")

ordersByLanguage = unique(data.frame(Language = D$Language, Direction = as.character(D$Direction)))


data = read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_balanced/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
#best = read.csv("../strongest_models/best-two-lambda09-best-balanced.csv")
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek"= "Ancient"))
#best = best %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek"= "Ancient"))
#data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))
data = data
languages = read.csv("../languages/languages-iso_codes.tsv", sep=",")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")
data = data %>% select(CoarseDependency, Family, Language, FileName, DH_Weight)
D = data %>% filter(CoarseDependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
data = merge(data, D, by=c("FileName"))
data = data %>% filter(CoarseDependency %in% ofInterest)
data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))
reverseOrder = c("aux")
data[data$CoarseDependency %in% reverseOrder,]$Agree = 1-data[data$CoarseDependency %in% reverseOrder,]$Agree
data$Direction = ifelse(data$DirObj == 1, "OV", "VO")
D = merge(data, ordersByLanguage, by=c("Language", "Direction"))
D = D %>% group_by(Language, Family, CoarseDependency) %>% summarise(Dir = mean(Dir), DirObj = mean(DirObj))

D_Eff = D




D_Ground = D_Ground %>% mutate(CoarseDependency = Dependency)
D_Ground$Type = "Real Languages"
D_Eff$Type = "Efficiency"

D = rbind(D_Ground, D_Eff)

D[D$CoarseDependency == "aux",]$Dir = 1-D[D$CoarseDependency == "aux",]$Dir

library(scales)

D$DirB = as.factor(as.character(round(pmax(0, pmin(1, D$Dir)))))
D$DirB=D$Dir
D_Ground_Mean = D_Ground %>% group_by(Language) %>% summarise(Dir = sum(DirObj))
D$Language_Ordered = factor(D$Language, levels=unique(D_Ground_Mean[order(D_Ground_Mean$Dir),]$Language), ordered=TRUE)

D = D[order(D$Language_Ordered),]

plot = ggplot(D, aes(x = CoarseDependency, y = Language_Ordered)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + facet_wrap(~Type)

ggsave(file="figures/coverage-ground_eff-byObj-circles.pdf", plot=plot)


#E = D %>% group_by(Language) %>% mutate(AgreeSum = sum(Agree))
##E = E[order(E$AgreeSum),]
#ordered_languages = unique(E$Language)
##write.csv(ordered_languages, file="ordered_languages.csv")
#
#E = E %>% group_by(Dependency) %>% mutate(AgreeSumDep = sum(Agree))
#E = E[order(-E$AgreeSumDep),]
#ordered_deps = unique(E$Dependency)
#write.csv(ordered_deps, file="ordered_dependencies.csv")
#
#
#
#plot = ggplot(E, aes(x = factor(Dependency, levels=ordered_deps), y = factor(Language, levels=ordered_languages))) + 
#  geom_tile(aes(fill=Agree)) + 
#  scale_fill_gradient(low="white", high="red", limits=c(0.5, 1), oob=squish) +
#  labs(x="Correlations", y="Languages", title="Matrix") +
#  theme_bw() + theme(axis.text.x=element_text(size=9, angle=0, vjust=0.3),
#                     axis.text.y=element_text(size=9),
#                     plot.title=element_text(size=11))
#
#
#ggsave(file="coverage-ground-ordered.png", plot=plot)
#
#
#
