ofInterest =  c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")

data = read.csv("../../grammars/manual_output_funchead_ground_coarse_final/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(dplyr)
library(tidyr)
library(ggplot2)
data = data %>% mutate(DH_Weight = DH_Mean_NoPunct)
library(forcats)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek" = "Ancient"))
languages = read.csv("../languages/languages-iso_codes.tsv", sep=",")
languages = languages %>% mutate(Family = ifelse(grepl("Dravid", Family), "Dravidian", as.character(Family)))
languages = languages %>% mutate(Family = ifelse(grepl("Turk", Family), "Turkic", as.character(Family)))
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
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


data = read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
best = read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/most-successful-seeds.tsv")
bestEff = best %>% group_by(Language) %>% summarise(Efficiency = min(Efficiency))
best = merge(best, bestEff, by=c("Language", "Efficiency"))
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
data = data %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek"= "Ancient"))
best = best %>% mutate(Language = fct_recode(Language, "Old_Church_Slavonic" = "Old", "Ancient_Greek"= "Ancient"))
data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))
data = data
languages = read.csv("../languages/languages-iso_codes.tsv", sep=",")
languages = languages %>% mutate(Family = ifelse(grepl("Dravid", Family), "Dravidian", as.character(Family)))
languages = languages %>% mutate(Family = ifelse(grepl("Turk", Family), "Turkic", as.character(Family)))
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
data = data %>% select(CoarseDependency, Family, Language, FileName, DH_Weight)
D = data %>% filter(CoarseDependency == "obj") %>% rename(DH_Weight_Obj = DH_Weight) %>% select(DH_Weight_Obj, FileName)
data = merge(data, D, by=c("FileName"))
data = data %>% filter(CoarseDependency %in% ofInterest)
data = data %>% mutate(Dir = pmax(0, sign(DH_Weight)), DirObj = pmax(0, sign(DH_Weight_Obj))) %>% mutate(Agree = (Dir == DirObj))
reverseOrder = c("aux")
data[data$CoarseDependency %in% reverseOrder,]$Agree = 1-data[data$CoarseDependency %in% reverseOrder,]$Agree
data$Direction = ifelse(data$DirObj == 1, "OV", "VO")
data2 = rbind(data) %>% mutate(Direction = ifelse(Direction == "OV", "VO", "OV"), Dir = 1-Dir, DirObj = 1-DirObj)
	data = rbind(data, data2)
D = merge(data, ordersByLanguage, by=c("Language", "Direction"))
D = D %>% group_by(Language, Family, CoarseDependency) %>% summarise(Dir = mean(Dir), DirObj = mean(DirObj))

D_Eff = D




D_Ground = D_Ground %>% mutate(CoarseDependency = Dependency)
D_Ground$Type = "Real Languages"
D_Eff$Type = "Efficiency"

D = rbind(D_Ground, D_Eff)

D[D$CoarseDependency == "aux",]$Dir = 1-D[D$CoarseDependency == "aux",]$Dir


#D = D %>% filter(CoarseDependency == "lifted_case")

library(scales)

D$DirB = as.factor(as.character(round(pmax(0, pmin(1, D$Dir)))))
D$DirB=D$Dir
D_Ground_Mean = D_Ground %>% group_by(Language, Family) %>% summarise(Dir = sum(DirObj))
D$Language_Ordered = factor(D$Language, levels=unique(D_Ground_Mean[order(D_Ground_Mean$Family),]$Language), ordered=TRUE)

D = D[order(D$Language_Ordered),]

plot = ggplot(D %>% filter(Type == "Efficiency"), aes(x = Family, y = Language_Ordered)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") #+ facet_grid(~Family) 
ggsave(file="test1.pdf")


plot = ggplot(D %>% filter(Type == "Real Languages"), aes(x = Family, y = Language_Ordered)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") #+ facet_grid(~Family) 
ggsave(file="test2.pdf")

plot = ggplot(D %>% filter(Type == "Real Languages"), aes(x = Family, y = Language_Ordered)) + 
  geom_point(aes(fill=DirObj, colour = DirObj, size =1)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") #+ facet_grid(~Family) 
ggsave(file="test3.pdf")

plot = ggplot(D, aes(x = 1, y = Language_Ordered, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_text(size=9),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + facet_grid(~Type) 
ggsave(file="test4.pdf")






plot_orders_real = ggplot(D %>% filter(Type == "Real Languages"), aes(x = 1, y = Language_Ordered, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL)
plot_orders_eff = ggplot(D %>% filter(Type == "Efficiency"), aes(x = 1, y = Language_Ordered, group=CoarseDependency)) + 
  geom_point(aes(fill=DirB, colour = DirB, size =1), position = position_dodge(width=2.0)) +
#  scale_color_gradient() + #values=c("blue", "green")) +
  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),axis.ticks=element_blank(),
                     plot.title=element_text(size=11)) +
  theme(axis.title=element_blank()) + 
  theme(legend.position="none") + labs(x=NULL)
D$LanguageNumeric = as.numeric(D$Language_Ordered)
DLang = unique(D %>% select(Language_Ordered))
DFam = D %>% group_by(Family) %>% summarise(Start = min(Language_Ordered), End = max(Language_Ordered), Mean = mean(LanguageNumeric))
plot_langs = ggplot(DLang) 
plot_langs = plot_langs +  theme_bw() + theme(axis.text.x=element_blank(), #element_text(size=9, angle=0, vjust=0.3),
                     axis.text.y=element_blank(),
                     plot.title=element_text(size=11)) 
plot_langs = plot_langs + geom_text(aes(x=0.5 + 0.05, y=Language_Ordered, label=Language), hjust=0, size=3, colour="grey30")
plot_langs = plot_langs +      	theme(axis.title=element_blank()) 
plot_langs = plot_langs + xlim(-2.0, 2.0)
#plot_langs = plot_langs + geom_segment(aes(x=0, y=Language_Ordered, xend=1, yend=Language_Ordered)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=Start, xend=0.5, yend=Start)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=End, xend=0.5, yend=End)) 
plot_langs = plot_langs + geom_segment(data=DFam, aes(x=0, y=Start, xend=0, yend=End))
plot_langs = plot_langs + geom_text(data=DFam, aes(x=-0.1, y=Mean , label=Family), hjust=1, size=3, colour="grey30")
plot_langs = plot_langs + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
		    panel.background = element_blank(), axis.line = element_blank(),
                    plot.margin=unit(c(0,0,0.0,0), "mm"),
		    axis.ticks = element_blank()) + labs(x=NULL)
library("gridExtra")
grid.arrange(plot_langs, plot_orders_real, plot_orders_eff, nrow=1, widths=c(1, 1.2, 1.2))


