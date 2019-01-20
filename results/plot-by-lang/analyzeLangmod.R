




library(lme4)

library(tidyr)
library(dplyr)
library(ggplot2)



dataS = read.csv("../../grammars/plane/plane-fixed.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS2 = read.csv("../../grammars/plane/plane-fixed-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = rbind(dataS, dataS2)

dataP = read.csv("../../grammars/plane/plane-parse.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP2 = read.csv("../../grammars/plane/plane-parse-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = rbind(dataP, dataP2)


bestLangmod = read.csv("../strongest_models/best-langmod-best-balanced.csv")$Model
bestParsing = read.csv("../strongest_models/best-parse-best-balanced.csv")$Model
bestEfficiency = read.csv("../strongest_models/best-two-lambda09-best-balanced.csv")$Model
bestDepl =  read.csv("../strongest_models/best-depl.csv")$Model
bestGround = read.csv("../strongest_models/models-mle.csv")$Model

#dataS = dataS %>% filter(Model %in% c("REAL_REAL", bestLangmod, bestParsing, bestEfficiency, bestDepl, bestGround) | Type %in% c("manual_output_funchead_RANDOM"))
#dataP = dataP %>% filter(Model %in% c("REAL_REAL", bestLangmod, bestParsing, bestEfficiency, bestDepl, bestGround) | Type %in% c("manual_output_funchead_RANDOM"))

dataS = dataS %>% filter(Model %in% c(bestLangmod, bestParsing, bestEfficiency, bestDepl, bestGround) | Type %in% c("manual_output_funchead_RANDOM"))
dataP = dataP %>% filter(Model %in% c(bestLangmod, bestParsing, bestEfficiency, bestDepl, bestGround) | Type %in% c("manual_output_funchead_RANDOM"))


data = merge(dataS, dataP, by=c("Language", "Model", "Type"))

library(forcats)

data = data %>% mutate(TypeN = fct_recode(Type, "Parseability" = "manual_output_funchead_two_coarse_parser_best_balanced", "Baseline Languages" = "manual_output_funchead_RANDOM", "Efficiency" = "manual_output_funchead_two_coarse_lambda09_best_balanced", "Dependency Length" = "manual_output_funchead_coarse_depl", "Predictability" = "manual_output_funchead_langmod_coarse_best_balanced", "Real Languages" = "manual_output_funchead_ground_coarse_final"))



  dodge = position_dodge(width = .9)
  dataL = data %>% filter(Language %in% c("Japanese", "English"))
  dataBars = dataL %>% filter(TypeN != "Baseline Languages") %>% group_by(Language, TypeN) %>% summarise(y=1, Surp=mean(Surp))
  widthOfBars = ifelse(dataBars$Language == "English", 0.006, 0.008)   #0.01 #(max(agr$Surp) - min(agr$Surp))/50

  plot1 = ggplot(dataL %>% filter(TypeN == "Baseline Languages"), aes(x=Surp,fill=TypeN)) +
    geom_density(aes(x=Surp, y=..scaled.., fill=TypeN), adjust=3) +
    geom_bar(stat="identity", width=widthOfBars, data=dataBars, aes(x=Surp, y=y, fill=TypeN, group=TypeN), position = position_dodge()) +
    theme_classic() +
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          axis.line.y=element_blank()) + facet_grid(~Language, scales="free") + scale_fill_brewer(palette="Set2")

  filename = paste("langmod-optimized-coarse.pdf",sep="")
  ggsave(filename=filename, plot=plot1, width=12, height=3)
  cat(filename,"\n")



  dodge = position_dodge(width = .9)
  dataL = data %>% filter(Language %in% c("Japanese", "English"))
  dataBars = dataL %>% filter(TypeN != "Baseline Languages") %>% group_by(Language, TypeN) %>% summarise(y=1, Pars=mean(Pars))
  widthOfBars = ifelse(dataBars$Language == "English", 0.006, 0.008)   #0.01 #(max(agr$Surp) - min(agr$Surp))/50

  plot2 = ggplot(dataL %>% filter(TypeN == "Baseline Languages"), aes(x=Pars,fill=TypeN)) +
    geom_density(aes(x=Pars, y=..scaled.., fill=TypeN), adjust=1) +
    geom_bar(stat="identity", width=widthOfBars, data=dataL %>% filter(TypeN != "Baseline Languages") %>% group_by(TypeN) %>% summarise(y=1, Pars=mean(Pars)), aes(x=Pars, y=y, fill=TypeN, group=TypeN)) +
    theme_classic() +
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          axis.line.y=element_blank()) + facet_grid(~Language, scales="free") + scale_fill_brewer(palette="Set2")

  filename = paste("parsing-optimized-coarse.pdf",sep="")
  ggsave(filename=filename, plot=plot2, width=12, height=3)
  cat(filename,"\n")





library(gridExtra)


#extract legend
#https://github.com/hadley/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}


mylegend = g_legend(plot1)


p3 <- grid.arrange(arrangeGrob(plot1 + theme(legend.position="none"),
                         plot2 + theme(legend.position="none"),
                         nrow=2),
             mylegend, ncol=2,widths=c(10, 1))



  filename = paste("grid-optimized.pdf",sep="")
  ggsave(filename=filename, plot=p3, width=16, height=3.6)







#
#
#createPlot = function(LanguageName) {
#  dataL = data %>% filter(Language == LanguageName)
#  dodge = position_dodge(width = .9)
#  widthOfBars = 0.001 #(max(agr$Surp) - min(agr$Surp))/50
#  plot = ggplot(dataL %>% filter(TypeN == "Baseline Languages"), aes(x=Surp,fill=TypeN)) +
#    geom_density(aes(x=Surp, y=..scaled.., fill=TypeN), adjust=4) +
#    geom_bar(stat="identity", width=widthOfBars, data=dataL %>% filter(TypeN != "Baseline Languages") %>% group_by(TypeN) %>% summarise(y=1, Surp=mean(Surp)), aes(x=Surp, y=y, fill=TypeN, group=TypeN)) +
#    theme_classic() +
#    theme(axis.title.y=element_blank(),
#          axis.text.y=element_blank(),
#          axis.ticks.y=element_blank(),
#          axis.line.y=element_blank()) +
#     xlab(LanguageName)
#  filename = paste("langmod-optimized-coarse-",LanguageName,".pdf",sep="")
#  ggsave(filename=filename, plot=plot, width=12, height=3)
#  cat(filename,"\n")
#}
#
#createPlot("English")
#createPlot("Japanese")
#
#createPlot = function(LanguageName) {
#  dataL = data %>% filter(Language == LanguageName)
#  dodge = position_dodge(width = .9)
#  widthOfBars = (max(dataL$UAS) - min(dataL$UAS))/100
#  plot = ggplot(dataL %>% filter(TypeN == "Baseline Languages"), aes(x=UAS,fill=TypeN)) +
#    geom_density(aes(x=UAS, y=..scaled.., fill=TypeN), adjust=1) +
#    geom_bar(stat="identity", width=widthOfBars, data=dataL %>% filter(TypeN != "Baseline Languages") %>% group_by(TypeN) %>% summarise(y=1, UAS=mean(UAS)), aes(x=UAS, y=y, fill=TypeN, group=TypeN)) +
#    theme_classic() +
#    theme(axis.title.y=element_blank(),
#          axis.text.y=element_blank(),
#          axis.ticks.y=element_blank(),
#          axis.line.y=element_blank()) +
#     xlab(LanguageName)
#  filename = paste("parsing-optimized-coarse-",LanguageName,".pdf",sep="")
#  ggsave(filename=filename, plot=plot, width=12, height=3)
#  cat(filename,"\n")
#}
#createPlot("English")
#createPlot("Japanese")
#
#
#createPlot = function(LanguageName) {
#  dataL = data %>% filter(Language == LanguageName)
#  dodge = position_dodge(width = .9)
#  widthOfBars = (max(data$LogLoss) - min(data$LogLoss))/50
#  plot = ggplot(dataL %>% filter(Type == "RANDOM"), aes(x=LogLoss,fill=Type)) +
#    geom_density(aes(x=LogLoss, y=..scaled.., fill=Type)) +
#    geom_bar(stat="identity", width=widthOfBars, data=dataL %>% filter(Type != "RANDOM") %>% group_by(Type) %>% summarise(y=1, LogLoss=mean(LogLoss)), aes(x=LogLoss, y=y, fill=Type, group=Type)) +
#    theme_classic() +
#    theme(axis.title.y=element_blank(),
#          axis.text.y=element_blank(),
#          axis.ticks.y=element_blank(),
#          axis.line.y=element_blank()) +
#     xlab(LanguageName)
#  filename = paste("parsing-optimized-logloss-",LanguageName,".pdf",sep="")
#  ggsave(filename=filename, plot=plot, width=6, height=1.5)
#}
#createPlot("English")
#createPlot("Japanese")
#


