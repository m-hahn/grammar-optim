library(dplyr)
library(tidyr)



dataP = read.csv("adversarial-parser.tsv", sep="\t") %>% rename(Pars=Loss)
dataS = read.csv("adversarial-lm.tsv", sep="\t")

library(ggplot2)
#plot = ggplot(data=dataP, aes(x=UAS, y=1, group=Model, color=Model, fill = Model)) + 
#      geom_bar(stat="identity") + facet_wrap(~Type)



dataPO = read.csv("../../grammars/plane/plane-parse.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataPO2 = read.csv("../../grammars/plane/plane-parse-best.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataPO = rbind(dataPO, dataPO2)


relevant = dataP %>% group_by(Language, Model, FileName) %>% summarise()
relevant = merge(dataPO %>% rename(FileName=Model, Model=Type) %>% mutate(Type = "real"), relevant, by=c("Language", "Model", "FileName")) %>% mutate(Model = as.character(Model), Type = as.character(Type))
dataP = rbind(dataP, relevant)



library(forcats)
dataP = dataP %>% mutate(Model = fct_recode(Model, "Baseline" = "manual_output_funchead_RANDOM", "Optimized" = "manual_output_funchead_two_coarse_lambda09_best_balanced"))
dataP = dataP %>% filter(Model %in% c("Baseline", "Optimized"))
dataP = dataP %>% group_by(Language, Model, FileName, Type) %>% summarise(UAS = mean(UAS), Pars = mean(Pars))


plot = ggplot(data=dataP %>% filter(Language == "English"), aes(x=UAS, group=Model, color=Model, fill = Model)) + 
      geom_histogram(binwidth=0.01) + facet_wrap(~Type, ncol=3, scales = "free") + theme_bw()


ggsave(plot=plot, filename="adversarial-parse-english.pdf", height=6, width=10)



plot = ggplot(data=dataP %>% filter(Language == "Japanese"), aes(x=UAS, group=Model, color=Model, fill = Model)) + 
      geom_histogram(binwidth=0.01) + facet_wrap(~Type, ncol=3, scales = "free") + theme_bw()

ggsave(plot=plot, filename="adversarial-parse-japanese.pdf", height=6, width=10)


plot = ggplot(data=dataP %>% filter(Language == "English"), aes(x=Pars, group=Model, color=Model, fill = Model)) + 
      geom_histogram(binwidth=0.01) + facet_wrap(~Type, ncol=3, scales = "free") + theme_bw()


ggsave(plot=plot, filename="adversarial-parse-loss-english.pdf", height=6, width=10)



plot = ggplot(data=dataP %>% filter(Language == "Japanese"), aes(x=Pars, group=Model, color=Model, fill = Model)) + 
      geom_histogram(binwidth=0.01) + facet_wrap(~Type, ncol=3, scales = "free") + theme_bw()

ggsave(plot=plot, filename="adversarial-parse-loss-japanese.pdf", height=6, width=10)




#
#
#plot = ggplot(data=dataS, aes(x=LM_Loss, y=1, group=Model, color=Model, fill = Model)) + 
#      geom_bar(stat="identity") + facet_wrap(~Type)
#
#ggsave(plot=plot, filename="figures/adversarial-lm.pdf")
