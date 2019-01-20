#data = read.csv("CS_SCR/deps/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(tidyr)
library(dplyr)

#data = data %>% filter(grepl("FuncHead", ModelName)) %>% filter(grepl("Coarse", ModelName))
##dataModels = data %>% filter(!(Model %in% c("RANDOM", "REAL")))
#modelData = read.csv("CS_SCR/deps//models_log_summary.tsv", sep="\t")
#data = merge(data, modelData %>% select(FileName, Command, Model) %>% rename(ModelOriginal=Model, Model=FileName), by=c("Model"), all.x=TRUE)
#data = data[order(data$AverageLengthPerWord),]
#data = data %>% mutate(LSTM = grepl("Running", ModelOriginal), DepL = grepl("DepL", ModelOriginal), Parsing = grepl("Reinf", ModelOriginal), Real = (Model == "REAL"), Random = (Model == "RANDOM") )
#data = data %>% mutate(type = case_when(LSTM ~ "LSTM", DepL ~ "DepL", Parsing ~ "Parsing", Real ~ "Real", Random ~ "Random", TRUE ~ "OTHER"))
#data = data %>% filter(Real | Random | Temperature == Inf)
#bestLSTM = data %>% filter(LSTM | Parsing) %>% group_by(Language, type) %>% summarise(BestOriginalLoss = min(OriginalLoss))
#data = merge(data, bestLSTM, by=c("Language", "type"), all=TRUE)
#data = data %>% filter(Real | Random | DepL | OriginalLoss == BestOriginalLoss)
#data = data[order(data$AverageLengthPerWord),]
#perLanguage = data %>% group_by(Language, type) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord))
#perLanguageSpread = perLanguage %>% spread(type, AverageLengthPerWord)

data = data.frame()

for(LanguageName in c("English", "Chinese", "Hindi", "Japanese")) {
   LanguageName = "English"
   
   relevantModels =  unique(read.csv(paste("/home/user/CS_SCR/deps/dependency_length/", LanguageName, "_forVisualization.tsv", sep=""), sep="\t"))
   
   for(i in (1:nrow(relevantModels))) {
      cat(i,nrow(relevantModels),"\n")
      data2 = read.csv(paste("/home/user/CS_SCR/deps/dependency_length/", LanguageName, "_computeDependencyLengthsByType_NEWPYTORCH_Deterministic_FuncHead_Coarse.py_model_",(relevantModels$Model[[i]]),".tsv", sep=""), sep="\t")
      data2 = data2 %>% group_by(SentenceNumber) %>% summarise(SentenceLength = NROW(Length), DepLength = sum(Length))
      data2$Type = relevantModels$Type[[i]]
      data2$Language = LanguageName
      data = rbind(data, data2)
   }
}

library(ggplot2)

plot = ggplot(data, aes(x=SentenceLength,y=DepLength,color=Type)) +
  geom_smooth(method = 'auto') +
  xlab("Sentence Length") +
  ylab("Dependency Length") +
       scale_color_brewer(palette="Set1") +
  theme_classic() + ylim(0,220) + xlim(0,50) +
  facet_wrap(~ Language, ncol=4) 



ggsave(plot=plot, filename="figures/depLength-facet.pdf", height=2.5, width=10)





