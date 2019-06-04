library(tidyr)
library(dplyr)

data = data.frame()

for(LanguageName in c("English", "Chinese", "Hindi", "Japanese")) {
      data2 = read.csv(paste("/home/user/CS_SCR/deps/dependency_length/", LanguageName, "_computeDependencyLengths_Real.py_model_REAL_REAL.tsv", sep=""), sep="\t")
      data2 = data2 %>% group_by(SentenceNumber) %>% summarise(SentenceLength = NROW(Length), DepLength = sum(Length))
      data2$Type = "REAL_REAL"
      data2$Language = LanguageName
      data = rbind(data, data2)

   relevantModels =  unique(read.csv(paste("forVisualization/", LanguageName, "_forVisualization.tsv", sep=""), sep="\t"))
   for(i in (1:nrow(relevantModels))) {
      cat(i,nrow(relevantModels),"\n")
      data2 = read.csv(paste("/home/user/CS_SCR/deps/dependency_length/", LanguageName, "_computeDependencyLengths_ForGrammars.py_model_",(relevantModels$Model[[i]]),".tsv", sep=""), sep="\t")
      data2 = data2 %>% group_by(SentenceNumber) %>% summarise(SentenceLength = NROW(Length), DepLength = sum(Length))
      data2$Type = relevantModels$Type[[i]]
      data2$Language = LanguageName
      data = rbind(data, data2)
   }
}


data = data %>% mutate(TypeN = case_when(Type == "manual_output_funchead_RANDOM" ~ "Baseline",
					 Type == "manual_output_funchead_coarse_depl" ~ "Optimized for Dep.L.",
					 Type == "manual_output_funchead_langmod_coarse_best_balanced" ~ "Optimized for Pred.",
					 Type == "manual_output_funchead_two_coarse_parser_best_balanced" ~ "Optimized for Pars.",
					 Type == "manual_output_funchead_two_coarse_lambda09_best_balanced" ~ "Optimized for Eff.",
					 Type == "manual_output_funchead_ground_coarse_final" ~ "MLE",
					 Type == "REAL_REAL" ~ "Actual Language"))

write.csv(data, file="/home/user/CS_SCR/dependencyLength-intermediate.csv")


