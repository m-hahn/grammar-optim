



library(lme4)

library(tidyr)
library(dplyr)
library(ggplot2)




depl = read.csv("/home/user/CS_SCR/deps/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(tidyr)
library(dplyr)


depl = depl %>% filter(Type %in% c("manual_output_funchead_coarse_depl", "manual_output_funchead_langmod_coarse_best_balanced","manual_output_funchead_RANDOM","manual_output_funchead_two_coarse_lambda09_best_balanced","manual_output_funchead_two_coarse_parser_best_balanced", "REAL_REAL"))

depl$Temperature = NULL
depl$Counter = NULL
depl$OriginalLoss = NULL
depl$OriginalCounter = NULL


bestDepL = read.csv("../strongest_models/best-depl.csv") %>% select(Language, Type, Model)
bestLangmod = read.csv("../strongest_models/best-langmod-best-balanced.csv") %>% select(Language, Type, Model)
bestParse = read.csv("../strongest_models/best-parse-best-balanced.csv") %>% select(Language, Type, Model)
bestEff = read.csv("../strongest_models/best-two-lambda09-best-balanced.csv") %>% select(Language, Type, Model)

bestModels = rbind(bestDepL, bestLangmod, bestParse, bestEff)

deplRand = depl %>% filter(Type == "manual_output_funchead_RANDOM")
deplReal = depl %>% filter(Type == "REAL_REAL")

depl = merge(depl, bestModels, by=c("Language", "Model", "Type"))
depl = rbind(depl, deplRand, deplReal)

depl2 = depl %>% group_by(Language, Type) %>% summarise(AverageLengthPerWord = mean(AverageLengthPerWord)) %>% spread(Type, AverageLengthPerWord)

mean(depl2$REAL_REAL < depl2$manual_output_funchead_coarse_depl)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_langmod_coarse_best_balanced)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_RANDOM, na.rm=TRUE)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_two_coarse_parser_best_balanced, na.rm=TRUE)
mean(depl2$REAL_REAL < depl2$manual_output_funchead_two_coarse_lambda09_best_balanced, na.rm=TRUE)


