# This script evaluates, for each language, whether the grammar optimized for efficiency (i.e., the one among the 8 optimized ones with the highest efficiency) has significantly lower average dependency length than the mean of the baseline grammar distribution.

library(lme4)
library(tidyr)
library(dplyr)
library(ggplot2)


deplen = read.csv("../../grammars/dependency_length/total_summary_funchead_coarse.tsv", sep="\t")# %>% rename(Quality=AverageLength)


deplen = deplen %>% filter(Type %in% c("manual_output_funchead_coarse_depl", "manual_output_funchead_langmod_coarse_best_balanced","manual_output_funchead_RANDOM","manual_output_funchead_two_coarse_lambda09_best_balanced","manual_output_funchead_two_coarse_parser_best_balanced", "REAL_REAL"))

deplen$Temperature = NULL
deplen$Counter = NULL
deplen$OriginalLoss = NULL
deplen$OriginalCounter = NULL


bestDepL = read.csv("../strongest_models/best-depl.csv") %>% select(Language, Type, Model)
bestLangmod = read.csv("../strongest_models/best-langmod-best-balanced.csv") %>% select(Language, Type, Model)
bestParse = read.csv("../strongest_models/best-parse-best-balanced.csv") %>% select(Language, Type, Model)
bestEff = read.csv("../strongest_models/best-two-lambda09-best-balanced.csv") %>% select(Language, Type, Model)

bestModels = rbind(bestDepL, bestLangmod, bestParse, bestEff)

deplenRand = deplen %>% filter(Type == "manual_output_funchead_RANDOM")
deplenReal = deplen %>% filter(Type == "REAL_REAL")

deplen = merge(deplen, bestModels, by=c("Language", "Model", "Type"))
deplen = rbind(deplen, deplenRand, deplenReal)





eff = deplen %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_balanced") %>% group_by(Language) %>% summarise(AverageLengthPerWord_eff = mean(AverageLengthPerWord))
ground = deplen %>% filter(Type == "REAL_REAL") %>% group_by(Language) %>% summarise(AverageLengthPerWord_ground = mean(AverageLengthPerWord))
depl = deplen %>% filter(Type == "manual_output_funchead_coarse_depl") %>% group_by(Language) %>% summarise(AverageLengthPerWord_depl = mean(AverageLengthPerWord))
random = deplen %>% filter(grepl("RANDOM", Type))


data = merge(random, eff, by=c("Language"))
data = merge(data, ground, by=c("Language"))
data = merge(data, depl, by=c("Language")) 


dataSummary = data %>% group_by(Language) %>% summarise(betterThanEff = mean(AverageLengthPerWord < AverageLengthPerWord_eff), betterThanGround = mean(AverageLengthPerWord < AverageLengthPerWord_ground), betterThanDepl = mean(AverageLengthPerWord < AverageLengthPerWord_depl))






dataNoRand = merge(eff, ground, by=c("Language"))
dataNoRand = merge(dataNoRand, depl, by=c("Language")) 
random2 = deplen %>% filter(grepl("RANDOM", Type)) %>% group_by(Language) %>% summarise(AverageLengthPerWord_random = mean(AverageLengthPerWord))
dataNoRand = merge(dataNoRand, random2, by=c("Language")) 



library("stats")

u = data

ps_eff = c()
ps_ground = c()
ps_eff_binom = c()
ps_ground_binom = c()
for(language in unique(data$Language)) {
  u = data %>% filter(Language == language)
  ps_eff = c(ps_eff, t.test(u$AverageLengthPerWord, mu=mean(u$AverageLengthPerWord_eff), alternative="greater")$p.value)
  ps_ground = c(ps_ground, t.test(u$AverageLengthPerWord, mu=mean(u$AverageLengthPerWord_ground), alternative="greater")$p.value)
  ps_eff_binom = c(ps_eff_binom, binom.test(sum(u$AverageLengthPerWord_eff > u$AverageLengthPerWord), nrow(u), 0.5, alternative="less")$p.value)
  ps_ground_binom = c(ps_ground_binom, binom.test(sum(u$AverageLengthPerWord_ground > u$AverageLengthPerWord), nrow(u), 0.5, alternative="less")$p.value)
}

ps_eff
ps_ground

# This result is referred to in the Discussin section of the main paper
mean(ps_eff < 0.05)
# Verifying that real orders also improve over baselines
mean(ps_ground < 0.05)

# Robustness Check using Binomial Test
mean(ps_eff_binom < 0.05)
mean(ps_ground_binom < 0.05)






