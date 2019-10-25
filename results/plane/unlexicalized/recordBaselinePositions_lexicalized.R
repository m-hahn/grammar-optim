library(tidyr)
library(dplyr)
library(ggplot2)

# Read predictability estimates
dataS =  read.csv("../../../grammars/plane/controls/plane-fixed-withoutPOS.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataS = dataS %>% group_by(Language, Type, Model) %>% summarise(Surprisal = mean(Surp, na.rm=TRUE))
dataS = as.data.frame(dataS)

# Read parseability estimates
dataP =  read.csv("../../../grammars/plane/controls/plane-parse-withoutPOS.tsv", sep="\t") %>% mutate(Model = as.character(Model))
dataP = dataP %>% group_by(Language, Type, Model) %>% summarise(UAS = mean(UAS, na.rm=TRUE), Pars = mean(Pars, na.rm=TRUE))
dataP = as.data.frame(dataP)

# Ensure everything is treated as a string, not a factor, to ensure the frames can be merged
dataS = dataS %>% mutate(Type = as.character(Type))
dataP = dataP %>% mutate(Type = as.character(Type))
dataS = dataS %>% mutate(Model = as.character(Model))
dataP = dataP %>% mutate(Model = as.character(Model))
dataS = dataS %>% mutate(Language = as.character(Language))
dataP = dataP %>% mutate(Language = as.character(Language))
data = merge(dataS, dataP, by=c("Language", "Model", "Type"), all.x=TRUE, all.y=TRUE)

# Collapse all five groups of random grammars (10 random grammars per group per language)
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM2", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM3", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM4", "manual_output_funchead_RANDOM", as.character(Type)))
data = data %>% mutate(Type = ifelse(Type == "manual_output_funchead_RANDOM5", "manual_output_funchead_RANDOM", as.character(Type)))


dataBaseline = data %>% filter(Type == "manual_output_funchead_RANDOM")
dataGround = data %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% select(Language, Surprisal, Pars) %>% rename(SurprisalGround = Surprisal) %>% rename(ParsGround = Pars) %>% mutate(EffGround = ParsGround + 0.9*SurprisalGround) %>% group_by(Language)
data = merge(dataBaseline, dataGround, by=c("Language"))


write.csv(data, file="analyze_pareto_optimality/pareto-data.tsv")


