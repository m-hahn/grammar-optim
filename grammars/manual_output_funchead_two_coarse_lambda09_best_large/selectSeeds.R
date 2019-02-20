library(dplyr)
library(tidyr)


seeds = read.csv("model-partition.tsv", sep="\t") %>% rename(Model=FileName)


data_cost = read.csv("../plane/plane-fixed-best-large.tsv", sep="\t")

data_parse = read.csv("../plane/plane-parse-best-large.tsv", sep="\t")


data = merge(seeds, data_cost, by=c("Language", "Model"))
data = merge(data, data_parse, by=c("Language", "Model", "Type"))

data = data %>% mutate(Efficiency = 0.9*Surp + Pars)

selectStrongestSeeds = data %>% group_by(Language, Group, Direction) %>% arrange(Efficiency) %>% slice(1)


write.csv(selectStrongestSeeds, file="successful-seeds.tsv")


strongestSeeds = data %>% group_by(Language, Direction) %>% arrange(Efficiency) %>% slice(1)


write.csv(strongestSeeds, file="most-successful-seeds.tsv")

