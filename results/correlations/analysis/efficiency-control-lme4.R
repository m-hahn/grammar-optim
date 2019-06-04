# Control using frequentist regression model

data = read.csv("../../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/auto-summary-lstm.tsv", sep="\t")
best = read.csv("../../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/successful-seeds.tsv")

library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)
data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))

data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))

languages = read.csv("../../languages/languages-iso_codes.tsv")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
dependency = "nmod"
dependencies = c("acl", "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp")
objs = data %>% filter(CoarseDependency == "obj") %>% mutate(obj = pmax(0, sign(DH_Weight))) %>% select(obj, Language, FileName)
data = data %>% filter((CoarseDependency %in% dependencies))
data = merge(data, objs, by=c("Language", "FileName"))
data = data %>% mutate(dir = pmax(0, sign(DH_Weight)))
data = data %>% mutate(dir = ifelse(DH_Weight == 0, NA, dir))
data$agree = (data$dir == data$obj)
data = unique(data %>% select(Family, Language, FileName, CoarseDependency, agree)) %>% spread(CoarseDependency, agree)


library(lme4)

sink("../analysis_frequentist/efficiency-results-lme4.tsv")
for(dependency in dependencies) {
     model = summary(glmer(paste(dependency, " ~ (1|Family) + (1|Language)"), family="binomial", data=data))
     p = (coef(model)[4])
     beta = coef(model)[1]
     se = coef(model)[2]
     z = coef(model)[3]
     cat(dependency,  beta, se, z, p, "\n", sep="\t")
}
sink()



