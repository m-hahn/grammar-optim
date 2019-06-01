# Per-relation analysis

data = read.csv("../../grammars/manual_output_funchead_coarse_depl/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)
data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))
languages = read.csv("../languages/languages-iso_codes.tsv")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
library("brms")
dependency = "nmod"
dependencies = c("acl", "advcl", "advmod", "amod", "appos", "aux", "ccomp", "compound", "conj", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "lifted_case", "lifted_cc", "lifted_cop", "lifted_mark", "list", "nmod", "nsubj", "nummod", "obl", "orphan", "parataxis", "reparandum", "vocative", "xcomp")
objs = data %>% filter(Dependency == "obj") %>% mutate(obj = pmax(0, sign(DH_Weight))) %>% select(obj, Language, FileName)
data = data %>% filter((Dependency %in% dependencies))
data = merge(data, objs, by=c("Language", "FileName"))
data = data %>% mutate(dir = pmax(0, sign(DH_Weight)))
data = data %>% mutate(dir = ifelse(DH_Weight == 0, NA, dir))
data$agree = (data$dir == data$obj)



library(ggplot2)


type = "depl"

u = data %>% filter(Dependency == "nmod")
model3 = brm(agree ~ (1|p|Family) + (1|q|Language), family="bernoulli", data=u, iter=100)


for(dependency in dependencies) {
   u = data %>% filter(Dependency == dependency)
   model3 = update(model3, newdata=u, iter=5000)
   u = posterior_samples(model3)
   write.csv(u, file=paste("~/CS_SCR/posteriors/posterior-", dependency, "-", type, "-large.csv", sep=""))
}



