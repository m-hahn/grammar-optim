data = read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
best = read.csv("../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/successful-seeds.tsv")

library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)
data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))

data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))

languages = read.csv("../languages/languages-iso_codes.tsv")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)
library("brms")
dependency = "nmod"
dependencies = c("acl",  "aux", "lifted_case", "lifted_cop", "lifted_mark", "nmod", "obl", "xcomp") 
objs = data %>% filter(CoarseDependency == "obj") %>% mutate(obj = pmax(0, sign(DH_Weight))) %>% select(obj, Language, FileName)
data = data %>% filter((CoarseDependency %in% dependencies))
data = merge(data, objs, by=c("Language", "FileName"))
data = data %>% mutate(dir = pmax(0, sign(DH_Weight)))
data = data %>% mutate(dir = ifelse(DH_Weight == 0, NA, dir))
data$agree = (data$dir == data$obj)
data = unique(data %>% select(Family, Language, FileName, CoarseDependency, agree)) %>% spread(CoarseDependency, agree)

model3 = brm(cbind(acl, aux, lifted_case, lifted_cop, lifted_mark, nmod, obl, xcomp) ~ (1|p|Family) + (1|q|Language), family="bernoulli", data=data, iter=5000)



u = posterior_samples(model3) 
mean(u$b_acl_Intercept < 0 |  u$b_aux_Intercept > 0  | u$b_liftedcase_Intercept < 0  | u$b_liftedcop_Intercept < 0  | u$b_liftedmark_Intercept < 0  | u$b_nmod_Intercept < 0  | u$b_obl_Intercept < 0  | u$b_xcomp_Intercept < 0 )

satisfied = 8 - ((u$b_acl_Intercept < 0) + (u$b_aux_Intercept > 0 ) + (u$b_liftedcase_Intercept < 0 ) + (u$b_liftedcop_Intercept < 0 ) + (u$b_liftedmark_Intercept < 0 ) + (u$b_nmod_Intercept < 0 ) + (u$b_obl_Intercept < 0 ) + (u$b_xcomp_Intercept < 0 ))

library(ggplot2)

plot = ggplot(data = data.frame(satisfied=satisfied), aes(x=satisfied)) + geom_histogram() + theme_bw() + xlim(1,10.5)


ggsave(plot=plot, filename="figures/posterior-satisfied-universals-efficiency-large.pdf")


write.csv(u, file="~/CS_SCR/posteriors/posterior-10-efficiency-large.csv")




