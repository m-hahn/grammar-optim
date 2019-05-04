library(tidyr)
library(dplyr)

data = read.csv("summary.tsv", sep="\t")

eff = data %>% filter(Type == "manual_output_funchead_two_coarse_lambda09_best_large") %>% group_by(Language) %>% summarise(BigramCE_eff = mean(BigramCE))
ground = data %>% filter(Type == "manual_output_funchead_ground_coarse_final") %>% group_by(Language) %>% summarise(BigramCE_ground = mean(BigramCE))
depl = data %>% filter(Type == "manual_output_funchead_coarse_depl") %>% group_by(Language) %>% summarise(BigramCE_depl = mean(BigramCE))
random = data %>% filter(grepl("RANDOM", Type))


data = merge(random, eff, by=c("Language"))
data = merge(data, ground, by=c("Language"))
data = merge(data, depl, by=c("Language")) 


dataSummary = data %>% group_by(Language) %>% summarise(betterThanEff = mean(BigramCE < BigramCE_eff), betterThanGround = mean(BigramCE < BigramCE_ground), betterThanDepl = mean(BigramCE < BigramCE_depl), meanCE = mean(BigramCE), sdCE = sd(BigramCE))

library("stats")

binom.test(1, 10, 0.5, alternative="less")

ps_eff = c()
ps_ground = c()
ps_eff_binom = c()
ps_ground_binom = c()
for(language in unique(data$Language)) {
  u = data %>% filter(Language == language)
  ps_eff = c(ps_eff, t.test(u$BigramCE, mu=mean(u$BigramCE_eff), alternative="greater")$p.value)
  ps_ground = c(ps_ground, t.test(u$BigramCE, mu=mean(u$BigramCE_ground), alternative="greater")$p.value)
  ps_eff_binom = c(ps_eff_binom, binom.test(sum(u$BigramCE_eff > u$BigramCE), nrow(u), 0.5, alternative="less")$p.value)
  ps_ground_binom = c(ps_ground_binom, binom.test(sum(u$BigramCE_ground > u$BigramCE), nrow(u), 0.5, alternative="less")$p.value)
}

ps_eff
ps_ground

#
mean(ps_eff < 0.05)
#
mean(ps_ground < 0.05)

# Robustness Check
mean(ps_eff_binom < 0.05)
mean(ps_ground_binom < 0.05)


library(ggplot2)
plot = ggplot(data, aes(x=BigramCE, group=Model)) + geom_histogram(binwidth=0.01, color="blue") + geom_histogram(aes(x=BigramCE_ground), color="red") + facet_wrap(~Language, ncol=10, scales="free") + theme_bw()


data = merge(data, dataSummary, by=c("Language"))
data = data %>% mutate(CE_z = (BigramCE-meanCE)/sdCE)
ground = merge(ground, dataSummary, by=c("Language"))
ground = ground %>% mutate(CE_z_ground = (BigramCE_ground-meanCE)/sdCE)

plot = ggplot(data, aes(x=CE_z)) + geom_histogram(binwidth=0.01, color="blue") + geom_histogram(binwidth=0.01, data=ground, aes(x=CE_z_ground), color="red") + theme_bw()





