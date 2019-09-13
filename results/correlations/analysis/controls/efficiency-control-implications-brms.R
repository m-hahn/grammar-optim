# Bayesian regression analysis for each UD relation

data = read.csv("../../../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/auto-summary-lstm.tsv", sep="\t")# %>% rename(Quality=AverageLength)
best = read.csv("../../../../grammars/manual_output_funchead_two_coarse_lambda09_best_large/successful-seeds.tsv")

library(forcats)
library(dplyr)
library(tidyr)
library(ggplot2)

data = data %>% mutate(Language = fct_recode(Language, "Ancient_Greek" = "Ancient", "Old_Church_Slavonic" = "Old"))
data = merge(data %>% mutate(FileName = as.character(FileName)), best %>% rename(FileName = Model), by=c("Language", "FileName"))

dryer_greenberg_fine = data

languages = read.csv("../../../languages/languages-iso_codes.tsv")
dryer_greenberg_fine  = merge(dryer_greenberg_fine, languages, by=c("Language"), all.x=TRUE)

library("brms")

dependency = "nmod"

getCorrPair = function(dependency1, dependency2) {
   corr_pair = dryer_greenberg_fine %>% filter((CoarseDependency == dependency1) | (CoarseDependency == dependency2))
   corr_pair = unique(corr_pair %>% select(Family, Language, FileName, CoarseDependency, DH_Weight )) %>% spread(CoarseDependency, DH_Weight)
   corr_pair$correlator1 = corr_pair[[dependency1]]
   corr_pair$correlator2 = corr_pair[[dependency2]]
   corr_pair = corr_pair %>% mutate(correlator1_s = pmax(0,sign(correlator1)), correlator2_s=pmax(0,sign(correlator2)))

   corr_pair = corr_pair %>% mutate(correlator1_s = ifelse(correlator1 == 0, NA, correlator1_s))
   corr_pair = corr_pair %>% mutate(correlator2_s = ifelse(correlator2 == 0, NA, correlator2_s)) 

   corr_pair$agree = (corr_pair$correlator1_s == corr_pair$correlator2_s)
   return(corr_pair)
}

corr_pair = getCorrPair("nmod", "lifted_case")

model3 = brm(agree ~ (1|Family) + (1|Language), family="bernoulli", data=corr_pair)
samples = posterior_samples(model3, "b_Intercept")[,]
posteriorOpposite = ecdf(samples)(0.0)

outpath = "output/results-prevalence-two-09-large-implications.tsv"
sink(outpath)
cat("")
sink()

cat(paste("dependency1", "dependency2", "satisfiedFraction", "posteriorMean", "posteriorSD", "posteriorOpposite", sep="\t"), file=outpath, append=TRUE, sep="\n")


runAnalysis = function(dependency1, dependency2) {
   corr_pair = getCorrPair(dependency1, dependency2)
   model3 = update(model3, newdata=corr_pair, iter=5000) # potentially add , control=list(adapt_delta=0.9)
   summary(model3)
   
   samples = posterior_samples(model3, "b_Intercept")[,]
   posteriorOpposite = ecdf(samples)(0.0)
   posteriorMean = mean(samples)
   posteriorSD = sd(samples)
   satisfiedFraction = mean((corr_pair$correlator1_s == corr_pair$correlator2_s), na.rm=TRUE)
   cat(paste(dependency1, dependency2, satisfiedFraction, posteriorMean, posteriorSD, posteriorOpposite, sep="\t"), file=outpath, append=TRUE, sep="\n")
}

runAnalysis("nmod", "lifted_case")
runAnalysis("acl", "lifted_case")

sink()
sink()



