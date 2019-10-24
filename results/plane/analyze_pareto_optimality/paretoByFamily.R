data = read.csv("pareto-data.tsv")

languages = read.csv("../../languages/languages-iso_codes.tsv")
data  = merge(data, languages, by=c("Language"), all.x=TRUE)

library(dplyr)
library(tidyr)
library(lme4)
library(ggplot2)

lambda = 0.9

data = data %>% mutate(Better = (Pars + lambda*Surprisal) < (ParsGround + lambda*SurprisalGround))

library(brms)

model = brm(Better ~ (1|Family) + (1|Language), family="bernoulli", data=data)




samples = posterior_samples(model, "b_Intercept")[,]
posteriorOpposite = ecdf(samples)(0.0)

#for(family in unique(languages$Family)) {
#
#   samples_ = (samples + posterior_samples(model)[[paste("r_Family[",sub(" ", ".", family),",Intercept]", sep="")]])
#
#   plot = ggplot(data.frame(BetterThan = 1/(1+exp(samples_))), aes(x=BetterThan, y=..scaled..)) + geom_density(fill="blue")
#   plot = plot + xlim(0,1)
#   plot = plot + theme_classic()
#   plot = plot + theme_void()
#   plot = plot + theme(legend.position="none")
#   plot = plot + geom_segment(aes(x=0.5, xend=0.5, y=0, yend=1), linetype=2)
##   plot = plot + theme(axis.line.x = element_line(colour = "black"))
#   ggsave(paste("figures/quantile_posterior_", lambda, "_", family, ".pdf", sep=""), plot=plot, height=1, width=2)
#
#
#
#}

#lambda = 0.9
summary(model)

# Only Surprisal
data = data %>% mutate(Better = (Surprisal) < (SurprisalGround))
model = update(model, newdata=data)
summary(model)

# Only Pars
data = data %>% mutate(Better = (Pars) < (ParsGround))
model = update(model, newdata=data)
summary(model)

