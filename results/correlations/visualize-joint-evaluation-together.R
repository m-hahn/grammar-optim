
library(dplyr)
library(tidyr)


u_depl = read.csv("/home/user/CS_SCR/posteriors/posterior-10-depl.csv") %>% mutate(Type = "Dependency Length")
u_langmod = read.csv("/home/user/CS_SCR/posteriors/posterior-10-langmod.csv") %>% mutate(Type = "Predictability")
u_parser = read.csv("/home/user/CS_SCR/posteriors/posterior-10-parseability.csv") %>% mutate(Type = "Parseability")
u_efficiency = read.csv("/home/user/CS_SCR/posteriors/posterior-10-efficiency.csv") %>% mutate(Type = "Efficiency")


u = rbind(u_depl, u_langmod, u_parser, u_efficiency)

u$satisfied = 10 - ((u$b_acl_Intercept < 0) + (u$b_advmod_Intercept > 0)  + (u$b_aux_Intercept > 0 ) + (u$b_liftedcase_Intercept < 0 ) + (u$b_liftedcop_Intercept < 0 ) + (u$b_liftedmark_Intercept < 0 ) + (u$b_nmod_Intercept < 0 ) + (u$b_nsubj_Intercept > 0 ) + (u$b_obl_Intercept < 0 ) + (u$b_xcomp_Intercept < 0 ))

library(ggplot2)

v = u %>% group_by(Type) %>% summarise(SamplesNum = NROW(satisfied))
u = merge(u, v, by=c("Type"))
data2 = u %>% group_by(Type, satisfied) %>% summarise(SamplesNum = mean(SamplesNum), posterior = NROW(satisfied)) %>% mutate(posteriorProb = posterior/SamplesNum)
u = NULL

#plot = ggplot(data = data, aes(x=satisfiedCount)) + geom_histogram() + theme_bw() + xlim(0,10.5) + ggtitle("Dependency Length") + xlab("Number of Predicted Correlations") + ylab("Number of Posterior Samples")

plot = ggplot(data = data2, aes(x=satisfied, y=posterior)) + geom_bar(stat="identity") + theme_bw() + xlim(0,10.5) + ggtitle("Dependency Length") + xlab("Number of Predicted Correlations") + ylab("Number of Posterior Samples") + geom_text(aes(label=posteriorProb), vjust=0, size=4.3) + ylim(0, 1.1*nrow(data)) +
    theme(text = element_text(size=20),
        axis.text.x = element_text(angle=90, hjust=1)) + facet_grid(~Type)

ggsave(plot=plot, filename="figures/posterior-satisfied-universals-together.pdf")



