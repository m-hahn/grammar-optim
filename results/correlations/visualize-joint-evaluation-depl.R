u = read.csv("/home/user/CS_SCR/posteriors/posterior-10-depl.csv")

satisfied = 10 - ((u$b_acl_Intercept < 0) + (u$b_advmod_Intercept > 0)  + (u$b_aux_Intercept > 0 ) + (u$b_liftedcase_Intercept < 0 ) + (u$b_liftedcop_Intercept < 0 ) + (u$b_liftedmark_Intercept < 0 ) + (u$b_nmod_Intercept < 0 ) + (u$b_nsubj_Intercept > 0 ) + (u$b_obl_Intercept < 0 ) + (u$b_xcomp_Intercept < 0 ))

library(ggplot2)

data = data.frame(satisfiedCount=satisfied)

library(dplyr)
library(tidyr)

data2 = data %>% group_by(satisfiedCount) %>% summarise(posterior = NROW(satisfiedCount)) %>% mutate(posteriorProb = posterior/nrow(data))


#plot = ggplot(data = data, aes(x=satisfiedCount)) + geom_histogram() + theme_bw() + xlim(0,10.5) + ggtitle("Dependency Length") + xlab("Number of Predicted Correlations") + ylab("Number of Posterior Samples")

plot = ggplot(data = data2, aes(x=satisfiedCount, y=posterior)) + geom_bar(stat="identity") + theme_bw() + xlim(0,10.5) + ggtitle("Dependency Length") + xlab("Number of Predicted Correlations") + ylab("Number of Posterior Samples") + geom_text(aes(label=posteriorProb), vjust=0) + ylim(0, 1.1*nrow(data))

ggsave(plot=plot, filename="figures/posterior-satisfied-universals-depl.pdf")



