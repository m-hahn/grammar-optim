library(dplyr)
library(tidyr)
library(ggplot2)




library(dplyr)
library(tidyr)
library(ggplot2)


cat("\nReading posterior samples\n")
#parse = read.csv("/home/user/CS_SCR/posteriors/posterior-10-parseability.csv")
u = read.csv("/home/user/CS_SCR/posteriors/posterior-10-efficiency-large.csv") %>% mutate(Type = "Efficiency")



u$satisfied = 8 - ((u$b_acl_Intercept < 0) + (u$b_aux_Intercept > 0 ) + (u$b_liftedcase_Intercept < 0 ) + (u$b_liftedcop_Intercept < 0 ) + (u$b_liftedmark_Intercept < 0 ) + (u$b_nmod_Intercept < 0 ) + (u$b_obl_Intercept < 0 ) + (u$b_xcomp_Intercept < 0 ))


library(ggplot2)

u$SamplesNum = NROW(u)
data2 = u %>% group_by(satisfied) %>% summarise(SamplesNum = mean(SamplesNum), posterior = NROW(satisfied)) %>% mutate(posteriorProb = posterior/SamplesNum)
u = NULL

#plot = ggplot(data = data, aes(x=satisfiedCount)) + geom_histogram() + theme_bw() + xlim(0,10.5) + ggtitle("Dependency Length") + xlab("Number of Predicted Correlations") + ylab("Number of Posterior Samples")



plot = ggplot(data = data2, aes(x=satisfied, y=posteriorProb)) + geom_bar(stat="identity") + theme_bw() 
plot = plot + xlim(0,8.5)
plot = plot + xlab("Number of Predicted Correlations") 
plot = plot + ylab("Posterior")
plot = plot + geom_text(aes(label=posteriorProb), vjust=0, size=3.3)
plot = plot + ylim(0, 1.1) 
plot = plot + theme(text = element_text(size=14),
        axis.text.x = element_text(angle=90, hjust=1)) 
plot = plot + theme(legend.position="none")

ggsave(plot=plot, filename="figures/posterior-satisfied-universals-efficiency-large.pdf", width=3, height=3)







