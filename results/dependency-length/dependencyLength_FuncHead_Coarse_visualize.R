
data= read.csv("/home/user/CS_SCR/dependencyLength-intermediate.csv")

library(tidyr)
library(dplyr)
library(ggplot2)

plot = ggplot(data %>% filter(TypeN != "MLE"), aes(x=SentenceLength,y=DepLength,color=TypeN)) +
  geom_smooth(method = 'auto', se=F) +
  xlab("Sentence Length") +
  ylab("Dependency Length") +
       scale_color_brewer(palette="Set1") +
  theme_classic() + ylim(0,220) + xlim(0,50) +
  facet_wrap(~ Language, ncol=4)  +
  theme(legend.title=element_blank())


ggsave(plot=plot, filename="figures/depLength-facet.pdf", height=2.5, width=10)




