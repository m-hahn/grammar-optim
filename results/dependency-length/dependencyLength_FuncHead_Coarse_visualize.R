

library(tidyr)
library(dplyr)
library(ggplot2)

data= read.csv("/home/user/CS_SCR/dependencyLength-intermediate.csv") %>% filter(TypeN != "MLE")


library(forcats)

data = data %>% mutate(TypeN = fct_recode(TypeN, "Parseability" = "Optimized for Pars.", "Predictability" = "Optimized for Pred.", "Efficiency" = "Optimized for Eff.", "Dependency Length" = "Optimized for Dep.L.", "Real Languages" = "Actual Language", "Baseline Languages" = "Baseline"))



data = data %>% mutate(TypeN = factor(TypeN, levels=c("Parseability", "Predictability", "Efficiency", "Dependency Length", "Real Languages", "Baseline Languages"), ordered=TRUE))
forColoring  = factor(data$TypeN, levels=c("Real Languages", "Baseline Languages", "Parseability", "Predictability", "Efficiency", "Dependency Length"), ordered=TRUE)
#Create a custom color scale
library(RColorBrewer)
myColors <- brewer.pal(6,"Set1")
names(myColors) <- levels(forColoring)



plot = ggplot(data %>% filter(TypeN != "MLE"), aes(x=SentenceLength,y=DepLength,color=TypeN)) +
  geom_smooth(method = 'auto', se=F) +
  xlab("Sentence Length") +
  ylab("Dependency Length") +
       scale_colour_manual(name = "TypeN",values = myColors) +
  theme_classic() + ylim(0,220) + xlim(0,50) +
  facet_wrap(~ Language, ncol=4)  +
  theme(legend.title=element_blank())


ggsave(plot=plot, filename="figures/depLength-facet.pdf", height=2.5, width=10)





plot = ggplot(data %>% filter(TypeN != "MLE", TypeN %in% c("Baseline Languages", "Dependency Length")), aes(x=SentenceLength,y=DepLength,color=TypeN)) +
  geom_smooth(method = 'auto', se=F) +
  xlab("Sentence Length") +
  ylab("Dependency Length") +
       scale_colour_manual(name = "TypeN",values = myColors) +
  theme_classic() + ylim(0,220) + xlim(0,50) +
  facet_wrap(~ Language, ncol=4)  +
  theme(legend.title=element_blank())


ggsave(plot=plot, filename="figures/depLength-facet-1.png", height=2.5, width=10)




plot = ggplot(data %>% filter(TypeN != "MLE", TypeN %in% c("Baseline Languages", "Dependency Length", "Real Languages")), aes(x=SentenceLength,y=DepLength,color=TypeN)) +
  geom_smooth(method = 'auto', se=F) +
  xlab("Sentence Length") +
  ylab("Dependency Length") +
       scale_colour_manual(name = "TypeN",values = myColors) +
  theme_classic() + ylim(0,220) + xlim(0,50) +
  facet_wrap(~ Language, ncol=4)  +
  theme(legend.title=element_blank())


ggsave(plot=plot, filename="figures/depLength-facet-2.png", height=2.5, width=10)




plot = ggplot(data %>% filter(TypeN != "MLE"), aes(x=SentenceLength,y=DepLength,color=TypeN)) +
  geom_smooth(method = 'auto', se=F) +
  xlab("Sentence Length") +
  ylab("Dependency Length") +
       scale_colour_manual(name = "TypeN",values = myColors) +
  theme_classic() + ylim(0,220) + xlim(0,50) +
  facet_wrap(~ Language, ncol=4)  +
  theme(legend.title=element_blank())


ggsave(plot=plot, filename="figures/depLength-facet-3.png", height=2.5, width=10)




