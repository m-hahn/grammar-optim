library(dplyr)
library(tidyr)



dataS = read.csv("adversarial-lm.tsv", sep="\t")

library(ggplot2)




library(forcats)
dataS = dataS %>% mutate(Model = fct_recode(Model, "Baseline" = "manual_output_funchead_RANDOM", "Optimized" = "manual_output_funchead_two_coarse_lambda09_best_balanced"))
dataS = dataS %>% filter(Model %in% c("Baseline", "Optimized"))
dataS = dataS %>% group_by(Language, Model, FileName, Type) %>% summarise(LM_Loss = mean(LM_Loss))


plot = ggplot(data=dataS %>% filter(Language == "English"), aes(x=LM_Loss, group=Model, color=Model, fill = Model)) + 
      geom_histogram(binwidth=0.01) + facet_wrap(~Type, ncol=3, scales = "free") + theme_bw()


#ggsave(plot=plot, filename="adversarial-parse-english.pdf", height=6, width=10)



plot = ggplot(data=dataS %>% filter(Language == "Japanese"), aes(x=LM_Loss, group=Model, color=Model, fill = Model)) + 
      geom_histogram(binwidth=0.01) + facet_wrap(~Type, ncol=3, scales = "free") + theme_bw()

#ggsave(plot=plot, filename="adversarial-parse-japanese.pdf", height=6, width=10)

