best = read.csv("../strongest_models/best-two-lambda09-best-balanced.csv")

library(dplyr)
library(tidyr)

data = read.csv("bigrams_results.tsv", sep="\t")

data = merge(data, best %>% select(Language, Model) %>% rename(BestModel = Model), by=c("Language"))

data$Directory = as.character(data$Directory)
#data[data$Model == "RANDOM",]$Directory = "manual_output_funchead_RANDOM"
data = data %>% filter(Directory == "manual_output_funchead_RANDOM" | Model == BestModel)
data = data %>% filter(Directory %in% c("manual_output_funchead_two_coarse_lambda09_best_balanced", "manual_output_funchead_RANDOM"))

library(ggplot2)

plot = ggplot(data=data, aes(x=Loss, group=Directory, color=Directory, fill = Directory)) + 
      geom_histogram(binwidth=0.01)  + facet_grid(~Language, , scales="free") + theme_bw() + theme(legend.position="none")

ggsave(plot=plot, filename="bigrams.pdf")


