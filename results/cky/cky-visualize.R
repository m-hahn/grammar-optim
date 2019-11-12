best = read.csv("../strongest_models/best-two.csv")

library(dplyr)
library(tidyr)

data = read.csv("cky-summary.tsv", sep="\t")

data = merge(data, best %>% select(Language, Model) %>% rename(BestModel = Model), by=c("Language"))

# Select optimized models
data = data %>% filter(Directory != "manual_output_funchead_two_coarse_final" | Model == BestModel)
data = data %>% filter(Directory %in% c("manual_output_funchead_two_coarse_final", "manual_output_funchead_RANDOM"))

library(ggplot2)

plot = ggplot(data=data, aes(x=Loss, group=Directory, color=Directory, fill = Directory)) + 
      geom_histogram(binwidth=0.02)  + facet_grid(~Language) + theme_bw() + theme(legend.position="none")
ggsave(plot=plot, filename="cky-parse.pdf", height=3, width=8)


